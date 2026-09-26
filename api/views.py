from django.views.decorators.http import require_GET
import json
import time
import logging
import math
from functools import wraps
from django.db import transaction
from django.core.exceptions import RequestDataTooBig
from openai import OpenAIError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from pgvector.django import CosineDistance
from .models import Chunk, Document, QueryLog
from django.shortcuts import render
from pypdf import PdfReader
import re
from django.conf import settings

client = OpenAI(timeout=45.0, max_retries=0)
logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


class InvalidInput(ValueError):
    pass


def api_errors(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except RequestDataTooBig:
            return JsonResponse({"error": "invalid_input", "message": "The request is too large. Text and files must be at most 2 MB."}, status=400)
        except InvalidInput as exc:
            return JsonResponse({"error": "invalid_input", "message": str(exc)}, status=400)
        except OpenAIError:
            logger.exception("AI request failed")
            return JsonResponse({"error": "ai_unavailable", "message": "The AI service is unavailable. Please try again."}, status=502)
        except Exception:
            logger.exception("API request failed")
            return JsonResponse({"error": "internal_error", "message": "Something went wrong. Please try again."}, status=500)
    return wrapped


def json_body(request):
    try:
        body = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise InvalidInput("Send a valid JSON object.")
    if not isinstance(body, dict):
        raise InvalidInput("Send a JSON object, not a list or scalar.")
    return body


def string_value(value, name, default="", max_length=None):
    if value is None:
        value = default
    if not isinstance(value, str):
        raise InvalidInput(f"{name} must be text.")
    value = value.strip()
    if max_length is not None and len(value) > max_length:
        raise InvalidInput(f"{name} must be at most {max_length} characters.")
    return value


def integer_value(value, name, minimum=1, maximum=20):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise InvalidInput(f"{name} must be an integer between {minimum} and {maximum}.")
    if len(str(value)) > len(str(maximum)):
        raise InvalidInput(f"{name} must be between {minimum} and {maximum}.")
    result = int(value)
    if not minimum <= result <= maximum:
        raise InvalidInput(f"{name} must be between {minimum} and {maximum}.")
    return result


def distance_value(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise InvalidInput("max_distance must be a number between 0 and 2.")
    if isinstance(value, bool) or not math.isfinite(result) or not 0 <= result <= 2:
        raise InvalidInput("max_distance must be a number between 0 and 2.")
    return result


@csrf_exempt
@api_errors
def retrieve(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    body = json_body(request)
    query = string_value(body.get("query"), "query", max_length=10000)
    if not query:
        raise InvalidInput("Enter a search query.")
    k = integer_value(body.get("k", 5), "k")

    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    ).data[0].embedding

    chunks = (
        Chunk.objects
        .exclude(embedding=None)
        .annotate(distance=CosineDistance("embedding", q_emb))
        .order_by("distance")[:k]
    )

    return JsonResponse({
        "query": query,
        "results": [
            {
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "distance": float(c.distance),
            }
            for c in chunks
        ]
    })

@csrf_exempt
@api_errors
def ask(request):
    t0 = time.perf_counter()
    log = None

    try:
        if request.method != "POST":
            return JsonResponse({"error": "POST only"}, status=405)

        body = json_body(request)
        question = string_value(body.get("question"), "question", max_length=10000)
        k = integer_value(body.get("k", 5), "k")

        if not question:
            return JsonResponse({"error": "question is required"}, status=400)

        max_distance = distance_value(body.get("max_distance", 0.95))

        # Determining doc intent 
        q = question.lower()
        doc_intent = any(p in q for p in [
            "summarize", "summary", "this pdf", "the pdf", "this document", "the document"
        ])

        # Priority: body doc_id > session doc_id > latest doc (if doc_intent)
        raw_doc_id = body.get("document_id")
        session_doc_id = request.session.get("current_document_id")

        effective_document_id = None

        if raw_doc_id not in (None, ""):
            try:
                effective_document_id = integer_value(raw_doc_id, "document_id", maximum=9223372036854775807)
            except (TypeError, ValueError):
                return JsonResponse({"error": "document_id must be an integer"}, status=400)
        elif session_doc_id:
            effective_document_id = int(session_doc_id)
        elif doc_intent:
            latest_doc = Document.objects.order_by("-id").first()
            if latest_doc:
                effective_document_id = latest_doc.id

        if effective_document_id is None:
            return JsonResponse(
                {"error": "no_document_selected", "message": "Select or ingest a document first."},
                status=400
            )

        if not Document.objects.filter(id=effective_document_id).exists():
            return JsonResponse({"error": "document_not_found", "message": "Select an existing document."}, status=404)

        log = QueryLog.objects.create(
            question=question, k=k, document_id=effective_document_id,
            max_distance=max_distance,
        )

        # 1) embed question
        q_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=question,
        ).data[0].embedding

        # 2) retrieve top-k (scoped)
        qs = Chunk.objects.exclude(embedding=None).filter(document_id=effective_document_id)
        chunks = (
            qs.annotate(distance=CosineDistance("embedding", q_emb))
              .order_by("distance")[:k]
        )


        best = chunks[0] if chunks else None
        best_distance = float(best.distance) if best else None

        log.best_distance = best_distance
        log.save(update_fields=["best_distance"])

        if not best or float(best.distance) > max_distance:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.answer = "I don't know."
            log.sources = []
            log.latency_ms = latency_ms
            log.save(update_fields=["answer", "sources", "latency_ms"])
            return JsonResponse({"answer": "I don't know.", "sources": []})

        sources = [
            {
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "distance": float(c.distance),
            }
            for c in chunks
        ]
        context = "\n\n".join([f"[source {i+1}] {c.text}" for i, c in enumerate(chunks)])

        # 3) answer grounded in sources
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "Answer using ONLY the provided sources. If the sources don't contain the answer, say: I don't know."},
                {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}"},
            ],
        )

        answer = resp.output_text
        latency_ms = int((time.perf_counter() - t0) * 1000)

        log.answer = answer
        log.sources = sources
        log.latency_ms = latency_ms
        log.save(update_fields=["answer", "sources", "latency_ms"])

        return JsonResponse({"question": question, "answer": answer, "sources": sources})

    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if log:
            log.error = repr(e)
            log.latency_ms = latency_ms
            log.save(update_fields=["error", "latency_ms"])
        raise

def chunk_text(text: str, max_chars: int = 900, overlap: int = 200):
    """Bound every chunk; prefer sentence/word boundaries and retain up to overlap characters."""
    if max_chars < 1 or not 0 <= overlap < max_chars:
        raise ValueError("Require max_chars > 0 and 0 <= overlap < max_chars.")
    text = re.sub(r"\s+", " ", (text or "").strip())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            segment = text[start:end]
            boundaries = [m.end() for m in re.finditer(r"[.!?](?=\s)", segment)]
            candidates = [n for n in boundaries if n > overlap]
            if candidates:
                end = start + candidates[-1]
            else:
                space = segment.rfind(" ")
                if space > overlap:
                    end = start + space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def store_document(request, title, text, source):
    title = string_value(title, "title", default="Untitled", max_length=255) or "Untitled"
    text = string_value(text, "text")
    if len(text.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise InvalidInput("Text must be at most 2 MB.")
    parts = chunk_text(text)
    if not parts:
        raise InvalidInput("Add some text before uploading.")

    # Finish every external call before changing the stored document.
    embeddings = []
    for offset in range(0, len(parts), 64):
        batch = parts[offset:offset + 64]
        items = client.embeddings.create(model="text-embedding-3-small", input=batch).data
        if len(items) != len(batch):
            raise RuntimeError("Incomplete embedding response")
        embeddings.extend(item.embedding for item in items)

    with transaction.atomic():
        doc, created = Document.objects.select_for_update().get_or_create(title=title, source=source)
        Chunk.objects.filter(document=doc).delete()
        Chunk.objects.bulk_create([
            Chunk(document=doc, chunk_index=i, text=part, embedding=embedding)
            for i, (part, embedding) in enumerate(zip(parts, embeddings))
        ])

    request.session["current_document_id"] = doc.id
    return JsonResponse({
        "document_id": doc.id, "title": doc.title, "chunks_created": len(parts),
        "status": "created" if created else "updated", "current_document_id": doc.id,
    })


def upload(request):
    uploaded = request.FILES.get("file")
    if uploaded is None:
        raise InvalidInput("Choose a file first.")
    if uploaded.size > MAX_UPLOAD_BYTES:
        raise InvalidInput("Files must be at most 2 MB.")
    return uploaded


@csrf_exempt
@api_errors
def ingest_text(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    body = json_body(request)
    return store_document(request, body.get("title"), body.get("text"), "ingested_text")


@csrf_exempt
@api_errors
def ingest_pdf(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    uploaded = upload(request)
    try:
        reader = PdfReader(uploaded)
        if reader.is_encrypted:
            raise InvalidInput("Password-protected PDFs are not supported.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except InvalidInput:
        raise
    except Exception:
        raise InvalidInput("This PDF could not be read. Choose a valid, text-based PDF.")
    if not text:
        raise InvalidInput("No text found. Scanned PDFs need OCR before uploading.")
    return store_document(request, request.POST.get("title") or uploaded.name, text, "pdf")


@csrf_exempt
@api_errors
def ingest_file(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    uploaded = upload(request)
    if not uploaded.name.lower().endswith((".txt", ".md")):
        raise InvalidInput("Choose a .txt or .md file.")
    try:
        text = uploaded.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InvalidInput("Save this file as UTF-8 text and try again.")
    return store_document(request, request.POST.get("title") or uploaded.name, text, "text_file")


@csrf_exempt
@api_errors
def documents(request):
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)

    limit = integer_value(request.GET.get("limit", 20), "limit", maximum=100)
    docs = Document.objects.order_by("-id")[:limit]

    return JsonResponse({
        "count": docs.count(),
        "documents": [
            {
                "id": d.id,
                "title": d.title,
                "source": d.source,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ],
        "current_document_id": request.session.get("current_document_id"),
        "current_document_title": Document.objects.filter(id=request.session.get("current_document_id")).values_list("title", flat=True).first(),
    })

@csrf_exempt
@api_errors
def select_document(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    
    body = json_body(request)
    
    doc_id = body.get("document_id")

    if not doc_id:
        return JsonResponse({"error": "document_id is required"}, status=400)
    try:
        doc_id = integer_value(doc_id, "document_id", maximum=9223372036854775807)
    except (TypeError, ValueError):
        return JsonResponse({"error": "document_id must be an integer"}, status=400)
    
    #validating document exists
    if not Document.objects.filter(id=doc_id).exists():
        return JsonResponse({"error": "Document not found"}, status=404)
    
    request.session["current_document_id"] = doc_id
    return JsonResponse({"current_document_id": doc_id, "status": "ok"})

@csrf_exempt
@api_errors
def clear_selected_document(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    request.session.pop("current_document_id", None)
    request.session.modified = True
    return JsonResponse({"ok": True, "current_document_id": None})

def app(request):
    return render(request, "app.html")

@csrf_exempt
@api_errors
def reset_data(request):
    """
    DEV ONLY: wipes all Documents, Chunks, and QueryLogs.
    Requires DEBUG=True and a confirmation string in the request body.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    if not getattr(settings, "DEBUG", False):
        return JsonResponse({"error": "forbidden"}, status=403)

    body = json_body(request)

    # require explicit confirmation to avoid accidental wipes
    if body.get("confirm") != "RESET":
        return JsonResponse(
            {"error": "confirm_required", "message": 'Send {"confirm":"RESET"} to wipe all data.'},
            status=400,
        )

    chunks_deleted, _ = Chunk.objects.all().delete()
    docs_deleted, _ = Document.objects.all().delete()
    logs_deleted, _ = QueryLog.objects.all().delete()

    request.session.pop("current_document_id", None)
    request.session.modified = True

    return JsonResponse({
        "ok": True,
        "deleted": {
            "chunks": chunks_deleted,
            "documents": docs_deleted,
            "query_logs": logs_deleted,
        },
        "current_document_id": None,
    })

@require_GET
@api_errors
def logs(request):
    limit = integer_value(request.GET.get("limit", 20), "limit", maximum=100)

    rows = QueryLog.objects.order_by("-id")[:limit]

    return JsonResponse({
        "count": rows.count(),
        "logs": [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "question": r.question,
                "answer": r.answer,
                "status": "error" if r.error else ("unanswered" if r.answer.strip().lower().replace("’", "'").rstrip(".") == "i don't know" else "answered"),
                "k": r.k,
                "document_id": r.document_id,
                "max_distance": r.max_distance,
                "best_distance": r.best_distance,
                "error": r.error,
                "latency_ms": r.latency_ms,
            }
            for r in rows
        ]
    })