from django.views.decorators.http import require_GET
import json, time
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from pgvector.django import CosineDistance
from .models import Chunk, Document, QueryLog
from django.shortcuts import render
from pypdf import PdfReader
import re

client = OpenAI()

@csrf_exempt
def retrieve(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    body = json.loads(request.body.decode("utf-8"))
    query = body.get("query", "")
    k = int(body.get("k", 5))

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
def ask(request):
    t0 = time.perf_counter()
    log = None

    try:
        if request.method != "POST":
            return JsonResponse({"error": "POST only"}, status=405)

        body = json.loads(request.body.decode("utf-8"))
        question = (body.get("question") or "").strip()
        k = int(body.get("k", 5))

        if not question:
            return JsonResponse({"error": "question is required"}, status=400)

        # Determine doc intent (optional fallback to latest doc)
        q = question.lower()
        doc_intent = any(p in q for p in [
            "summarize", "summary", "this pdf", "the pdf", "this document", "the document"
        ])

        # Priority: body doc_id > session doc_id > latest doc (if doc_intent)
        raw_doc_id = body.get("document_id")
        session_doc_id = request.session.get("current_document_id")

        effective_document_id = None

        if raw_doc_id not in (None, "", 0):
            try:
                effective_document_id = int(raw_doc_id)
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

        scoped = True
        max_distance = float(body.get("max_distance", 0.95))  # scoped default

        best = chunks[0] if chunks else None
        best_distance = float(best.distance) if best else None

        # log early
        log = QueryLog.objects.create(
            question=question,
            k=k,
            document_id=effective_document_id,
            max_distance=max_distance,
            best_distance=best_distance,
        )

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
        return JsonResponse({"error": "internal_error", "details": repr(e)}, status=500)
'''
def simple_chunk(text: str, max_chars: int = 900):
    """Split text into roughly max_chars chunks (v1)."""
    text = (text or "").strip()
    if not text:
        return[]
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars].strip()
        if chunk:
            chunks.append(chunk)
        i += max_chars
    return chunks
'''

def chunk_text(text: str, max_chars: int = 900, overlap: int = 200):
    """
    v2 chunker:
    - splits on sentences/paragraphs
    - packs into chunks up to max_chars
    - overlaps last 'overlap' chars between chunks
    """
    text = (text or "").strip()
    if not text:
        return []

    # Split into sentence-ish units (simple, good enough for v2)
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+|\n+', text) if p.strip()]

    chunks = []
    buf = ""

    for p in parts:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf = f"{buf} {p}"
        else:
            chunks.append(buf.strip())
            # start next buffer with overlap from previous chunk
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = f"{tail} {p}".strip()

    if buf.strip():
        chunks.append(buf.strip())

    return chunks

@csrf_exempt
def ingest_text(request):
    try:
        if request.method != "POST":
            return JsonResponse({"error": "POST only"}, status=405)

        # Parse JSON safely
        try:
            body = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"error": "invalid_json"}, status=400)

        title = (body.get("title") or "Untitled").strip()
        text = body.get("text") or ""

        parts = chunk_text(text)
        if not parts:
            return JsonResponse({"error": "No text to ingest"}, status=400)

        doc, created = Document.objects.get_or_create(title=title, source="ingested_text")

        # Store selected/current doc in session
        request.session["current_document_id"] = doc.id
        request.session.modified = True

        # If doc already exists, wipe old chunks so this is an "update"
        if not created:
            Chunk.objects.filter(document=doc).delete()

        # Embed in one call (cheaper/faster than one-by-one)
        embs = client.embeddings.create(
            model="text-embedding-3-small",
            input=parts,
        ).data

        for i, (chunk_str, item) in enumerate(zip(parts, embs)):
            Chunk.objects.create(
                document=doc,
                chunk_index=i,
                text=chunk_str,
                embedding=item.embedding,
            )

        return JsonResponse({
            "document_id": doc.id,
            "chunks_created": len(parts),
            "title": doc.title,
            "status": "created" if created else "updated",
            "current_document_id": doc.id,
        })

    except Exception as e:
        # Always return JSON even in DEBUG, so your frontend doesn't get HTML
        return JsonResponse({"error": "internal_error", "details": repr(e)}, status=500)
@require_GET
def logs(request):
    limit = int(request.GET.get("limit", 20))
    limit = max(1, min(limit, 100)) 

    rows = QueryLog.objects.order_by("-id")[:limit]

    return JsonResponse({
        "count": rows.count(),
        "logs": [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "question": r.question,
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

@csrf_exempt
def ingest_pdf(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    if "file" not in request.FILES:
        return JsonResponse({"error": "Missing file field"}, status=400)

    uploaded = request.FILES["file"]
    title = (request.POST.get("title") or uploaded.name or "Untitled").strip()

    reader = PdfReader(uploaded)
    text = "\n".join([(page.extract_text() or "") for page in reader.pages]).strip()

    if not text:
        return JsonResponse({"error": "Could not extract text from PDF"}, status=400)

    # Reuse your existing ingestion logic
    parts = chunk_text(text)
    if not parts:
        return JsonResponse({"error": "No text to ingest"}, status=400)

    doc, created = Document.objects.get_or_create(title=title, source="pdf")

    request.session["current_document_id"] = doc.id
    request.session.modified = True

    if not created:
        Chunk.objects.filter(document=doc).delete()

    embs = client.embeddings.create(
        model="text-embedding-3-small",
        input=parts,
    ).data

    for i, (chunk_str, item) in enumerate(zip(parts, embs)):
        Chunk.objects.create(
            document=doc,
            chunk_index=i,
            text=chunk_str,
            embedding=item.embedding,
        )

    return JsonResponse({
        "document_id": doc.id,
        "title": doc.title,
        "chunks_created": len(parts),
        "status": "created" if created else "updated",
        "current_document_id": request.session["current_document_id"],
    })

@csrf_exempt
def documents(request):
    if request.method != "GET":
        return JsonResponse({"error": "GET only"}, status=405)

    limit = int(request.GET.get("limit", 20))
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
    })

@csrf_exempt
def select_document(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    
    try:
        body = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid_json"}, status=400)

    body = json.loads(request.body.decode("utf-8"))
    doc_id = body.get("document_id")

    if not doc_id:
        return JsonResponse({"error": "document_id is required"}, status=400)
    try:
        doc_id = int(doc_id)
    except (TypeError, ValueError):
        return JsonResponse({"error": "document_id must be an integer"}, status=400)
    
    #validate document exists
    if not Document.objects.filter(id=int(doc_id)).exists():
        return JsonResponse({"error": "Document not found"}, status=404)
    
    request.session["current_document_id"] = int(doc_id)
    return JsonResponse({"current_document_id": int(doc_id), "status": "ok"})
@csrf_exempt
def ingest_and_ask_text(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    body = json.loads(request.body.decode("utf-8"))
    title = (body.get("title") or "Untitled").strip()
    text = body.get("text") or ""
    question = (body.get("question") or "").strip()
    k = int(body.get("k", 5))

    if not text.strip():
        return JsonResponse({"error": "text is required"}, status=400)
    if not question:
        return JsonResponse({"error": "question is required"}, status=400)

    # 1) Ingest (same as ingest_text)
    parts = chunk_text(text)
    if not parts:
        return JsonResponse({"error": "No text to ingest"}, status=400)

    doc, created = Document.objects.get_or_create(title=title, source="ingested_text")
    if not created:
        Chunk.objects.filter(document=doc).delete()

    embs = client.embeddings.create(
        model="text-embedding-3-small",
        input=parts,
    ).data

    for i, (chunk_str, item) in enumerate(zip(parts, embs)):
        Chunk.objects.create(
            document=doc,
            chunk_index=i,
            text=chunk_str,
            embedding=item.embedding,
        )

    # Set session doc
    request.session["current_document_id"] = doc.id
    request.session.modified = True

    # 2) Ask (scoped to that doc)
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    ).data[0].embedding

    qs = Chunk.objects.exclude(embedding=None).filter(document_id=doc.id)
    chunks = (
        qs.annotate(distance=CosineDistance("embedding", q_emb))
          .order_by("distance")[:k]
    )

    max_distance = float(body.get("max_distance", 0.95))
    best = chunks[0] if chunks else None
    if not best or float(best.distance) > max_distance:
        return JsonResponse({
            "document_id": doc.id,
            "status": "created" if created else "updated",
            "answer": "I don't know.",
            "sources": [],
        })

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

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": "Answer using ONLY the provided sources. If the sources don't contain the answer, say: I don't know."},
            {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}"},
        ],
    )

    return JsonResponse({
        "document_id": doc.id,
        "title": doc.title,
        "status": "created" if created else "updated",
        "current_document_id": request.session["current_document_id"],
        "question": question,
        "answer": resp.output_text,
        "sources": sources,
    })

@csrf_exempt
def ingest_and_ask_pdf(request):
    t0 = time.perf_counter()
    log = None

    try:
        if request.method != "POST":
            return JsonResponse({"error": "POST only"}, status=405)

        # multipart fields
        if "file" not in request.FILES:
            return JsonResponse({"error": "Missing file field"}, status=400)

        uploaded = request.FILES["file"]
        title = (request.POST.get("title") or uploaded.name or "Untitled").strip()

        question = (request.POST.get("question") or "").strip()
        if not question:
            return JsonResponse({"error": "question is required"}, status=400)

        k = int(request.POST.get("k") or 5)

        # 1) Extract text from PDF
        reader = PdfReader(uploaded)
        text = "\n".join([(page.extract_text() or "") for page in reader.pages]).strip()
        if not text:
            return JsonResponse({"error": "Could not extract text from PDF"}, status=400)

        # 2) Chunk + upsert document
        parts = chunk_text(text)
        if not parts:
            return JsonResponse({"error": "No text to ingest"}, status=400)

        doc, created = Document.objects.get_or_create(title=title, source="pdf")

        # Set "current document" in session so future /ask calls are user-friendly
        request.session["current_document_id"] = doc.id
        request.session.modified = True

        if not created:
            Chunk.objects.filter(document=doc).delete()

        # 3) Embed chunks and save
        embs = client.embeddings.create(
            model="text-embedding-3-small",
            input=parts,
        ).data

        for i, (chunk_str, item) in enumerate(zip(parts, embs)):
            Chunk.objects.create(
                document=doc,
                chunk_index=i,
                text=chunk_str,
                embedding=item.embedding,
            )

        # 4) Answer immediately (scoped to this doc)
        q_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=question,
        ).data[0].embedding

        qs = Chunk.objects.exclude(embedding=None).filter(document_id=doc.id)

        chunks = (
            qs.annotate(distance=CosineDistance("embedding", q_emb))
              .order_by("distance")[:k]
        )

        # guardrail (scoped doc => higher default)
        max_distance = float(request.POST.get("max_distance") or 0.95)

        best = chunks[0] if chunks else None
        best_distance = float(best.distance) if best else None

        log = QueryLog.objects.create(
            question=question,
            k=k,
            document_id=doc.id,
            max_distance=max_distance,
            best_distance=best_distance,
        )

        if not best or float(best.distance) > max_distance:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.answer = "I don't know."
            log.sources = []
            log.latency_ms = latency_ms
            log.save(update_fields=["answer", "sources", "latency_ms"])
            return JsonResponse({
                "document_id": doc.id,
                "title": doc.title,
                "status": "created" if created else "updated",
                "current_document_id": request.session["current_document_id"],
                "question": question,
                "answer": "I don't know.",
                "sources": [],
            })

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

        return JsonResponse({
            "document_id": doc.id,
            "title": doc.title,
            "status": "created" if created else "updated",
            "current_document_id": request.session["current_document_id"],
            "question": question,
            "answer": answer,
            "sources": sources,
        })

    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if log:
            log.error = repr(e)
            log.latency_ms = latency_ms
            log.save(update_fields=["error", "latency_ms"])
        return JsonResponse({"error": "internal_error", "details": repr(e)}, status=500)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def ingest_file(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    if "file" not in request.FILES:
        return JsonResponse({"error": "Missing file field"}, status=400)

    uploaded = request.FILES["file"]
    title = (request.POST.get("title") or uploaded.name or "Untitled").strip()

    # Basic type check (MVP)
    filename = (uploaded.name or "").lower()
    if not (filename.endswith(".txt") or filename.endswith(".md")):
        return JsonResponse({"error": "Only .txt or .md supported"}, status=400)

    # Read bytes -> text
    try:
        raw = uploaded.read()
        text = raw.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        return JsonResponse({"error": "Could not read file", "details": repr(e)}, status=400)

    if not text:
        return JsonResponse({"error": "Empty file"}, status=400)

    # Reuse your existing ingest_text logic (copy/paste or refactor into helper)
    parts = chunk_text(text)
    if not parts:
        return JsonResponse({"error": "No text to ingest"}, status=400)

    doc, created = Document.objects.get_or_create(title=title, source="text_file")

    request.session["current_document_id"] = doc.id
    request.session.modified = True

    if not created:
        Chunk.objects.filter(document=doc).delete()

    embs = client.embeddings.create(
        model="text-embedding-3-small",
        input=parts,
    ).data

    for i, (chunk_str, item) in enumerate(zip(parts, embs)):
        Chunk.objects.create(
            document=doc,
            chunk_index=i,
            text=chunk_str,
            embedding=item.embedding,
        )

    return JsonResponse({
        "document_id": doc.id,
        "title": doc.title,
        "chunks_created": len(parts),
        "status": "created" if created else "updated",
        "current_document_id": request.session["current_document_id"],
    })

@csrf_exempt
def clear_selected_document(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    request.session.pop("current_document_id", None)
    request.session.modified = True
    return JsonResponse({"ok": True, "current_document_id": None})

def app(request):
    return render(request, "app.html")