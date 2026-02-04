from django.views.decorators.http import require_GET
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from pgvector.django import CosineDistance
from .models import Chunk, Document, QueryLog
import re
import time
from pypdf import PdfReader

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
        question = body.get("question", "")
        k = int(body.get("k", 5))
        document_id = body.get("document_id")

        if not question.strip():
            return JsonResponse({"error": "question is required"}, status=400)

        # 1) embed question
        q_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=question,
        ).data[0].embedding

        # 2) retrieve top-k chunks
        qs = Chunk.objects.exclude(embedding=None)

        if document_id not in (None, "", 0):
            qs = qs.filter(document_id=int(document_id))

        chunks = (
            qs.annotate(distance=CosineDistance("embedding", q_emb))
            .order_by("distance")[:k]
        )

        # guardrail: if best match is too weak, refuse
        max_distance = float(body.get("max_distance", 0.75))  # tune later

        best = chunks[0] if chunks else None
        best_distance = float(best.distance) if best else None

        # Create log early (so we can update it regardless of outcome)
        log = QueryLog.objects.create(
            question=question,
            k=k,
            document_id=int(document_id) if document_id not in (None, "", 0) else None,
            max_distance=max_distance,
            best_distance=best_distance,
        )

        # guardrail
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

        # 3) generate answer grounded in context
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
            "question": question,
            "answer": answer,
            "sources": sources,
        })
    
    except Exception as e:
        # If anything breaks, record it
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if log:
            log.error = repr(e)
            log.latency_ms = latency_ms
            log.save(update_fields=["error", "latency_ms"])
            return JsonResponse({"error": "internal_error"}, status=500)
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
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    body = json.loads(request.body.decode("utf-8"))
    title = (body.get("title") or "Untitled").strip()
    text = body.get("text") or ""

    parts = chunk_text(text)
    if not parts:
        return JsonResponse({"error": "No text to ingest"}, status=400)

    doc, created = Document.objects.get_or_create(title=title, source="ingested_text")
    
    #IF doc already exists, wipe old chunks so this is an "update"
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
    })

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
    })