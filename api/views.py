from django.shortcuts import render
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from pgvector.django import CosineDistance
from .models import Chunk, Document
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
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    body = json.loads(request.body.decode("utf-8"))
    question = body.get("question", "")
    k = int(body.get("k", 5))
    document_id = body.get("document_id")

    if document_id not in (None, "", 0):
        qs = qs.filter(document_id=int(document_id))

    if not question.strip():
        return JsonResponse({"error": "question is required"}, status=400)

    # 1) embed question
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    ).data[0].embedding

    # 2) retrieve top-k chunks
    qs = Chunk.objects.exclude(embedding=None)

    if document_id:
        qs = qs.filter(document_id=document_id)

    chunks = (
        qs.annotate(distance=CosineDistance("embedding", q_emb))
        .order_by("distance")[:k]
    )

    if not chunks:
        return JsonResponse({"answer": "I don't know.", "sources": []})

    context = "\n\n".join([f"[source {i+1}] {c.text}" for i, c in enumerate(chunks)])

    # 3) generate answer grounded in context
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": "Answer using ONLY the provided sources. If the sources don't contain the answer, say: I don't know."},
            {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}"},
        ],
    )

    return JsonResponse({
        "question": question,
        "answer": resp.output_text,
        "sources": [
            {
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "distance": float(c.distance),
            }
            for c in chunks
        ],
    })

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

    parts = simple_chunk(text)
    if not parts:
        return JsonResponse({"error": "No text to ingest"}, status=400)

    doc = Document.objects.create(title=title, source="ingest_text")

    # Embed in one call (cheaper/faster than one-by-one)
    embs = client.embeddings.create(
        model="text-embedding-3-small",
        input=parts,
    ).data

    for i, (chunk_text, item) in enumerate(zip(parts, embs)):
        Chunk.objects.create(
            document=doc,
            chunk_index=i,
            text=chunk_text,
            embedding=item.embedding,
        )

    return JsonResponse({
        "document_id": doc.id,
        "chunks_created": len(parts),
        "title": doc.title,
    })