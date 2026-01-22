from django.shortcuts import render
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from pgvector.django import CosineDistance
from .models import Chunk

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

    # 1) embed question
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    ).data[0].embedding

    # 2) retrieve top-k chunks
    chunks = (
        Chunk.objects
        .exclude(embedding=None)
        .annotate(distance=CosineDistance("embedding", q_emb))
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
