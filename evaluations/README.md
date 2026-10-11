# Synthetic RAG evaluation

Recorded: 2026-10-10 15:50 PDT. Full answers, passages, distances and timings are in
[latest.json](latest.json); the source text and questions are in [cases.json](cases.json).

| Check | Result |
|---|---|
| Answer checks | 20/20 |
| Supported answers and expected evidence | 12/12 answers; 12/12 evidence |
| Unsupported-question abstentions | 8/8 |
| Matching query logs with distance and latency | 20/20 |
| Median / maximum request latency | 1,104 / 4,074 ms |

The run uses live OpenAI calls (`text-embedding-3-small`, `gpt-4.1-mini`),
PostgreSQL/pgvector and the Django ingestion/question handlers locally. One
synthetic museum document produces 4 overlapping chunks; retrieval uses top-k 3
and a maximum cosine distance of 0.95. It does not exercise the public web proxy.

Supported answers must match expected-fact patterns; retrieved passages must
separately match evidence patterns. Unsupported questions must return “I don't
know.” after normalization. Every request must succeed and have a matching log.
The recorded answers were also read and checked against the fixture. The report
includes the fixture's SHA-256 hash. This small smoke evaluation does not establish
general accuracy; model outputs and latency can vary.

Run `python manage.py evaluate_rag` to regenerate the report (API charges apply).
Temporary evaluation users, documents and query logs are rolled back afterward.

API-failure logging is verified separately by automated tests with simulated
embedding connection errors, answer connection errors and answer rate limits.
Answer failures preserve retrieved passages/distances, elapsed latency and the
error entry; these simulations do not count toward the 20 live answer checks.
