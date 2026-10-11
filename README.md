# RAG Knowledge Base

A deployed document Q&A app built with Django, PostgreSQL/pgvector, and the OpenAI API.
Upload a document, ask a question, and inspect the retrieved passages behind the answer.

**[Live demo](https://77.113.94.93)** — choose **Try the demo**; no signup required.
Each visitor gets a separate temporary workspace with sample documents and uploads.
Workspaces expire after 24 hours, and request/upload limits bound demo usage.

## What it demonstrates

- PDF, text/Markdown, JSON, and CSV ingestion with validation, overlapping chunks, and atomic document replacement.
- Vector retrieval with `text-embedding-3-small`, grounded answers with `gpt-4.1-mini`, and unsupported-question abstention.
- Query history with source passages, retrieval distances, latency, and API-failure diagnostics.
- Authenticated owner access and a session-isolated public demo.
- Linux deployment on AWS Lightsail using Docker Compose, Nginx, Gunicorn, HTTPS, and automatic certificate renewal.
- GitHub Actions tests against PostgreSQL/pgvector, builds an image, and deploys passing `main` updates with health checks and rollback.

Flow: document → overlapping chunks → embeddings in pgvector → nearest passages → answer with sources.

## Evidence

The October 10, 2026 live synthetic evaluation passed **20/20 answer checks**, including
**8/8 unsupported-question abstentions**. All 12 supported questions retrieved the
expected evidence, and all 20 distance/latency logging checks passed.

See the [rubric and results](evaluations/README.md), [full recorded answers](evaluations/latest.json),
and [GitHub Actions runs](https://github.com/LuSilverX/Rag-Chatbot/actions).
This small synthetic evaluation is a smoke check, not a general accuracy benchmark.
API connection and rate-limit failures are tested separately with simulated errors.

## Run locally

Requires Docker Compose and an OpenAI API key. Copy the configuration, then fill in
your API key, a random Django secret, and a database password; keep this file private.

```bash
cp deploy/container.env.example .env.container
# Edit .env.container before continuing.
sh deploy/compose.sh build web
sh deploy/compose.sh up -d --wait
sh deploy/compose.sh exec web python manage.py prepare_demo
```

Open **http://localhost:8088** and choose **Try the demo**. Preparing samples and
using document Q&A make billable OpenAI calls. For an owner login, run
`sh deploy/compose.sh exec web python manage.py createsuperuser`.

Run the live evaluation with
`sh deploy/compose.sh exec web python manage.py evaluate_rag --output /tmp/rag-evaluation.json`.
Temporary evaluation database records are rolled back; the report stays inside the container.

[Deployment details](deploy/server.md) · [CI/CD and rollback behavior](deploy/ci.md)

Rollback restores the previous image and configuration; migrations must remain
compatible with the previous version. It does not restore database data.
