# RAG MVP (Django + pgvector)

A simple Retrieval-Augmented Generation (RAG) MVP built with Django and Postgres (pgvector). It lets you ingest text, PDFs, and `.txt` files into a database, embed the content into vectors, retrieve the most relevant chunks for a question, and generate an answer grounded only in those retrieved sources.
For local development, the database runs via **Docker Compose** (Postgres + pgvector), with a non-Docker option included.


This project is meant as a portfolio-ready demonstration of a real RAG pipeline:
- ingestion (text / PDF /`.txt`)
- chunking
- embeddings storage (Postgres + pgvector)
- similarity search (cosine distance)
- grounded answering with citations/sources
- a minimal web UI to test the system end-to-end

---

## Features

- **Ingest text** (title + body)
- **Ingest PDFs** (multipart upload; extract text then chunk)
- **Ingest `.txt` files** (multipart upload; extract text then chunk)
- **Vector search with pgvector** (Cosine distance in Postgres)
- **RAG answering** (LLM answers using *only* retrieved chunks)
- **“I don’t know” guardrail** when similarity is too low
- **Document selection** stored in session (`current_document_id`)
- **Query logging** for debugging (latency, best distance, sources, errors)
- Optional “convenience” endpoints: **ingest + ask** in one call

---

## Architecture (high-level)

1. **Ingestion**
   - Text/PDF/`.txt` → extract text → chunk → embed chunks → store chunks + vectors in Postgres
2. **Retrieval**
   - Question → embed → cosine distance search in DB → top-k chunks
3. **Generation**
   - Send retrieved chunks as “Sources” → model answers using *only* sources
4. **Response**
   - Return `answer` + `sources` (doc id, chunk index, distance, chunk text)

---

## Tech stack

- Python + Django
- Postgres + **pgvector**
- **Docker + Docker Compose** (local Postgres + pgvector dev environment)
- OpenAI embeddings + chat model
- Minimal HTML/JS UI (fetch-based)

---

## Setup

### 1) Clone repo & create venv

```bash
git clone git@github.com:LuSilverX/Rag-Chatbot.git
cd rag-chatbot

python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```
### 3) Configure environment variables
Create a .env file (or export env vars in your shell):

```bash
export OPENAI_API_KEY="your_key_here"
```
If you use .env, make sure your Django settings load it (e.g., with python-dotenv).

### 4) Start Postgres + pgvector (Docker)
This project uses Docker Compose to run Postgres with the pgvector extension locally.

```bash
docker compose up -d
```

 **Option B: Local Postgres (no Docker)** 
Example (once Postgres is installed and running):

```bash
createdb ragdb
createuser raguser --pwprompt
psql -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```
Update your Django DATABASES config (example):

```python
DATABASES = {
  "default": {
    "ENGINE": "django.db.backends.postgresql",
    "NAME": "ragdb",
    "USER": "raguser",
    "PASSWORD": "your_password",
    "HOST": "127.0.0.1",
    "PORT": "5432",
  }
}
```

### 5) Run migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 6) Run the server

```bash
python manage.py runserver
```

Open:
http://127.0.0.1:8000/ (UI)

---

## Quick demo (sample docs + test questions)

This repo includes `sample_docs/` so you can try the app immediately after setup.

### 1) Ingest a sample document
In the UI:
- Use **Ingest Text File (.txt/.md)** to upload:
  - `sample_docs/demo.txt`
  - `sample_docs/infra_notes.txt`
  - `sample_docs/policies.txt`
- Or upload the PDF:
  - `sample_docs/RAG_MVP_Demo_PDF.pdf`

### 2) Ask sample questions

**After ingesting `demo.txt`:**
- What is RAG?
- List the steps of the RAG pipeline.
- What database + extension are used for vector search?
- What does cosine distance mean here?
- What should the system do if sources don’t contain the answer?
- **Guardrail test:** What year was this project founded?

**After ingesting `infra_notes.txt`:**
- Why does this project use Docker Compose?
- What is pgvector used for?
- What does “smaller distance” mean?
- **Guardrail test:** What is the CEO’s name?

**After ingesting `policies.txt`:**
- How long should passwords be?
- Is MFA required? For who?
- Who can access production databases?
- How far in advance should PTO be requested?
- **Guardrail test:** What is the company’s stock ticker?

**After ingesting `RAG_MVP_Demo_PDF.pdf`:**
- Summarize this PDF briefly.
- What are the three stages of the RAG pipeline described here?
- What does cosine distance mean, and what does a lower distance indicate?
- Why does chunking improve retrieval quality?
- When should the system respond “I don’t know”?
- **Guardrail test:** What is the author’s phone number?





