# RAG MVP (Django + pgvector)

A simple Retrieval-Augmented Generation (RAG) MVP built with Django and Postgres (pgvector). It lets you ingest text and PDFs into a database, embed the content into vectors, retrieve the most relevant chunks for a question, and generate an answer grounded only in those retrieved sources.

This project is meant as a portfolio-ready demonstration of a real RAG pipeline:
- ingestion (text / PDF)
- chunking
- embeddings storage (pgvector)
- similarity search
- grounded answering with citations/sources
- a minimal web UI to test the system end-to-end

---

## Features

- **Ingest text** (title + body)
- **Ingest PDFs** (multipart upload; extract text then chunk)
- **Vector search with pgvector** (Cosine distance)
- **RAG answering** (LLM answers using *only* retrieved chunks)
- **“I don’t know” guardrail** when similarity is too low
- **Document selection** stored in session (`current_document_id`)
- **Query logging** for debugging (latency, best distance, sources, errors)
- Optional “convenience” endpoints: **ingest + ask** in one call

---

## Architecture (high-level)

1. **Ingestion**
   - Text/PDF → extract text → chunk → embed chunks → store chunks + vectors in Postgres
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
- OpenAI embeddings + chat model
- Minimal HTML/JS UI (fetch-based)

---

## Setup

### 1) Clone repo & create venv

```bash
git clone <YOUR_REPO_URL>
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

### 4) Set up Postgres + pgvector
Install Postgres and enable the pgvector extension.
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





