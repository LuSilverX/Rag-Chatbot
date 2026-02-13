from django.urls import path
from .views import retrieve, ask, ingest_text, logs, ingest_pdf, documents, select_document

urlpatterns = [
    path("retrieve/", retrieve),
    path("ask/", ask),
    path("ingest_text/", ingest_text),
    path("logs/", logs),
    path("ingest_pdf/", ingest_pdf),
    path("documents/", documents),
    path("select_document/", select_document),
]