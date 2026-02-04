from django.urls import path
from .views import retrieve, ask, ingest_text, logs, ingest_pdf

urlpatterns = [
    path("retrieve/", retrieve),
    path("ask/", ask),
    path("ingest_text/", ingest_text),
    path("logs/", logs),
    path("ingest_pdf/", ingest_pdf),
]