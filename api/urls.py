from django.urls import path
from .views import retrieve, ask, ingest_text, logs

urlpatterns = [
    path("retrieve/", retrieve),
    path("ask/", ask),
    path("ingest_text/", ingest_text),
    path("logs/", logs),
]