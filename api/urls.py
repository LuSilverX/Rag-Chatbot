from django.urls import path
from .views import retrieve, ask, ingest_text

urlpatterns = [
    path("retrieve/", retrieve),
    path("ask/", ask),
    path("ingest_text/", ingest_text),
]