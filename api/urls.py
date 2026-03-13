from django.urls import path
from .views import (
    clear_selected_document, 
    retrieve, 
    ask, 
    ingest_text, 
    logs, 
    ingest_pdf, 
    documents,
    select_document, 
    app, 
    ingest_file, 
    reset_data
)

urlpatterns = [
    path("", app, name="app"),
    path("retrieve/", retrieve),
    path("ask/", ask),
    path("ingest_text/", ingest_text),
    path("logs/", logs),
    path("ingest_pdf/", ingest_pdf),
    path("documents/", documents),
    path("select_document/", select_document),
    path("ingest_file/", ingest_file),
    path("clear_document/", clear_selected_document),
    path("reset_data/", reset_data),
]