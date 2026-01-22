from django.urls import path
from .views import retrieve

urlpatterns = [
    path("retrieve/", retrieve),
]