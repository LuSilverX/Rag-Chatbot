from django.db import models
from pgvector.django import VectorField, HnswIndex


class Document(models.Model):
    title = models.CharField(max_length=255)
    source = models.CharField(max_length=1024, blank=True)  # filename, url, etc.
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.IntegerField()  # ordering within the doc
    text = models.TextField()
   
    # using 1536 as a default
    embedding = VectorField(dimensions=1536, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [('document', 'chunk_index')]
        indexes = [
            HnswIndex(
                name='chunk_embedding_hnsw',
                fields=['embedding'],
                opclasses=['vector_cosine_ops'],
            )
        ]

class QueryLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    question = models.TextField()
    answer = models.TextField(blank=True, default="")
    
    k = models.IntegerField(default=5)  
    document_id = models.IntegerField(null=True, blank=True)  

    max_distance = models.FloatField(null=True, blank=True) 
    best_distance = models.FloatField(null=True, blank=True)  

    sources = models.JSONField(default=list)  # list of source + distances
    latency_ms = models.IntegerField(null=True, blank=True)

    error = models.TextField(blank=True, default="")

    def __str__(self):
        return f"{self.created_at: %Y-%m-%d %H:%M:%S} - {self.question[:40]}"