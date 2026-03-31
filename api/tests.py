from django.test import TestCase
from .views import chunk_text  
import json
from unittest.mock import patch, MagicMock
from .models import Document, Chunk

class ChunkTextTests(TestCase):
    
    def test_empty_string_returns_empty_list(self):
        """If we pass nothing, we should get nothing back."""
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n  "), [])
        self.assertEqual(chunk_text(None), [])

    def test_basic_sentence_splitting(self):
        """It should pack sentences into chunks up to max_chars."""
        text = "Sentence one. Sentence two. Sentence three."
        
        # Setting max_chars artificially low (20) to force it to split the sentences up
        # Setting overlap to 0 to keep the math simple for this test
        chunks = chunk_text(text, max_chars=20, overlap=0)
        
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Sentence one.")
        self.assertEqual(chunks[1], "Sentence two.")
        self.assertEqual(chunks[2], "Sentence three.")

    def test_overlap_logic(self):
        """It should carry over the specified overlap characters to the next chunk."""
        text = "This is the first part. And this is the second part."
        
        chunks = chunk_text(text, max_chars=30, overlap=5)
        
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "This is the first part.")
        self.assertEqual(chunks[1], "part. And this is the second part.")

class ApiIntegrationTests(TestCase):

    @patch('api.views.client.embeddings.create') 
    def test_ingest_text_endpoint(self, mock_embeddings_create):
        """
        Tests the /api/ingest_text/ endpoint.
        Ensures the DB saves the Document, the Chunks, and handles session state correctly.
        """
        
        # Setting up fake openai response 
        fake_vector = [0.01] * 1536 
        
        # Creating a fake "item" that mimics the structure of OpenAI's response
        mock_item = MagicMock()
        mock_item.embedding = fake_vector
        mock_embeddings_create.return_value.data = [mock_item]

        # Firing the request
        payload = {
            "title": "Integration Test Doc",
            "text": "This is a very simple test document."
        }
        
        response = self.client.post(
            '/api/ingest_text/',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Did the server say OK?
        self.assertEqual(response.status_code, 200)
        
        # Did it actually save to PostgreSQL?
        self.assertEqual(Document.objects.count(), 1)
        self.assertEqual(Chunk.objects.count(), 1)
        
        # Did it save the right data?
        saved_doc = Document.objects.first()
        self.assertEqual(saved_doc.title, "Integration Test Doc")
        
        saved_chunk = Chunk.objects.first()
        self.assertEqual(saved_chunk.text, "This is a very simple test document.")
        self.assertIsNotNone(saved_chunk.embedding)

        # Did it set the session correctly?
        self.assertEqual(self.client.session.get("current_document_id"), saved_doc.id)

    @patch('api.views.client.responses.create') 
    @patch('api.views.client.embeddings.create')
    def test_ask_endpoint(self, mock_embeddings_create, mock_responses_create):
        """
        Tests the /api/ask/ endpoint.
        Ensures vector retrieval works, the LLM is called, and the QueryLog is saved.
        """
        
        # Need a document and a chunk in the database to actually "search" against
        doc = Document.objects.create(title="Test Knowledge Base")
        
        # Gives the chunk an embedding of straight 0.1s
        chunk_vector = [0.1] * 1536 
        Chunk.objects.create(
            document=doc,
            chunk_index=0,
            text="The secret password is 'Pineapple'.",
            embedding=chunk_vector
        )

        # Force the test client's session to have this document selected
        session = self.client.session
        session["current_document_id"] = doc.id
        session.save()
        
        # Makes the question's vector EXACTLY match the chunk's vector so the distance is 0.0
        mock_emb_item = MagicMock()
        mock_emb_item.embedding = [0.1] * 1536 
        mock_embeddings_create.return_value.data = [mock_emb_item]

        # Fakes what GPT-4 would say after reading the chunk
        mock_responses_create.return_value.output_text = "The secret password is Pineapple."

        # Firing the request
        payload = {
            "question": "What is the secret password?",
            "k": 1
        }
        
        response = self.client.post(
            '/api/ask/',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Checking the response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Did the LLM return the mocked answer?
        self.assertEqual(data["answer"], "The secret password is Pineapple.")
        
        # Did the vector search find the source chunk?
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["text"], "The secret password is 'Pineapple'.")
        
        # Did the distance calculation work? (Should be very close to 0.0 since the vectors match)
        self.assertTrue(data["sources"][0]["distance"] < 0.01)

        # Did it log the query for the analytics?
        from .models import QueryLog 
        self.assertEqual(QueryLog.objects.count(), 1)
        log = QueryLog.objects.first()
        self.assertEqual(log.question, "What is the secret password?")
        self.assertEqual(log.answer, "The secret password is Pineapple.")