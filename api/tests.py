from django.test import TestCase, override_settings
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
        
        self.assertEqual(chunks[0], "This is the first part.")
        self.assertTrue(chunks[1].startswith("part."))
        self.assertTrue(all(len(chunk) <= 30 for chunk in chunks))
        self.assertTrue(chunks[-1].endswith("part."))

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

class ReliabilityTests(TestCase):
    def post_json(self, endpoint, payload):
        return self.client.post('/api/' + endpoint + '/', data=json.dumps(payload), content_type='application/json')

    @override_settings(DEBUG=True)
    def test_invalid_inputs_return_400_before_ai_calls(self):
        with patch('api.views.client.embeddings.create') as embed:
            for endpoint in ['ask', 'retrieve', 'ingest_text', 'select_document', 'reset_data']:
                for body in ['{', '[]', 'null', '42']:
                    with self.subTest(endpoint=endpoint, body=body):
                        response = self.client.post('/api/' + endpoint + '/', data=body, content_type='application/json')
                        self.assertEqual(response.status_code, 400)
            for k in [0, -1, 21, 'oops', 1.5, True, None]:
                for endpoint, field in [('ask', 'question'), ('retrieve', 'query')]:
                    self.assertEqual(self.post_json(endpoint, {field: 'hello', 'k': k}).status_code, 400)
            for value in ['nan', 'inf', -1, 3, [], True]:
                self.assertEqual(self.post_json('ask', {'question': 'hello', 'max_distance': value}).status_code, 400)
            for endpoint in ['documents', 'logs']:
                for limit in ['oops', '-1', '0', '101']:
                    self.assertEqual(self.client.get(f'/api/{endpoint}/?limit={limit}').status_code, 400)
            self.assertEqual(self.post_json('ingest_text', {'text': [], 'title': 'bad'}).status_code, 400)
            embed.assert_not_called()

    def test_corrupt_pdf_and_invalid_text_file_are_validation_errors(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        for endpoint, filename, data in [('ingest_pdf', 'bad.pdf', b'broken'), ('ingest_file', 'bad.txt', b'\xff'), ('ingest_file', 'bad.exe', b'text')]:
            response = self.client.post(f'/api/{endpoint}/', {'file': SimpleUploadedFile(filename, data)})
            self.assertEqual(response.status_code, 400)
            self.assertIn('message', response.json())

    def test_upload_size_is_checked_server_side(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        with patch('api.views.client.embeddings.create') as embed:
            response = self.client.post('/api/ingest_file/', {'file': SimpleUploadedFile('large.txt', b'x' * (2 * 1024 * 1024 + 1))})
            self.assertEqual(response.status_code, 400)
            embed.assert_not_called()

    def original_document(self):
        doc = Document.objects.create(title='Keep me', source='ingested_text')
        Chunk.objects.create(document=doc, chunk_index=0, text='Original content', embedding=[0.1] * 1536)
        session = self.client.session
        session['current_document_id'] = doc.id
        session.save()
        return doc

    @patch('api.views.client.embeddings.create')
    def test_embedding_failure_preserves_document_and_session(self, embed):
        from openai import APIConnectionError
        import httpx
        doc = self.original_document()
        embed.side_effect = APIConnectionError(request=httpx.Request('POST', 'https://api.openai.com'))
        response = self.post_json('ingest_text', {'title': doc.title, 'text': 'Replacement'})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(doc.chunks.get().text, 'Original content')
        self.assertEqual(self.client.session['current_document_id'], doc.id)
        self.assertNotIn('details', response.json())

    @patch('api.views.client.embeddings.create')
    def test_database_failure_rolls_back_deleted_chunks(self, embed):
        from django.db import IntegrityError
        from types import SimpleNamespace
        doc = self.original_document()
        embed.return_value.data = [SimpleNamespace(embedding=[0.2] * 1536)]
        with patch('api.views.Chunk.objects.bulk_create', side_effect=IntegrityError('simulated write failure')):
            response = self.post_json('ingest_text', {'title': doc.title, 'text': 'Replacement'})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(doc.chunks.get().text, 'Original content')

    @patch('api.views.client.embeddings.create')
    def test_incomplete_embeddings_preserve_old_content(self, embed):
        doc = self.original_document()
        embed.return_value.data = []
        self.assertEqual(self.post_json('ingest_text', {'title': doc.title, 'text': 'Replacement'}).status_code, 500)
        self.assertEqual(doc.chunks.get().text, 'Original content')

    @patch('api.views.client.embeddings.create')
    def test_successful_update_replaces_chunks(self, embed):
        from types import SimpleNamespace
        doc = self.original_document()
        embed.return_value.data = [SimpleNamespace(embedding=[0.2] * 1536)]
        response = self.post_json('ingest_text', {'title': doc.title, 'text': 'Replacement'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'updated')
        self.assertEqual(Document.objects.count(), 1)
        self.assertEqual(doc.chunks.get().text, 'Replacement')

    @patch('api.views.client.embeddings.create')
    def test_file_upload_uses_text_file_source(self, embed):
        from types import SimpleNamespace
        from django.core.files.uploadedfile import SimpleUploadedFile
        embed.return_value.data = [SimpleNamespace(embedding=[0.2] * 1536)]
        response = self.client.post('/api/ingest_file/', {'file': SimpleUploadedFile('notes.md', b'# Synthetic notes')})
        self.assertEqual(response.status_code, 200)
        doc = Document.objects.get()
        self.assertEqual(doc.source, 'text_file')
        self.assertEqual(self.client.get('/api/documents/').json()['current_document_title'], 'notes.md')

    def test_history_exposes_unanswered_error_and_zero_distance(self):
        from .models import QueryLog
        QueryLog.objects.create(question='Unknown?', answer="I don't know.", best_distance=0, latency_ms=0)
        QueryLog.objects.create(question='Failure?', error='outage')
        logs = self.client.get('/api/logs/').json()['logs']
        self.assertEqual(logs[0]['status'], 'error')
        self.assertEqual(logs[1]['status'], 'unanswered')
        self.assertEqual(logs[1]['best_distance'], 0)
        self.assertEqual(logs[1]['latency_ms'], 0)

    @patch('api.views.client.responses.create')
    @patch('api.views.client.embeddings.create')
    def test_guardrail_and_document_scope(self, embed, respond):
        from types import SimpleNamespace
        doc = Document.objects.create(title='Chosen document')
        other = Document.objects.create(title='Other document')
        Chunk.objects.create(document=doc, chunk_index=0, text='Chosen facts', embedding=[-0.1] * 1536)
        Chunk.objects.create(document=other, chunk_index=0, text='Other facts', embedding=[0.1] * 1536)
        embed.return_value.data = [SimpleNamespace(embedding=[0.1] * 1536)]
        response = self.post_json('ask', {'question': 'What facts?', 'document_id': doc.id})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['answer'], "I don't know.")
        self.assertEqual(response.json()['sources'], [])
        respond.assert_not_called()
        self.assertEqual(self.client.get('/api/logs/').json()['logs'][0]['status'], 'unanswered')

    def test_long_chunks_are_bounded_and_preserve_text(self):
        for text in ['word ' * 3000, 'x' * 3000, ('a' * 850) + '. ' + ('b' * 850) + '.']:
            chunks = chunk_text(text)
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(0 < len(chunk) <= 900 for chunk in chunks))
            self.assertTrue(chunks[0].startswith(text[:20]))
            self.assertTrue(chunks[-1].endswith(text.strip()[-20:]))
        text = ''.join(str(i % 10) for i in range(3000))
        chunks = chunk_text(text)
        self.assertEqual(chunks[0] + ''.join(chunk[200:] for chunk in chunks[1:]), text)

    def test_chunk_parameter_validation(self):
        for maximum, overlap in [(0, 0), (20, 20), (20, -1)]:
            with self.assertRaises(ValueError):
                chunk_text('text', maximum, overlap)

    @patch('api.views.client.embeddings.create')
    def test_ai_failure_is_logged_for_questions(self, embed):
        from openai import APIConnectionError
        from .models import QueryLog
        import httpx
        doc = self.original_document()
        embed.side_effect = APIConnectionError(request=httpx.Request('POST', 'https://api.openai.com'))
        response = self.post_json('ask', {'question': 'What does it say?', 'document_id': doc.id})
        self.assertEqual(response.status_code, 502)
        log = QueryLog.objects.get()
        self.assertTrue(log.error)
        self.assertIsNotNone(log.latency_ms)

    @patch('api.views.client.embeddings.create')
    def test_uploaded_replacements_preserve_chunks_on_ai_failure(self, embed):
        from types import SimpleNamespace
        from django.core.files.uploadedfile import SimpleUploadedFile
        for endpoint, source, filename in [('ingest_file', 'text_file', 'notes.txt'), ('ingest_pdf', 'pdf', 'notes.pdf')]:
            doc = Document.objects.create(title=filename, source=source)
            Chunk.objects.create(document=doc, chunk_index=0, text='Original content', embedding=[0.1] * 1536)
            embed.side_effect = RuntimeError('simulated outage')
            with patch('api.views.PdfReader', return_value=SimpleNamespace(is_encrypted=False, pages=[SimpleNamespace(extract_text=lambda: 'Replacement')])):
                response = self.client.post(f'/api/{endpoint}/', {'file': SimpleUploadedFile(filename, b'Replacement')})
            self.assertEqual(response.status_code, 500)
            self.assertEqual(doc.chunks.get().text, 'Original content')

    def test_oversized_json_returns_validation_error(self):
        response = self.post_json('ingest_text', {'text': 'x' * (3 * 1024 * 1024)})
        self.assertEqual(response.status_code, 400)
