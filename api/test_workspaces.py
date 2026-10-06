import json
from datetime import timedelta
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import Client, TestCase, override_settings
from django.utils import timezone
from .models import Chunk, DemoAccount, Document, QueryLog, VisitorWorkspace


class VisitorWorkspaceTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        user = get_user_model().objects.create_user(username='workspace-visitor')
        DemoAccount.objects.create(user=user)
        cls.template = Document.objects.create(title='Sample', source='portfolio_demo', is_demo=True)
        Chunk.objects.create(document=cls.template, chunk_index=0, text='Original sample.', embedding=[0.1] * 1536)
        cls.owner_doc = Document.objects.create(title='Owner private notes')

    def setUp(self):
        self.first = self.enter()
        self.second = self.enter()
        self.embed = patch('api.views.client.embeddings.create').start()
        self.respond = patch('api.views.client.responses.create').start()
        self.addCleanup(patch.stopall)
        self.embed.side_effect = lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in (kwargs['input'] if isinstance(kwargs['input'], list) else [kwargs['input']])])
        self.respond.return_value.output_text = 'An answer from your document.'

    def enter(self):
        client = Client()
        self.assertEqual(client.post('/accounts/demo/').status_code, 302)
        return client

    def post(self, client, endpoint, data):
        return client.post(f'/api/{endpoint}/', json.dumps(data), content_type='application/json')

    def upload(self, client, **extra):
        return self.post(client, 'ingest_text', {'title': 'My document', 'text': 'My original content.', **extra})

    def test_equal_titles_do_not_replace_other_visitors_content(self):
        a = self.upload(self.first).json()['document_id']
        b = self.upload(self.second, text='Second visitor content.').json()['document_id']
        self.assertNotEqual(a, b)
        self.assertEqual(Document.objects.get(pk=a).chunks.get().text, 'My original content.')
        self.assertEqual(Document.objects.get(pk=b).chunks.get().text, 'Second visitor content.')
        self.assertNotIn(a, [doc['id'] for doc in self.second.get('/api/documents/').json()['documents']])

    def test_replacement_is_scoped_and_failure_preserves_original(self):
        doc_id = self.upload(self.first).json()['document_id']
        response = self.upload(self.first, document_id=doc_id, text='Replacement.')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['document_id'], doc_id)
        self.assertEqual(Document.objects.get(pk=doc_id).chunks.get().text, 'Replacement.')
        self.embed.side_effect = RuntimeError('simulated failure')
        response = self.upload(self.first, document_id=doc_id, text='Must not replace.')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(Document.objects.get(pk=doc_id).chunks.get().text, 'Replacement.')

    def test_cannot_select_ask_replace_or_delete_someone_elses_document(self):
        other_id = self.upload(self.second).json()['document_id']
        self.embed.reset_mock()
        for doc_id in [other_id, self.owner_doc.pk, self.template.pk]:
            for endpoint, data in [('select_document', {}), ('ask', {'question': 'Read it'}), ('ingest_text', {'title': 'Overwrite', 'text': 'Bad change'}), ('delete_document', {})]:
                with self.subTest(endpoint=endpoint, doc_id=doc_id):
                    self.assertEqual(self.post(self.first, endpoint, {**data, 'document_id': doc_id}).status_code, 404)
        self.embed.assert_not_called()
        self.assertTrue(Document.objects.filter(pk=other_id).exists())

    def test_workspace_identifier_from_another_session_is_not_accepted(self):
        session = self.first.session
        session['workspace_id'] = self.second.session['workspace_id']
        session.save()
        self.assertEqual(self.first.get('/api/documents/').status_code, 401)
        self.assertEqual(self.upload(self.first).status_code, 401)
        self.embed.assert_not_called()

    def test_pdf_and_text_uploads_are_available(self):
        response = self.first.post('/api/ingest_file/', {'file': SimpleUploadedFile('notes.txt', b'Visitor text')})
        self.assertEqual(response.status_code, 200)
        with patch('api.views.PdfReader', return_value=SimpleNamespace(is_encrypted=False, pages=[SimpleNamespace(extract_text=lambda: 'Visitor PDF text')])):
            response = self.first.post('/api/ingest_pdf/', {'file': SimpleUploadedFile('notes.pdf', b'fixture')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Document.objects.get(pk=response.json()['document_id']).workspace_id, VisitorWorkspace.objects.get(pk=self.first.session['workspace_id']).pk)

    def test_pdf_and_file_replacement_ids_are_authorized(self):
        other_id = self.second.session['current_document_id']
        response = self.first.post('/api/ingest_file/', {'document_id': other_id, 'file': SimpleUploadedFile('notes.txt', b'Overwrite')})
        self.assertEqual(response.status_code, 404)
        with patch('api.views.PdfReader', return_value=SimpleNamespace(is_encrypted=False, pages=[SimpleNamespace(extract_text=lambda: 'Overwrite')])):
            response = self.first.post('/api/ingest_pdf/', {'document_id': other_id, 'file': SimpleUploadedFile('notes.pdf', b'fixture')})
        self.assertEqual(response.status_code, 404)
        self.embed.assert_not_called()

    def test_search_and_history_only_include_current_workspace(self):
        own_id = self.upload(self.first).json()['document_id']
        other_id = self.upload(self.second, text='Other private content.').json()['document_id']
        response = self.post(self.first, 'ask', {'document_id': own_id, 'question': 'My question?'})
        self.assertEqual(response.status_code, 200)
        self.post(self.second, 'ask', {'document_id': other_id, 'question': 'Other private question?'})
        logs = self.first.get('/api/logs/').json()['logs']
        self.assertEqual([log['question'] for log in logs], ['My question?'])
        self.assertEqual(logs[0]['answer'], 'An answer from your document.')
        self.assertTrue(logs[0]['sources'])
        response = self.post(self.first, 'retrieve', {'query': 'Content', 'k': 20})
        self.assertEqual(response.status_code, 200)
        allowed = set(Document.objects.filter(workspace_id=self.first.session['workspace_id']).values_list('id', flat=True))
        self.assertTrue(all(row['document_id'] in allowed for row in response.json()['results']))

    def test_deleting_sample_copy_preserves_template_and_other_workspace(self):
        own_id = self.first.session['current_document_id']
        self.post(self.first, 'ask', {'document_id': own_id, 'question': 'A question?'})
        response = self.post(self.first, 'delete_document', {'document_id': own_id})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Document.objects.filter(pk=own_id).exists())
        self.assertFalse(Chunk.objects.filter(document_id=own_id).exists())
        self.assertEqual(self.first.get('/api/logs/').json()['logs'], [])
        self.assertNotIn('current_document_id', self.first.session)
        self.assertTrue(self.template.chunks.exists())
        self.assertTrue(Document.objects.filter(pk=self.second.session['current_document_id']).exists())

    def test_expired_workspace_cannot_read_or_call_ai_before_cleanup(self):
        VisitorWorkspace.objects.filter(pk=self.first.session['workspace_id']).update(expires_at=timezone.now() - timedelta(seconds=1))
        for endpoint in ['documents', 'logs']:
            self.assertEqual(self.first.get(f'/api/{endpoint}/').status_code, 401)
        self.assertEqual(self.upload(self.first).status_code, 401)
        self.assertEqual(self.post(self.first, 'ask', {'question': 'Hello?'}).status_code, 401)
        self.embed.assert_not_called()

    def test_cleanup_removes_only_expired_content_and_history(self):
        workspace_id = self.first.session['workspace_id']
        own_id = self.upload(self.first).json()['document_id']
        self.post(self.first, 'ask', {'document_id': own_id, 'question': 'Temporary?'})
        VisitorWorkspace.objects.filter(pk=workspace_id).update(expires_at=timezone.now() - timedelta(seconds=1))
        call_command('cleanup_workspaces', stdout=StringIO())
        self.assertFalse(VisitorWorkspace.objects.filter(pk=workspace_id).exists())
        self.assertFalse(Document.objects.filter(pk=own_id).exists())
        self.assertFalse(Chunk.objects.filter(document_id=own_id).exists())
        self.assertFalse(QueryLog.objects.filter(workspace_id=workspace_id).exists())
        self.assertTrue(VisitorWorkspace.objects.filter(pk=self.second.session['workspace_id']).exists())
        self.assertTrue(Document.objects.filter(pk=self.owner_doc.pk).exists())
        self.assertTrue(Document.objects.filter(pk=self.template.pk).exists())

    def test_expiration_during_embedding_does_not_resurrect_upload(self):
        workspace_id = self.first.session['workspace_id']
        def expire(**kwargs):
            VisitorWorkspace.objects.filter(pk=workspace_id).update(expires_at=timezone.now() - timedelta(seconds=1))
            call_command('cleanup_workspaces', stdout=StringIO())
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1536)])
        self.embed.side_effect = expire
        self.assertEqual(self.upload(self.first).status_code, 401)
        self.assertFalse(Document.objects.filter(workspace_id=workspace_id).exists())

    @override_settings(DEMO_MAX_DOCUMENTS=1)
    def test_document_limit_allows_replacing_an_existing_copy(self):
        self.assertEqual(self.upload(self.first).status_code, 400)
        self.embed.assert_not_called()
        self.assertEqual(self.upload(self.first, document_id=self.first.session['current_document_id']).status_code, 200)

    @override_settings(DEMO_UPLOAD_SESSION_HOURLY_LIMIT=1)
    def test_upload_quota_is_reserved_before_ai_calls(self):
        self.assertEqual(self.upload(self.first).status_code, 200)
        response = self.upload(self.first, title='Second upload')
        self.assertEqual(response.status_code, 429)
        self.assertIn('Retry-After', response)
        self.assertEqual(self.embed.call_count, 1)

    def test_extracted_text_and_pdf_page_limits_precede_ai_calls(self):
        self.assertEqual(self.upload(self.first, text='x' * 20001).status_code, 400)
        with patch('api.views.PdfReader', return_value=SimpleNamespace(is_encrypted=False, pages=[None] * 21)):
            response = self.first.post('/api/ingest_pdf/', {'file': SimpleUploadedFile('long.pdf', b'fixture')})
        self.assertEqual(response.status_code, 400)
        self.embed.assert_not_called()

    def test_expired_session_can_start_a_fresh_workspace(self):
        old_id = self.first.session['workspace_id']
        VisitorWorkspace.objects.filter(pk=old_id).update(expires_at=timezone.now() - timedelta(seconds=1))
        self.assertEqual(self.first.post('/accounts/demo/').status_code, 302)
        self.assertNotEqual(self.first.session['workspace_id'], old_id)
        self.assertEqual(self.first.get('/api/documents/').status_code, 200)

    def test_owner_interface_does_not_list_temporary_visitor_content(self):
        owner = get_user_model().objects.create_user(username='owner')
        client = Client()
        client.force_login(owner)
        visitor_id = self.upload(self.first).json()['document_id']
        ids = [doc['id'] for doc in client.get('/api/documents/').json()['documents']]
        self.assertIn(self.owner_doc.pk, ids)
        self.assertNotIn(visitor_id, ids)

    def test_in_progress_history_is_not_marked_answered(self):
        QueryLog.objects.create(workspace_id=self.first.session['workspace_id'], question='In progress')
        self.assertEqual(self.first.get('/api/logs/').json()['logs'][0]['status'], 'pending')

    def test_workspace_mutations_require_csrf(self):
        client = Client(enforce_csrf_checks=True)
        client.get('/accounts/login/')
        client.post('/accounts/demo/', HTTP_X_CSRFTOKEN=client.cookies['csrftoken'].value)
        token = client.cookies['csrftoken'].value
        for endpoint in ['ingest_text', 'delete_document']:
            self.assertEqual(self.post(client, endpoint, {'document_id': client.session['current_document_id'], 'text': 'New'}).status_code, 403)
        response = client.post('/api/ingest_text/', json.dumps({'title': 'CSRF upload', 'text': 'Protected request.'}), content_type='application/json', HTTP_X_CSRFTOKEN=token)
        self.assertEqual(response.status_code, 200)


from concurrent.futures import ThreadPoolExecutor
from django.db import close_old_connections
from django.test import TransactionTestCase
import threading


class WorkspaceConcurrencyTests(TransactionTestCase):
    @override_settings(DEMO_MAX_DOCUMENTS=2)
    def test_parallel_uploads_cannot_exceed_document_cap(self):
        user = get_user_model().objects.create_user(username='concurrent-visitor')
        DemoAccount.objects.create(user=user)
        template = Document.objects.create(title='Seed', source='portfolio_demo', is_demo=True)
        Chunk.objects.create(document=template, chunk_index=0, text='Seed facts.', embedding=[0.1] * 1536)
        browser = Client()
        browser.post('/accounts/demo/')
        session_cookie = browser.cookies['sessionid'].value
        workspace_id = browser.session['workspace_id']
        barrier = threading.Barrier(2)
        def embed(**kwargs):
            barrier.wait(timeout=10)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1536)])
        def upload(index):
            close_old_connections()
            try:
                visitor = Client()
                visitor.cookies['sessionid'] = session_cookie
                return visitor.post('/api/ingest_text/', json.dumps({'title': f'Upload {index}', 'text': 'New facts.'}), content_type='application/json').status_code
            finally:
                close_old_connections()
        with patch('api.views.client.embeddings.create', side_effect=embed), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(upload, [1, 2]))
        self.assertEqual(sorted(results), [200, 400])
        self.assertEqual(Document.objects.filter(workspace_id=workspace_id).count(), 2)
