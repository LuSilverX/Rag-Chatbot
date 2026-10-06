import json
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.db import close_old_connections
from django.test import Client, TestCase, TransactionTestCase, override_settings

from .demo import consume_quota, DemoLimitExceeded
from .models import Chunk, DemoAccount, DemoQuota, Document, QueryLog, VisitorWorkspace


class PortfolioDemoTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.demo = get_user_model().objects.create_user(username='demo-test')
        DemoAccount.objects.create(user=cls.demo)
        cls.owner = get_user_model().objects.create_superuser(username='owner-test', password='Test-owner-password-123!')
        cls.sample = Document.objects.create(title='Harbor Museum — Demo', is_demo=True)
        cls.private = Document.objects.create(title='Private owner notes')
        for doc in [cls.sample, cls.private]:
            Chunk.objects.create(document=doc, chunk_index=0, text=f'Content for {doc.title}', embedding=[0.1] * 1536)

    def enter(self, client=None):
        client = client or self.client
        self.assertRedirects(client.post('/accounts/demo/'), '/api/')
        return client

    def ask(self, client=None, **overrides):
        return (client or self.client).post('/api/ask/', json.dumps({'question': 'What does it say?', 'document_id': (client or self.client).session.get('current_document_id'), **overrides}), content_type='application/json')

    def fake_ai(self, embed, respond):
        embed.return_value.data = [SimpleNamespace(embedding=[0.1] * 1536)]
        respond.return_value.output_text = 'A sample answer.'

    def test_demo_entry_requires_post_and_csrf(self):
        client = Client(enforce_csrf_checks=True)
        self.assertContains(client.get('/accounts/login/'), 'Try the demo')
        self.assertEqual(client.get('/accounts/demo/').status_code, 405)
        self.assertEqual(client.post('/accounts/demo/').status_code, 403)
        token = client.cookies['csrftoken'].value
        self.assertRedirects(client.post('/accounts/demo/', HTTP_X_CSRFTOKEN=token), '/api/')
        self.assertEqual(int(client.session['_auth_user_id']), self.demo.pk)
        self.assertLessEqual(client.session.get_expiry_age(), 86400)
        self.assertFalse(self.demo.has_usable_password())
        self.assertRedirects(client.get('/admin/'), '/admin/login/?next=/admin/')

    def test_demo_interface_and_approved_document_listing(self):
        self.enter()
        page = self.client.get('/api/')
        self.assertContains(page, 'Your temporary workspace')
        self.assertContains(page, 'Museum opening hours')
        for element in ['id="btnResetDb"']:
            self.assertNotContains(page, element)
        data = self.client.get('/api/documents/').json()
        self.assertEqual([d['id'] for d in data['documents']], [self.client.session['current_document_id']])
        self.assertNotEqual(data['documents'][0]['id'], self.sample.pk)
        for element in ['id="btnUpload"', 'id="btnIngestText"', 'id="developer"']:
            self.assertContains(page, element)
        self.assertEqual(data['current_document_id'], self.client.session['current_document_id'])

    def test_direct_prohibited_requests_never_call_ai_or_change_documents(self):
        self.enter()
        QueryLog.objects.create(question='Private question', answer='Private answer')
        with patch('api.views.client.embeddings.create') as embed:
            for endpoint in ['reset_data']:
                with self.subTest(endpoint=endpoint):
                    response = self.client.post(f'/api/{endpoint}/', json.dumps({'text': 'Replacement', 'confirm': 'RESET', 'query': 'Private'}), content_type='application/json')
                    self.assertEqual(response.status_code, 403)
            response = self.client.get('/api/logs/')
            self.assertEqual(response.status_code, 200)
            self.assertNotIn('Private question', response.content.decode())
            embed.assert_not_called()
        self.assertEqual(Document.objects.count(), 3)
        self.assertEqual(Chunk.objects.count(), 3)

    def test_private_ids_and_stale_session_selection_are_rejected(self):
        self.enter()
        with patch('api.views.client.embeddings.create') as embed:
            self.assertEqual(self.ask(document_id=self.private.pk).status_code, 404)
            response = self.client.post('/api/select_document/', json.dumps({'document_id': self.private.pk}), content_type='application/json')
            self.assertEqual(response.status_code, 404)
            session = self.client.session
            session['current_document_id'] = self.private.pk
            session.save()
            self.assertEqual(self.ask(document_id=None).status_code, 404)
            embed.assert_not_called()

    @patch('api.views.client.responses.create')
    @patch('api.views.client.embeddings.create')
    def test_answers_include_only_approved_sources_and_bounded_generation(self, embed, respond):
        self.fake_ai(embed, respond)
        self.enter()
        response = self.ask(k=20, max_distance=2)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['sources'][0]['document_id'], self.client.session['current_document_id'])
        self.assertEqual(respond.call_args.kwargs['max_output_tokens'], 300)
        self.assertEqual(QueryLog.objects.get().k, 3)
        self.assertEqual(QueryLog.objects.get().max_distance, 0.95)
        self.assertEqual(self.ask(question='x' * 1001).status_code, 400)
        self.assertEqual(embed.call_count, 1)

    @override_settings(DEMO_SESSION_HOURLY_LIMIT=1)
    @patch('api.views.client.responses.create')
    @patch('api.views.client.embeddings.create')
    def test_session_limit_and_reentering_cannot_reset_it(self, embed, respond):
        self.fake_ai(embed, respond)
        self.enter()
        self.assertEqual(self.ask().status_code, 200)
        self.enter()
        response = self.ask()
        self.assertEqual(response.status_code, 429)
        self.assertGreater(int(response['Retry-After']), 0)
        self.assertEqual(embed.call_count, 1)

    @override_settings(DEMO_IP_HOURLY_LIMIT=1)
    @patch('api.views.client.responses.create')
    @patch('api.views.client.embeddings.create')
    def test_new_cookies_and_forwarded_headers_do_not_reset_ip_budget(self, embed, respond):
        self.fake_ai(embed, respond)
        self.enter()
        self.assertEqual(self.ask().status_code, 200)
        other = self.enter(Client(HTTP_X_FORWARDED_FOR='1.2.3.4'))
        self.assertEqual(self.ask(other).status_code, 429)
        self.assertEqual(embed.call_count, 1)

    @override_settings(DEMO_DAILY_LIMIT=1)
    @patch('api.views.client.responses.create')
    @patch('api.views.client.embeddings.create')
    def test_global_budget_spans_networks_and_sessions(self, embed, respond):
        self.fake_ai(embed, respond)
        self.enter()
        self.assertEqual(self.ask().status_code, 200)
        other = self.enter(Client(REMOTE_ADDR='192.0.2.2'))
        self.assertEqual(self.ask(other).status_code, 429)
        self.assertEqual(embed.call_count, 1)

    @override_settings(DEMO_SESSION_HOURLY_LIMIT=1)
    @patch('api.views.client.embeddings.create')
    def test_api_failures_still_consume_budget(self, embed):
        import httpx
        from openai import APIConnectionError
        self.enter()
        embed.side_effect = APIConnectionError(request=httpx.Request('POST', 'https://api.openai.com'))
        self.assertEqual(self.ask().status_code, 502)
        self.assertEqual(self.ask().status_code, 429)
        self.assertEqual(embed.call_count, 1)

    def test_visitors_have_separate_sessions_and_private_history(self):
        first = self.enter()
        second = self.enter(Client())
        self.assertNotEqual(first.session.session_key, second.session.session_key)
        first.post('/api/clear_document/')
        self.assertIsNotNone(second.session['current_document_id'])
        self.assertNotEqual(second.session['workspace_id'], first.session['workspace_id'])
        for visitor in [first, second]:
            self.assertEqual(visitor.get('/api/logs/').status_code, 200)

    def test_owner_access_is_preserved(self):
        self.client.force_login(self.owner)
        self.enter()
        self.assertEqual(int(self.client.session['_auth_user_id']), self.owner.pk)
        self.assertContains(self.client.get('/api/'), 'id="btnUpload"')
        self.assertEqual(len(self.client.get('/api/documents/').json()['documents']), 2)
        self.assertEqual(self.client.get('/api/logs/').status_code, 200)

    def test_demo_fails_closed_if_identity_gains_privileges(self):
        self.demo.is_staff = True
        self.demo.save()
        self.assertEqual(self.client.post('/accounts/demo/').status_code, 503)
        self.assertNotIn('_auth_user_id', self.client.session)

    def test_empty_workspace_is_usable_without_samples(self):
        Document.objects.filter(is_demo=True).update(is_demo=False)
        self.enter()
        self.assertEqual(self.client.get('/api/documents/').json()['documents'], [])

    def test_seed_command_is_repeatable_and_preserves_private_documents(self):
        with patch('api.views.client.embeddings.create') as embed:
            embed.side_effect = lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in kwargs['input']])
            call_command('prepare_demo', stdout=StringIO())
            calls = embed.call_count
            call_command('prepare_demo', stdout=StringIO())
            self.assertEqual(embed.call_count, calls)
        self.private.refresh_from_db()
        self.assertFalse(self.private.is_demo)
        self.assertEqual(DemoAccount.objects.count(), 1)
        self.assertEqual(Document.objects.filter(source='portfolio_demo', is_demo=True).count(), 2)


class DemoQuotaConcurrencyTests(TransactionTestCase):
    @override_settings(DEMO_DAILY_LIMIT=1)
    def test_parallel_requests_only_reserve_one_global_slot(self):
        import threading
        barrier = threading.Barrier(2)
        def reserve(index):
            close_old_connections()
            request = SimpleNamespace(META={'REMOTE_ADDR': f'192.0.2.{index}'}, session=SimpleNamespace(session_key=f'session-{index}'))
            try:
                barrier.wait(timeout=5)
                consume_quota(request)
                return True
            except DemoLimitExceeded:
                return False
            finally:
                close_old_connections()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(reserve, [1, 2]))
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(DemoQuota.objects.get(key='ask:global').count, 1)

    @override_settings(DEMO_SESSION_HOURLY_LIMIT=1)
    def test_hour_window_resets_and_daily_budget_remains(self):
        request = SimpleNamespace(META={'REMOTE_ADDR': '192.0.2.1'}, session=SimpleNamespace(session_key='session'))
        with patch('api.demo.time.time', return_value=864000):
            consume_quota(request)
            with self.assertRaises(DemoLimitExceeded):
                consume_quota(request)
        with patch('api.demo.time.time', return_value=867600):
            consume_quota(request)
        self.assertEqual(DemoQuota.objects.get(key='ask:global').count, 2)
