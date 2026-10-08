from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from django.core.management import call_command
from django.db import DatabaseError
from django.http import JsonResponse
from django.test import RequestFactory, SimpleTestCase, TestCase, override_settings
from config.middleware import TrustedClientIPMiddleware


class ReadinessTests(TestCase):
    def test_ready_without_login_or_ai_calls(self):
        with patch('api.views.client.embeddings.create') as embed:
            response = self.client.get('/health/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ready'})
        self.assertEqual(response['Cache-Control'], 'no-store')
        embed.assert_not_called()

    def test_database_failure_returns_unavailable_without_details(self):
        with patch('config.health.connection.cursor', side_effect=DatabaseError('private database details')):
            response = self.client.get('/health/')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {'status': 'unavailable'})
        self.assertNotIn('private', response.content.decode())

    @override_settings(DEBUG=False, SECURE_SSL_REDIRECT=True)
    def test_internal_health_does_not_redirect_to_https(self):
        self.assertEqual(self.client.get('/health/').status_code, 200)
        self.assertEqual(self.client.get('/accounts/login/').status_code, 301)

    def test_post_is_not_a_health_check(self):
        self.assertEqual(self.client.post('/health/').status_code, 405)

    def test_cleanup_heartbeat_only_written_after_success(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'heartbeat'
            with patch('api.management.commands.cleanup_workspaces.cleanup_expired_workspaces', side_effect=RuntimeError('unavailable')):
                with self.assertRaises(RuntimeError):
                    call_command('cleanup_workspaces', heartbeat_file=str(path), verbosity=0)
            self.assertFalse(path.exists())
            call_command('cleanup_workspaces', heartbeat_file=str(path), verbosity=0)
            self.assertGreater(float(path.read_text()), 0)


class ProxyAddressTests(SimpleTestCase):
    def request(self, **headers):
        request = RequestFactory().get('/', REMOTE_ADDR='192.0.2.1', **headers)
        return TrustedClientIPMiddleware(lambda req: JsonResponse({'ip': req.META['REMOTE_ADDR']}))(request)

    @override_settings(DJANGO_TRUST_CLIENT_IP=False)
    def test_direct_development_requests_cannot_spoof_ip(self):
        response = self.request(HTTP_X_REAL_IP='198.51.100.10', HTTP_X_FORWARDED_FOR='203.0.113.2')
        self.assertEqual(response.content, b'{"ip": "192.0.2.1"}')

    @override_settings(DJANGO_TRUST_CLIENT_IP=True)
    def test_private_proxy_uses_validated_single_client_address(self):
        response = self.request(HTTP_X_REAL_IP='2001:db8::1')
        self.assertEqual(response.content, b'{"ip": "2001:db8::1"}')
        self.assertEqual(self.request(HTTP_X_REAL_IP='1.2.3.4, 5.6.7.8').status_code, 400)
        self.assertEqual(self.request().content, b'{"ip": "192.0.2.1"}')
