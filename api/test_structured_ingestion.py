import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, SimpleTestCase, TestCase, override_settings
from .ingestion import json_to_text, csv_to_text, StructuredInputError
from .models import DemoAccount, Document, Chunk, DemoQuota


class StructuredParserTests(SimpleTestCase):
    def test_nested_json_preserves_paths_types_and_unicode(self):
        text = json_to_text('{"products":[{"name":"Café","price":12.50,"available":true,"note":null}]}', 20000)
        for expected in ['$["products"][0]["name"]: "Café"', '["price"]: 12.50', '["available"]: true', '["note"]: null']:
            self.assertIn(expected, text)

    def test_json_array_and_large_finite_number(self):
        self.assertIn('$[0]: 1E+999', json_to_text('[1e999,false]', 20000))

    def test_invalid_or_empty_json(self):
        for raw in ['{', 'null', '12', '"hello"', '{}', '[]', '{"x":[]}', '{"x":"  "}', '{"x":1,"x":2}', '{"a":NaN}', '{"a":Infinity}', '[' * 30 + '1' + ']' * 30]:
            with self.subTest(raw=raw), self.assertRaises(StructuredInputError):
                json_to_text(raw, 20000)

    def test_csv_quotes_newlines_and_labels(self):
        text = csv_to_text('name,notes\r\nCafé,"Courtyard, east entrance"\r\nTrail,"Line one\nLine two"\r\n', 20000)
        self.assertIn('Row 1 | "name": "Café"', text)
        self.assertIn('Row 1 | "notes": "Courtyard, east entrance"', text)
        self.assertIn('Row 2 | "notes": "Line one\\nLine two"', text)

    def test_invalid_or_empty_csv(self):
        for raw in ['', 'name,notes\n', 'name,name\na,b', ',notes\na,b', 'name,notes\na', 'name\na,b', 'name,notes\na,"unfinished', 'name\n"bad"suffix', 'name\n\n', ','.join(f'c{i}' for i in range(101))+'\n'+','.join(['v']*101)]:
            with self.subTest(raw=raw), self.assertRaises(StructuredInputError):
                csv_to_text(raw, 20000)

    def test_conversion_expansion_is_bounded(self):
        for parser, raw in [(json_to_text, '{"long_repeated_key": [1,2,3,4,5]}'), (csv_to_text, 'long_column_name\na\nb\nc\nd')]:
            with self.subTest(parser=parser.__name__), self.assertRaises(StructuredInputError):
                parser(raw, 30)


class StructuredUploadTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        user = get_user_model().objects.create_user(username='structured-visitor')
        DemoAccount.objects.create(user=user)

    def setUp(self):
        self.client.post('/accounts/demo/')
        self.embed = patch('api.views.client.embeddings.create').start()
        self.addCleanup(patch.stopall)
        self.embed.side_effect = lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[0.1]*1536) for _ in (kwargs['input'] if isinstance(kwargs['input'], list) else [kwargs['input']])])

    def upload(self, filename, content, client=None, **extra):
        if isinstance(content, str):
            content = content.encode('utf-8')
        return (client or self.client).post('/api/ingest_file/', {'file': SimpleUploadedFile(filename, content), **extra})

    def test_bundled_json_and_csv_ingest_and_answer_with_sources(self):
        fixtures = [('catalog.json', 'json', 'Trail Lantern'), ('workshops.csv', 'csv', 'Map Reading')]
        for filename, source, expected in fixtures:
            with self.subTest(filename=filename):
                content = (settings.BASE_DIR / 'sample_docs' / filename).read_bytes()
                response = self.upload(filename, content)
                self.assertEqual(response.status_code, 200)
                doc = Document.objects.get(pk=response.json()['document_id'])
                self.assertEqual(doc.source, source)
                self.assertIsNotNone(doc.workspace_id)
                self.assertIn(expected, ' '.join(doc.chunks.values_list('text', flat=True)))
                with patch('api.views.client.responses.create', return_value=SimpleNamespace(output_text=expected)):
                    answer = self.client.post('/api/ask/', json.dumps({'question': 'What is listed?', 'document_id': doc.pk}), content_type='application/json')
                self.assertEqual(answer.status_code, 200)
                self.assertTrue(any(expected in item['text'] for item in answer.json()['sources']))

    def test_utf8_bom_and_uppercase_extensions(self):
        for filename, content in [('data.JSON', b'\xef\xbb\xbf{"name":"Caf\xc3\xa9"}'), ('data.CSV', b'\xef\xbb\xbfname\r\nCaf\xc3\xa9')]:
            self.assertEqual(self.upload(filename, content).status_code, 200)

    def test_bad_input_never_calls_ai_or_spends_upload_quota(self):
        for filename, data in [('bad.json', '{'), ('bad.csv', 'name,name\na,b'), ('empty.json', '{}'), ('empty.csv', 'name\n'), ('bad.json', b'\xff'), ('bad.csv', b'\xff'), ('bad.txt', b'\x00'), ('bad.json', '{"x":"\\ud800"}'), ('bad.csv', 'name\n'+('x'*140000))]:
            with self.subTest(filename=filename, data=str(data)[:40]):
                response = self.upload(filename, data)
                self.assertEqual(response.status_code, 400)
                self.assertIn('message', response.json())
        self.embed.assert_not_called()
        self.assertFalse(DemoQuota.objects.filter(key__startswith='upload:').exists())
        self.assertEqual(Document.objects.count(), 0)

    def test_failed_replacement_preserves_title_source_and_chunks(self):
        for suffix, first, changed in [('json','{"name":"Original"}','{"name":"Changed"}'), ('csv','name\nOriginal','name\nChanged')]:
            response = self.upload('original.'+suffix, first)
            doc = Document.objects.get(pk=response.json()['document_id'])
            original = list(doc.chunks.values_list('text', flat=True))
            with patch('api.views.client.embeddings.create', side_effect=RuntimeError('simulated outage')):
                failed = self.upload('changed.'+suffix, changed, document_id=doc.pk)
            self.assertEqual(failed.status_code, 500)
            doc.refresh_from_db()
            self.assertEqual(doc.title, 'original.'+suffix)
            self.assertEqual(list(doc.chunks.values_list('text', flat=True)), original)
            self.assertEqual(self.client.session['current_document_id'], doc.pk)

    def test_malformed_replacement_preserves_existing_document(self):
        doc_id = self.upload('data.json','{"name":"Original"}').json()['document_id']
        self.embed.reset_mock()
        self.assertEqual(self.upload('data.json','{',document_id=doc_id).status_code,400)
        self.assertIn('Original', Document.objects.get(pk=doc_id).chunks.get().text)
        self.embed.assert_not_called()

    def test_cross_format_replacement_keeps_document_id(self):
        doc_id = self.upload('data.json','{"name":"Original"}').json()['document_id']
        response = self.upload('data.csv','name\nUpdated',document_id=doc_id)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['document_id'],doc_id)
        doc=Document.objects.get(pk=doc_id)
        self.assertEqual(doc.source,'csv')
        self.assertIn('Updated',doc.chunks.get().text)
        self.assertNotIn('Original',doc.chunks.get().text)

    def test_structured_uploads_cannot_replace_another_visitors_file(self):
        other=Client()
        other.post('/accounts/demo/')
        doc_id=self.upload('private.json','{"name":"Other visitor"}',client=other).json()['document_id']
        self.embed.reset_mock()
        for filename,content in [('data.json','{"name":"Bad"}'),('data.csv','name\nBad')]:
            self.assertEqual(self.upload(filename,content,document_id=doc_id).status_code,404)
        self.embed.assert_not_called()
        self.assertIn('Other visitor',Document.objects.get(pk=doc_id).chunks.get().text)

    @override_settings(DEMO_UPLOAD_SESSION_HOURLY_LIMIT=1)
    def test_new_formats_share_existing_upload_budget(self):
        self.assertEqual(self.upload('data.json','{"name":"First"}').status_code,200)
        self.assertEqual(self.upload('data.csv','name\nSecond').status_code,429)
        self.assertEqual(self.embed.call_count,1)

    def test_upload_and_normalized_text_size_limits(self):
        self.assertEqual(self.upload('big.json',b'x'*(2*1024*1024+1)).status_code,400)
        self.assertEqual(self.upload('expanded.json',json.dumps({'long_key'*100:list(range(100))})).status_code,400)
        self.embed.assert_not_called()

    def test_interface_accepts_both_formats(self):
        response=self.client.get('/api/')
        self.assertContains(response,'accept=".pdf,.txt,.md,.json,.csv"')
        self.assertContains(response,'CSV uses commas')
