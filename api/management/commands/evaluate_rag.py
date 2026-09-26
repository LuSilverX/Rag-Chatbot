"""Run a small live evaluation using only the bundled synthetic fixture."""
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.test import RequestFactory

from api.models import QueryLog
from api.views import ask, ingest_text


class Session(dict):
    modified = False


def matches(patterns, text):
    return all(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


class Command(BaseCommand):
    help = "Evaluate 20 synthetic questions with live OpenAI calls; roll back all evaluation database records."

    def add_arguments(self, parser):
        parser.add_argument('--output', default='evaluations/latest.json')

    def handle(self, *args, **options):
        fixture_path = Path(settings.BASE_DIR) / 'evaluations/cases.json'
        fixture = json.loads(fixture_path.read_text())
        factory = RequestFactory()
        session = Session()
        rows = []
        self.stdout.write('Using only synthetic museum text. Live OpenAI API charges apply.')
        with transaction.atomic():
            request = factory.post('/api/ingest_text/', data=json.dumps({
                'title': f'Synthetic evaluation {time.time_ns()}', 'text': fixture['document'],
            }), content_type='application/json')
            request.session = session
            response = ingest_text(request)
            data = json.loads(response.content)
            if response.status_code != 200:
                raise CommandError(f"Evaluation ingestion failed: {data.get('message', data.get('error'))}")
            document_id = data['document_id']
            chunks_created = data['chunks_created']
            for case in fixture['cases']:
                request = factory.post('/api/ask/', data=json.dumps({
                    'document_id': document_id, 'question': case['question'], 'k': 3,
                }), content_type='application/json')
                request.session = session
                started = time.perf_counter()
                response = ask(request)
                elapsed = round((time.perf_counter() - started) * 1000)
                result = json.loads(response.content)
                answer = result.get('answer', '')
                unanswerable = case.get('unanswerable', False)
                abstained = answer.strip().lower().replace('’', "'").rstrip('.') == "i don't know"
                source_text = '\n'.join(source['text'] for source in result.get('sources', []))
                correct = abstained if unanswerable else bool(matches(case['patterns'], answer)) and not abstained
                retrieved = None if unanswerable else bool(matches(case['evidence'], source_text))
                log = QueryLog.objects.filter(document_id=document_id).order_by('-id').first()
                rows.append({
                    'id': case['id'], 'question': case['question'], 'unanswerable': unanswerable,
                    'http_status': response.status_code, 'answer': answer,
                    'answer_check_passed': response.status_code == 200 and correct,
                    'expected_evidence_retrieved': retrieved,
                    'latency_ms': elapsed, 'best_distance': log.best_distance if log else None,
                    'sources': [{key: value for key, value in source.items() if key != 'document_id'} for source in result.get('sources', [])],
                    'error': result.get('error'),
                })
                self.stdout.write(f"{case['id']}: {'PASS' if rows[-1]['answer_check_passed'] else 'FAIL'} ({elapsed} ms)")
                self.stdout.flush()
            transaction.set_rollback(True)

        supported = [row for row in rows if not row['unanswerable']]
        unsupported = [row for row in rows if row['unanswerable']]
        report = {
            'recorded_at': datetime.now(timezone.utc).isoformat(),
            'models': {'embeddings': 'text-embedding-3-small', 'answers': 'gpt-4.1-mini'},
            'fixture': 'cases.json', 'chunks_created': chunks_created, 'k': 3, 'max_distance': 0.95,
            'method': 'Deterministic expected-fact regex checks and exact abstention checks; inspect recorded answers for semantic correctness. This small synthetic set is not a general accuracy benchmark.',
            'summary': {
                'questions': len(rows), 'answer_checks_passed': sum(row['answer_check_passed'] for row in rows),
                'supported_answers_passed': sum(row['answer_check_passed'] for row in supported),
                'supported_questions': len(supported),
                'unsupported_abstentions_passed': sum(row['answer_check_passed'] for row in unsupported),
                'unsupported_questions': len(unsupported),
                'evidence_retrieved': sum(row['expected_evidence_retrieved'] is True for row in supported),
                'median_latency_ms': round(statistics.median(row['latency_ms'] for row in rows)),
                'max_latency_ms': max(row['latency_ms'] for row in rows),
            },
            'results': rows,
        }
        output = Path(options['output'])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
        self.stdout.write(json.dumps(report['summary'], indent=2))
        self.stdout.write(f'Saved {output}. Evaluation documents and logs were rolled back.')
        if not all(row['answer_check_passed'] and row['expected_evidence_retrieved'] is not False for row in rows):
            raise CommandError('Some evaluation checks failed; inspect the report.')
