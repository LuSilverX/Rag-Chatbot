"""Prepare approved portfolio samples; never publish existing private documents."""
import json
import uuid
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.test import RequestFactory
from api.models import DemoAccount, Document
from api.views import store_document


class Command(BaseCommand):
    help = 'Load two bundled sample documents using live embeddings, and create a restricted demo identity. Existing ready samples are skipped.'

    def handle(self, *args, **options):
        fixture = json.loads((settings.BASE_DIR / 'evaluations/cases.json').read_text())
        samples = [
            ('Harbor Museum — Demo', fixture['document']),
            ('How RAG Works — Demo', (settings.BASE_DIR / 'sample_docs/demo.txt').read_text()),
        ]
        for title, text in samples:
            existing = Document.objects.filter(title=title, source='portfolio_demo', is_demo=True, chunks__isnull=False).first()
            if existing:
                self.stdout.write(f'Already ready: {title}')
                continue
            request = RequestFactory().post('/')
            request.session = {}
            self.stdout.write(f'Embedding bundled sample: {title} (live API call)')
            response = store_document(request, title, text, 'portfolio_demo')
            data = json.loads(response.content)
            Document.objects.filter(pk=data['document_id']).update(is_demo=True)
        with transaction.atomic():
            account = DemoAccount.objects.select_related('user').first()
            if not account:
                user = get_user_model().objects.create_user(username='portfolio-demo-' + uuid.uuid4().hex[:16])
                DemoAccount.objects.create(user=user)
            elif account.user.is_staff or account.user.is_superuser or account.user.has_usable_password() or not account.user.is_active:
                raise CommandError('The demo identity must be active, non-staff, non-superuser, and have an unusable password.')
        self.stdout.write(self.style.SUCCESS('Demo ready. Use Try the demo on the sign-in page.'))
