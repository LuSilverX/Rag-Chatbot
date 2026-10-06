from django.core.management.base import BaseCommand
from django.utils import timezone
from api.models import DemoQuota


class Command(BaseCommand):
    help = 'Remove expired demo quota counters. Active budgets are preserved.'

    def handle(self, *args, **options):
        count, _ = DemoQuota.objects.filter(expires_at__lte=timezone.now()).delete()
        self.stdout.write(f'Removed {count} expired counters.')
