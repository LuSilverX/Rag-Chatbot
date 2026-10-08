"""Run once for a scheduler, or as a continuously running cleanup worker."""
import time
import os
from pathlib import Path
from django.core.management.base import BaseCommand
from django.db import close_old_connections
from api.demo import cleanup_expired_workspaces


class Command(BaseCommand):
    help = 'Delete expired visitor workspaces, including their documents, chunks and history.'

    def add_arguments(self, parser):
        parser.add_argument('--heartbeat-file', help='Update this file after each successful cleanup pass.')
        parser.add_argument('--watch', action='store_true', help='Repeat every 60 seconds while this worker runs.')

    def handle(self, *args, **options):
        while True:
            if options["watch"]:
                close_old_connections()
            try:
                count = cleanup_expired_workspaces()
                if options['heartbeat_file']:
                    heartbeat = Path(options['heartbeat_file'])
                    temporary = heartbeat.with_suffix('.tmp')
                    temporary.write_text(str(time.time()))
                    os.replace(temporary, heartbeat)
                if count or not options['watch']:
                    self.stdout.write(f'Deleted {count} expired visitor workspaces.')
                    self.stdout.flush()
            except Exception:
                if not options['watch']:
                    raise
                self.stderr.write('Workspace cleanup failed; retrying in 60 seconds.')
                self.stderr.flush()
            if not options['watch']:
                return
            time.sleep(60)
