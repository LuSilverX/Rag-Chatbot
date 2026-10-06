"""Demo authorization and database-backed quotas shared by all app workers."""
import time
from datetime import datetime, timezone
from django.conf import settings
from django.db import transaction
from django.utils.crypto import salted_hmac
from .models import DemoAccount, DemoQuota, Document, VisitorWorkspace, Chunk
from django.utils import timezone as django_timezone
from datetime import timedelta
from django.core.exceptions import ValidationError


class DemoLimitExceeded(Exception):
    def __init__(self, retry_after):
        self.retry_after = retry_after


def is_demo(request):
    if not hasattr(request, '_is_demo'):
        request._is_demo = getattr(request, "user", None) is not None and request.user.is_authenticated and DemoAccount.objects.filter(user_id=request.user.pk).exists()
    return request._is_demo


class WorkspaceExpired(Exception):
    pass


def get_workspace(request):
    if not is_demo(request):
        return None
    if hasattr(request, '_workspace'):
        return request._workspace
    try:
        workspace = VisitorWorkspace.objects.filter(
            pk=request.session.get('workspace_id'),
            session_digest=fingerprint(request.session.session_key or ''),
            expires_at__gt=django_timezone.now(),
        ).first()
    except (ValidationError, ValueError):
        workspace = None
    if workspace is None:
        raise WorkspaceExpired()
    request._workspace = workspace
    return workspace


def workspace_is_active(workspace):
    return VisitorWorkspace.objects.filter(pk=workspace.pk, expires_at__gt=django_timezone.now()).exists()


def lock_workspace(workspace):
    current = VisitorWorkspace.objects.select_for_update().filter(pk=workspace.pk, expires_at__gt=django_timezone.now()).first()
    if current is None:
        raise WorkspaceExpired()
    return current


def visible_documents(request):
    workspace = get_workspace(request)
    return Document.objects.filter(workspace=workspace)


def create_workspace(request):
    with transaction.atomic():
        workspace = VisitorWorkspace.objects.create(
            session_digest=fingerprint(request.session.session_key),
            expires_at=django_timezone.now() + timedelta(hours=24),
        )
        # Each visitor receives editable copies, including existing vectors: no API calls.
        samples = Document.objects.filter(is_demo=True, workspace__isnull=True).order_by('id')[:2]
        first_id = None
        for sample in samples:
            copy = Document.objects.create(workspace=workspace, title=sample.title, source=sample.source)
            Chunk.objects.bulk_create([
                Chunk(document=copy, chunk_index=chunk.chunk_index, text=chunk.text, embedding=chunk.embedding)
                for chunk in sample.chunks.all()
            ])
            first_id = first_id or copy.pk
    request.session['workspace_id'] = str(workspace.pk)
    request.session['current_document_id'] = first_id
    request.session.set_expiry(workspace.expires_at)
    return workspace


def cleanup_expired_workspaces():
    count = 0
    # Small batches and per-workspace locks avoid racing in-flight document writes.
    for workspace_id in VisitorWorkspace.objects.filter(expires_at__lte=django_timezone.now()).values_list('pk', flat=True)[:100]:
        with transaction.atomic():
            workspace = VisitorWorkspace.objects.select_for_update(skip_locked=True).filter(pk=workspace_id, expires_at__lte=django_timezone.now()).first()
            if workspace is not None:
                workspace.delete()  # Cascades to documents, vectors and query history only.
                count += 1
    return count


def fingerprint(value):
    return salted_hmac('portfolio-demo', value).hexdigest()


def consume_quota(request, *, entry=False, upload=False):
    now = int(time.time())
    # Never trust a visitor-supplied X-Forwarded-For header.
    address = fingerprint(request.META.get('REMOTE_ADDR', 'unknown'))
    if entry:
        limits = [('entry:global', 86400, 500), ('entry:ip:' + address, 3600, 20)]
    elif upload:
        limits = [
            ('upload:global', 86400, settings.DEMO_UPLOAD_DAILY_LIMIT),
            ('upload:ip:' + address, 3600, settings.DEMO_UPLOAD_IP_HOURLY_LIMIT),
            ('upload:session:' + fingerprint(request.session.session_key or ''), 3600, settings.DEMO_UPLOAD_SESSION_HOURLY_LIMIT),
        ]
    else:
        limits = [
            ('ask:global', 86400, settings.DEMO_DAILY_LIMIT),
            ('ask:ip:' + address, 3600, settings.DEMO_IP_HOURLY_LIMIT),
            ('ask:session:' + fingerprint(request.session.session_key or ''), 3600, settings.DEMO_SESSION_HOURLY_LIMIT),
        ]
    # Unique keys plus row locks prevent concurrent requests exceeding a quota.
    # All counters commit together, before any billable call starts.
    with transaction.atomic():
        for key, seconds, maximum in sorted(limits):
            window = now // seconds
            end = (window + 1) * seconds
            counter, _ = DemoQuota.objects.select_for_update().get_or_create(
                key=key, defaults={'window': window, 'expires_at': datetime.fromtimestamp(end, timezone.utc)},
            )
            if counter.window != window:
                counter.window, counter.count = window, 0
            if counter.count >= maximum:
                raise DemoLimitExceeded(end - now)
            counter.count += 1
            counter.expires_at = datetime.fromtimestamp(end, timezone.utc)
            counter.save(update_fields=['window', 'count', 'expires_at'])
