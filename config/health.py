"""Readiness checks with no authentication, AI calls, or diagnostic disclosures."""
import logging
from django.db import connection, DatabaseError
from django.http import JsonResponse
from django.views.decorators.http import require_GET

logger = logging.getLogger(__name__)


@require_GET
def health(request):
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT '[1,0]'::vector <=> '[1,0]'::vector")
            cursor.fetchone()
            cursor.execute('SELECT workspace_id FROM api_document LIMIT 1')
            cursor.fetchone()
        response = JsonResponse({'status': 'ready'})
    except DatabaseError:
        logger.warning('Database readiness check failed')
        response = JsonResponse({'status': 'unavailable'}, status=503)
    response['Cache-Control'] = 'no-store'
    return response
