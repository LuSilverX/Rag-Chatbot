from ipaddress import ip_address
from django.conf import settings
from django.http import JsonResponse


class TrustedClientIPMiddleware:
    """Use the client address overwritten by our private Nginx proxy only."""
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if settings.DJANGO_TRUST_CLIENT_IP:
            value = request.META.get('HTTP_X_REAL_IP')
            if value:
                try:
                    request.META['REMOTE_ADDR'] = str(ip_address(value))
                except ValueError:
                    return JsonResponse({'error': 'invalid_proxy_address'}, status=400)
        return self.get_response(request)
