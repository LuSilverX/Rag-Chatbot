import os

bind = '0.0.0.0:8000'
workers = int(os.environ.get('WEB_WORKERS', '2'))
worker_class = 'gthread'
threads = 2
timeout = 120
graceful_timeout = 120
accesslog = '-'
errorlog = '-'
capture_output = True
# Only Nginx is published; it overwrites the forwarded scheme header.
forwarded_allow_ips = '*'
secure_scheme_headers = {'X-FORWARDED-PROTO': 'https'}
