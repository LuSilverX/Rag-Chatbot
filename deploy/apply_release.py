"""Apply a tested image and restore the previous release on failure.

Installed outside the app at /usr/local/lib/rag-deploy. Database migrations must
remain compatible with the previous image: rollback never reverses database data.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import urllib.request

FILES = (
    'compose.container.yml', 'compose.server.yml', 'compose.public.yml',
    'deploy/public.sh', 'deploy/nginx.conf', 'deploy/nginx.server.conf',
    'deploy/compose.sh', 'deploy/server.sh', 'deploy/gunicorn.conf.py',
    'deploy/renew-certificate.sh',
)


def command(args, *, cwd=None, env=None):
    return subprocess.run(args, cwd=cwd, env=env, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def copy_config(source, destination):
    for name in FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)


def compose(app, tag, *args):
    env = dict(os.environ, APP_IMAGE_TAG=tag)
    return command(['sh', './deploy/public.sh', *args], cwd=app, env=env)


def healthy(app, tag, url):
    compose(app, tag, 'up', '-d', '--no-build', '--wait', '--wait-timeout', '90')
    # Bind-mounted Nginx files need an explicit reload when the container stays up.
    compose(app, tag, 'exec', '-T', 'nginx', 'nginx', '-t')
    compose(app, tag, 'exec', '-T', 'nginx', 'nginx', '-s', 'reload')
    for path in ('/health/', '/accounts/login/', '/static/admin/css/base.css'):
        with urllib.request.urlopen(url + path, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError('Public readiness check failed')
            if path == '/health/' and json.load(response) != {'status': 'ready'}:
                raise RuntimeError('Application is not ready')


def deploy(candidate, tag, app=Path('/opt/rag-portfolio'), url='https://77.113.94.93'):
    if not re.fullmatch('[0-9a-f]{40}', tag):
        raise ValueError('Expected a full Git commit SHA')
    if any(not (candidate / name).is_file() for name in FILES):
        raise ValueError('Incomplete release configuration')
    state = app / '.deploy-state'
    state.mkdir(mode=0o700, exist_ok=True)
    with (state / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current_id = compose(app, 'local', 'ps', '-q', 'web').strip()
        previous_image = command(['docker', 'inspect', current_id, '--format', '{{.Config.Image}}']).strip()
        if not previous_image.startswith('rag-chatbot:'):
            raise RuntimeError('Cannot identify the previous application image')
        previous_tag = previous_image.split(':', 1)[1]
        command(['docker', 'load', '-i', str(candidate / 'image.tar.gz')])
        revision = command(['docker', 'image', 'inspect', 'rag-chatbot:' + tag,
                            '--format', '{{index .Config.Labels "org.opencontainers.image.revision"}}']).strip()
        if revision != tag:
            raise ValueError('Image revision does not match the release')
        backup = Path(tempfile.mkdtemp(prefix='previous-', dir=state))
        copy_config(app, backup)
        try:
            copy_config(candidate, app)
            healthy(app, tag, url)
        except BaseException:
            copy_config(backup, app)
            try:
                healthy(app, previous_tag, url)
            except BaseException as rollback_error:
                raise RuntimeError('Deployment failed and rollback needs attention') from rollback_error
            raise RuntimeError('Deployment failed; previous image and configuration restored') from None
        else:
            temporary = state / 'active.env.new'
            temporary.write_text('APP_IMAGE_TAG=' + tag + '\n')
            temporary.replace(state / 'active.env')
            (state / 'previous-image').write_text(previous_image + '\n')
            previous_config = state / 'previous-config'
            if previous_config.exists():
                shutil.rmtree(previous_config)
            backup.replace(previous_config)
            # Retain current and previous images; bounded storage on the small host.
            images = command(['docker', 'image', 'ls', 'rag-chatbot', '--format', '{{.Tag}}']).splitlines()
            for old in images:
                if re.fullmatch('[0-9a-f]{40}', old) and old not in (tag, previous_tag):
                    subprocess.run(['docker', 'image', 'rm', 'rag-chatbot:' + old], check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {'status': 'deployed', 'revision': tag, 'previous_image': previous_image}
        finally:
            if backup.exists():
                shutil.rmtree(backup)
