"""Authenticated deployment uploads, reachable through the HTTPS Nginx proxy."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import tarfile
import tempfile
from apply_release import FILES, deploy

MAX_UPLOAD = 512 * 1024 * 1024


def extract_release(archive, target):
    with tarfile.open(archive, 'r:*') as bundle:
        members = bundle.getmembers()
        expected = set(FILES) | {'image.tar.gz'}
        names = [member.name.removeprefix('./') for member in members]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError('Unexpected or missing release files')
        if any(not member.isfile() for member in members) or sum(m.size for m in members) > MAX_UPLOAD:
            raise ValueError('Invalid release archive')
        bundle.extractall(target, filter='data')


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        token = Path('/etc/rag-portfolio/deploy-token').read_text().strip()
        if self.path != '/_deploy/' or not hmac.compare_digest(
            self.headers.get('Authorization', ''), 'Bearer ' + token
        ):
            self.reply(401, {'error': 'unauthorized'})
            return
        tag = self.headers.get('X-Release-SHA', '')
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            length = 0
        if not re.fullmatch('[0-9a-f]{40}', tag) or not 0 < length <= MAX_UPLOAD:
            self.reply(400, {'error': 'invalid_release'})
            return
        self.connection.settimeout(120)
        try:
            with tempfile.TemporaryDirectory(prefix='rag-release-') as directory:
                root = Path(directory)
                archive = root / 'release.tar'
                with archive.open('wb') as output:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(65536, remaining))
                        if not chunk:
                            raise ValueError('Incomplete upload')
                        output.write(chunk)
                        remaining -= len(chunk)
                candidate = root / 'candidate'
                candidate.mkdir()
                extract_release(archive, candidate)
                result = deploy(candidate, tag)
                self.reply(200, result)
        except (ValueError, tarfile.TarError):
            self.reply(400, {'error': 'invalid_release'})
        except BlockingIOError:
            self.reply(409, {'error': 'deployment_in_progress'})
        except Exception as error:
            # Exception strings from commands can contain diagnostics; only expose
            # our fixed deployment/rollback result. Detailed errors stay in journald.
            import traceback
            traceback.print_exc()
            message = str(error) if isinstance(error, RuntimeError) else 'Deployment failed before activation'
            self.reply(503, {'error': message})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind', default='172.17.0.1')
    args = parser.parse_args()
    ThreadingHTTPServer((args.bind, 9000), Handler).serve_forever()
