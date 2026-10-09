# Ubuntu server setup

The full app runs on a 1 GB Ubuntu server, with one worker and 2 GB swap.

- Run `sudo sh deploy/setup-server.sh` on Ubuntu to install Docker and add 2 GB swap.
- Copy the app to `/opt/rag-portfolio`. Configure `.env.container` from
  `deploy/server.env.example`, with unique server secrets and file permissions `600`.
- Place the TLS certificate and key in `/etc/rag-portfolio/tls/fullchain.pem` and
  `privkey.pem`. The key must be readable only by root. Private verification uses
  a temporary self-signed certificate for localhost; replace it with a trusted
  certificate for the public domain before launch.
- Start with `sudo ./deploy/server.sh up --build -d --wait` from the app directory.
- Seed the demo once: `sudo ./deploy/server.sh exec -T web python manage.py prepare_demo`.
  This uses the OpenAI API. Existing ready samples are skipped.
- Inspect services with `sudo ./deploy/server.sh ps` and memory with
  `sudo docker stats --no-stream` and `free -m`.

Only loopback ports 8088 (HTTP health/redirect) and 8443 (HTTPS) are published.
Use an SSH tunnel to access HTTPS privately. Verify the certificate explicitly;
do not disable certificate checking. Django debug is off, secure cookies are on,
and one Gunicorn worker handles requests. PostgreSQL uses persistent storage.

`sudo ./deploy/server.sh stop` stops the app without deleting its data.
Do not use `down -v` unless you intend to delete the database.

## Public HTTPS

The deployed address is `https://77.113.94.93`. No domain purchase is required.
Public mode uses `sudo ./deploy/public.sh` instead of `server.sh` for every
Compose command. It publishes only ports 80 and 443; the database stays private.
Add the public IP/domain to `DJANGO_ALLOWED_HOSTS` and its HTTPS origin to
`DJANGO_CSRF_TRUSTED_ORIGINS` in the server environment.

Certbot stores certificates under `/etc/letsencrypt/live/rag-portfolio` and
validates ownership using `/var/www/certbot` on port 80. The IP certificate uses
Let's Encrypt's six-day `shortlived` profile. `deploy/renew-certificate.sh`
checks renewal, copies the current certificate into Nginx's TLS directory,
validates Nginx configuration, and reloads it without stopping the app.

The installed `rag-certificate.timer` runs twice daily with a randomized delay.
Check with `sudo systemctl list-timers rag-certificate.timer` and
`sudo journalctl -u rag-certificate.service`. Test renewal with
`sudo ./deploy/public.sh run --rm certbot renew --dry-run --no-random-sleep-on-renew`; run the full renewal
and reload path with `sudo systemctl start rag-certificate.service`.

Keep port 80 open for renewals and HTTP-to-HTTPS redirects. SSH is restricted to
the administrator's IP in Lightsail; update that rule if the address changes.
Local development still uses `deploy/compose.sh`. Never put server secrets,
private SSH keys, or TLS private keys in Git.
