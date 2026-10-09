#!/bin/sh
# Invoked by systemd twice daily. Certbot decides when renewal is due.
set -eu
cd "$(dirname "$0")/.."
exec 9>/run/lock/rag-certificate.lock
flock -n 9 || exit 0
# The systemd timer already adds jitter; avoid a second delay inside Certbot.
./deploy/public.sh run --rm certbot renew --quiet --no-random-sleep-on-renew
source_dir=/etc/letsencrypt/live/rag-portfolio
target_dir=/etc/rag-portfolio/tls
openssl x509 -in "$source_dir/fullchain.pem" -checkend 3600 -noout
if ! cmp -s "$source_dir/fullchain.pem" "$target_dir/fullchain.pem" || ! cmp -s "$source_dir/privkey.pem" "$target_dir/privkey.pem"; then
    install -m 600 "$source_dir/privkey.pem" "$target_dir/privkey.pem.new"
    install -m 644 "$source_dir/fullchain.pem" "$target_dir/fullchain.pem.new"
    mv "$target_dir/privkey.pem.new" "$target_dir/privkey.pem"
    mv "$target_dir/fullchain.pem.new" "$target_dir/fullchain.pem"
fi
# Always reload: a previous reload may have failed after the files were copied.
./deploy/public.sh exec -T nginx nginx -t
./deploy/public.sh exec -T nginx nginx -s reload
