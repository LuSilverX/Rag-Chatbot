#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
[ -f .env.container ] || { echo 'Configure .env.container on the server first.' >&2; exit 1; }
if [ -f .deploy-state/active.env ]; then
    set -- --env-file .deploy-state/active.env "$@"
fi
exec docker compose --env-file .env.container -f compose.container.yml -f compose.server.yml -f compose.public.yml "$@"
