#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -f .env.container ]; then
    echo 'Copy deploy/container.env.example to .env.container and fill in your secrets first.' >&2
    exit 1
fi
exec docker compose --env-file .env.container -f compose.container.yml "$@"
