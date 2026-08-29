#!/bin/sh
set -eu

mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HF_HOME"

exec /opt/dj-venv/bin/python \
  -m data_juicer.tools.plan_flow.container_entry \
  "$@"
