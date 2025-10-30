#!/usr/bin/env bash
set -euo pipefail

# usage: PORT=8501 bash app.sh
PORT="${PORT:-${1:-8888}}"
DOMINO_DOMAIN="${DOMINO_DOMAIN:-https://fitch.domino-eval.com}"

# 1) Kill anything listening on $PORT via fuser (works where ss/lsof aren’t available)
if command -v fuser &>/dev/null; then
  echo "Killing any process on port $PORT…"
  fuser -k "${PORT}/tcp" || true
  sleep 1
else
  echo "fuser not found, skipping port kill."
fi

# 2) Also kill any stray Streamlit / Flask / app.py
pkill -f streamlit 2>/dev/null && echo "Killed existing Streamlit processes."
pkill -f flask      2>/dev/null && echo "Killed existing Flask processes."
pkill -f app.py     2>/dev/null && echo "Killed existing app.py processes."

# 3) Show the Domino proxy URL
if [ -n "${DOMINO_RUN_HOST_PATH:-}" ]; then
  CLEAN_PATH=$(echo "$DOMINO_RUN_HOST_PATH" | sed 's|/r||g')
  URL="${DOMINO_DOMAIN}${CLEAN_PATH}proxy/${PORT}/"
  echo "========================================="
  echo "Flask URL: $URL"
  echo "========================================="
else
  echo "DOMINO_RUN_HOST_PATH not set — running locally at http://0.0.0.0:${PORT}"
fi

# 4) Launch your Flask app
export FLASK_APP=app.py
python app.py