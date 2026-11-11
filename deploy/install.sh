#!/usr/bin/env bash
# Puts the Fire LangGraph Agent online at https://HOSTNAME on this server (Apache + certbot already installed).
#   sudo bash deploy/install.sh agent.papillon.website [USER]
# What it changes (as root):
#   1. Gets a Let's Encrypt certificate for HOSTNAME (webroot /var/www/html/public; reloads Apache on renewal).
#   2. Installs systemd service agent-api (uvicorn on 127.0.0.1:8310, runs as USER), started now and on every boot.
#   3. Adds and enables the Apache site agent.conf (HTTPS proxy to 127.0.0.1:8310).
# It does not change the firewall (ports 80/443 are already open) or any other site.
# Safe to re-run: it also deploys code/settings changes (restarts agent-api).
# Undo: systemctl disable --now agent-api; a2dissite agent; systemctl reload apache2;
#       rm /etc/systemd/system/agent-api.service /etc/apache2/sites-available/agent.conf
set -euo pipefail

HOST="${1:?usage: install.sh HOSTNAME [USER]}"
USER_NAME="${2:-${SUDO_USER:-linuxuser}}"
[[ "$HOST" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "Bad hostname: $HOST"; exit 1; }
[ "$(id -u)" -eq 0 ] || { echo "Run with sudo"; exit 1; }
HOME_DIR="$(getent passwd "$USER_NAME" | cut -d: -f6)"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ENV_FILE="$REPO/.env"
[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE"; exit 1; }
grep -qE '^AGENT_ACCESS_TOKEN=.{16,}' "$ENV_FILE" || { echo ".env needs AGENT_ACCESS_TOKEN (16+ chars) before going public"; exit 1; }
grep -qE '^EVAL_ADMIN_TOKEN=.{16,}' "$ENV_FILE" || { echo ".env needs EVAL_ADMIN_TOKEN (16+ chars)"; exit 1; }
[ -x "$REPO/.venv/bin/uvicorn" ] || { echo "Missing $REPO/.venv (pip install -r requirements.txt)"; exit 1; }
sudo -u "$USER_NAME" mkdir -p "$REPO/data"

render() { sed -e "s|@HOST@|$HOST|g" -e "s|@USER@|$USER_NAME|g" -e "s|@HOME@|$HOME_DIR|g" -e "s|@REPO@|$REPO|g" "$1"; }

echo "1/4 Certificate for $HOST"
if [ ! -f "/etc/letsencrypt/live/$HOST/fullchain.pem" ]; then
  certbot certonly --webroot -w /var/www/html/public -d "$HOST" --non-interactive --agree-tos \
    --register-unsafely-without-email --deploy-hook "systemctl reload apache2"
else
  echo "   already present"
fi

echo "2/4 Service"
# A copy started by hand during setup would hold the database; stop it (only uvicorn on port 8010 from this repo).
pkill -u "$USER_NAME" -f "uvicorn app.main:app --host 127.0.0.1 --port 8010" || true
render "$HERE/agent-api.service" > /etc/systemd/system/agent-api.service
systemctl daemon-reload
systemctl enable agent-api
systemctl restart agent-api

echo "3/4 Apache site"
render "$HERE/agent.conf" > /etc/apache2/sites-available/agent.conf
a2enmod -q headers proxy proxy_http ssl >/dev/null
a2ensite -q agent
apache2ctl configtest
systemctl reload apache2

echo "4/4 Check"
for i in $(seq 1 40); do curl -fsS "https://$HOST/api/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "https://$HOST/api/health" && echo && echo "DONE: https://$HOST is online"
