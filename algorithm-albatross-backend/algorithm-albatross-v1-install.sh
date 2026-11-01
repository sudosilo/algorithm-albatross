#!/usr/bin/env bash
# Algorithm Albatross v1 install for the Frankfurt Ubuntu machine.
# Installs the backend as a service plus a Cloudflare tunnel so the Vercel page can reach it.
set -e
REPO_DIR="$HOME/algorithm-albatross"
BACK="$REPO_DIR/algorithm-albatross-backend"
VENV="$HOME/algorithm-albatross-venv"
LIB="$HOME/algorithm-albatross-library"
DATA="$HOME/algorithm-albatross-data"
PORT=8417
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

$SUDO apt-get update -y
$SUDO apt-get install -y python3 python3-venv python3-pip ffmpeg curl git

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$BACK/algorithm-albatross-v1-requirements.txt"

mkdir -p "$LIB/uploads" "$DATA"
TOKEN_FILE="$DATA/algorithm-albatross-v1-token"
if [ ! -s "$TOKEN_FILE" ]; then
  head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$TOKEN_FILE"
fi
TOKEN="$(cat "$TOKEN_FILE")"

if ! command -v cloudflared >/dev/null 2>&1; then
  ARCH="$(dpkg --print-architecture)"
  curl -fsSL -o /tmp/algorithm-albatross-cloudflared.deb "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$ARCH.deb"
  $SUDO dpkg -i /tmp/algorithm-albatross-cloudflared.deb
fi
CLOUDFLARED="$(command -v cloudflared)"

$SUDO tee /etc/systemd/system/algorithm-albatross-backend.service >/dev/null <<UNIT
[Unit]
Description=Algorithm Albatross backend
After=network-online.target

[Service]
User=$(whoami)
WorkingDirectory=$BACK
Environment=ALBATROSS_TOKEN=$TOKEN
Environment=ALBATROSS_LIBRARY=$LIB
Environment=ALBATROSS_DATA=$DATA
Environment=ALBATROSS_PORT=$PORT
ExecStart=$VENV/bin/python $BACK/algorithm_albatross_v1_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT

$SUDO tee /etc/systemd/system/algorithm-albatross-tunnel.service >/dev/null <<UNIT
[Unit]
Description=Algorithm Albatross Cloudflare tunnel
After=network-online.target algorithm-albatross-backend.service

[Service]
ExecStart=$CLOUDFLARED tunnel --no-autoupdate --url http://127.0.0.1:$PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

$SUDO systemctl daemon-reload
$SUDO systemctl enable algorithm-albatross-backend algorithm-albatross-tunnel
$SUDO systemctl restart algorithm-albatross-backend
$SUDO systemctl restart algorithm-albatross-tunnel

bash "$BACK/algorithm-albatross-v1-url.sh"
