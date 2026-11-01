#!/usr/bin/env bash
# Pulls the newest code and restarts the backend. The tunnel keeps its address.
set -e
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
cd "$HOME/algorithm-albatross"
git pull
"$HOME/algorithm-albatross-venv/bin/pip" install -r algorithm-albatross-backend/algorithm-albatross-v1-requirements.txt
$SUDO systemctl restart algorithm-albatross-backend
echo "Algorithm Albatross backend updated and restarted."
