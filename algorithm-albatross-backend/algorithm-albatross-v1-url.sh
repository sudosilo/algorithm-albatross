#!/usr/bin/env bash
# Prints the current tunnel address and token for the Algorithm Albatross page.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
URL=""
for i in $(seq 1 40); do
  URL="$($SUDO journalctl -u algorithm-albatross-tunnel --no-pager -n 300 2>/dev/null | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | tail -1)"
  if [ -n "$URL" ]; then break; fi
  sleep 3
done
echo
echo "*************** SAVE THIS ***************"
echo "SERVER URL:  ${URL#https://}"
echo "TOKEN:       $(cat "$HOME/algorithm-albatross-data/algorithm-albatross-v1-token")"
echo "LIBRARY:     $HOME/algorithm-albatross-library"
echo "*****************************************"
