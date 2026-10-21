#!/usr/bin/env bash
set -euo pipefail

if pgrep -x kit >/dev/null 2>&1; then
  pkill -TERM -x kit
  echo "Isaac Sim viewer stopped."
else
  echo "Isaac Sim viewer is not running."
fi
