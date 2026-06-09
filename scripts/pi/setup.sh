#!/usr/bin/env bash
# One-time setup for running the collector on an always-on Raspberry Pi
# (replaces the Mac launchd agent, which skips runs while the laptop sleeps).
#
# Run ON the Pi, after:  git clone https://github.com/brandoncuii/cs2-skin-pricer.git && cd cs2-skin-pricer
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

# Minimal deps — collection only. Training (lightgbm), the API, and the UI
# stay on the Mac; the Pi just accumulates data/collector/observations.db.
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet requests python-dotenv pandas

mkdir -p data/collector

# Every 12h. Cron has no missed-run catch-up, but the Pi is always on so
# that doesn't matter (it's why we're here instead of on the laptop).
CRON_LINE="17 */12 * * * cd $REPO_DIR && PYTHONPATH=. .venv/bin/python scripts/collect.py >> data/collector/collect.log 2>&1"
( crontab -l 2>/dev/null | grep -vF "scripts/collect.py" || true; echo "$CRON_LINE" ) | crontab -

echo "Cron entry installed:"
crontab -l | grep collect.py
cat <<'EOF'

Manual steps to finish:
  1. API key:        echo 'CSFLOAT_API_KEY=<your key>' > .env
  2. Carry over the Mac's DB so its baseline snapshot isn't lost:
                     scp <mac-host>:coding-projects/cs2-skin-pricer/data/collector/observations.db data/collector/
  3. Smoke test:     PYTHONPATH=. .venv/bin/python scripts/collect.py
  4. On the MAC, disable the old agent (two collectors = two diverging DBs):
                     launchctl unload ~/Library/LaunchAgents/com.cs2pricer.collector.plist
                     rm ~/Library/LaunchAgents/com.cs2pricer.collector.plist

When training v1.5 on the Mac, pull the DB back first:
  rsync <pi-host>:cs2-skin-pricer/data/collector/observations.db ~/coding-projects/cs2-skin-pricer/data/collector/
EOF
