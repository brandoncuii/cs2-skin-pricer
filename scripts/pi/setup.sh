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

LOCK_FILE="/tmp/cs2-collector.lock"

# --- collector crons (flock prevents overlapping runs) -----------------------
# Locked skins (fast) — every 3h.
CRON_LOCKED="17 */3 * * * flock --nonblock $LOCK_FILE bash -c 'cd $REPO_DIR && PYTHONPATH=. .venv/bin/python scripts/collect.py >> data/collector/collect_locked.log 2>&1'"

# Full sweep (slow) — every 12h, offset to minute 47.
CRON_FULL="47 */12 * * * flock --nonblock $LOCK_FILE bash -c 'cd $REPO_DIR && PYTHONPATH=. .venv/bin/python scripts/collect.py --full --min-qty 1 >> data/collector/collect_full.log 2>&1'"

# --- backup cron (daily at 06:00 UTC) ----------------------------------------
BACKUP_CRON="0 6 * * * cd $REPO_DIR && bash scripts/pi/backup_db.sh >> data/collector/backup.log 2>&1"

# Install all crons, replacing old entries if present.
( crontab -l 2>/dev/null \
    | grep -vF "scripts/collect.py" \
    | grep -vF "scripts/pi/backup_db.sh" \
    || true
  echo "$CRON_LOCKED"
  echo "$CRON_FULL"
  echo "$BACKUP_CRON"
) | crontab -

echo "Cron entries installed:"
crontab -l | grep -E 'collect\.py|backup_db\.sh'
cat <<'EOF'

Dual-schedule collector (flock prevents overlapping runs):
  Locked skins : every 3h  (minute 17) — 4 names, 3 pages each
  Full sweep   : every 12h (minute 47) — all knives, --min-qty 1 gated
  Lock file    : /tmp/cs2-collector.lock (flock --nonblock; skip if held)
  Logs         : data/collector/collect_locked.log
                 data/collector/collect_full.log

Manual steps to finish:
  1. API key:        echo 'CSFLOAT_API_KEY=<your key>' >> .env
  2. Carry over the Mac's DB so its baseline snapshot isn't lost:
                     scp <mac-host>:coding-projects/cs2-skin-pricer/data/collector/observations.db data/collector/
  3. Smoke test:     PYTHONPATH=. .venv/bin/python scripts/collect.py

  4. On the MAC, disable the old agent (two collectors = two diverging DBs):
                     launchctl unload ~/Library/LaunchAgents/com.cs2pricer.collector.plist
                     rm ~/Library/LaunchAgents/com.cs2pricer.collector.plist

  === Daily backup setup ===

  5. Create a PRIVATE repo for backups (one-time):
       gh repo create cs2-collector-backup --private --clone=false

  6. Generate a fine-grained GitHub PAT with push access to that repo:
       → https://github.com/settings/personal-access-tokens/new
       • Repository access → Only select repositories → cs2-collector-backup
       • Permissions → Contents: Read and write
       • Save the token

  7. Configure git credentials on the Pi so pushes work:
       git config --global credential.helper store
       # Do a test clone/push — git will prompt once, then remember:
       git clone https://github.com/brandoncuii/cs2-collector-backup.git ~/cs2-backup-repo

  8. Add backup env vars to .env:
       echo 'BACKUP_REPO_URL=https://github.com/brandoncuii/cs2-collector-backup.git' >> .env
       echo 'BACKUP_LOCAL_DIR=~/cs2-backup-repo' >> .env

  9. Smoke-test the backup:
       bash scripts/pi/backup_db.sh

When training v1.5 on the Mac, pull the DB back first:
  rsync <pi-host>:cs2-skin-pricer/data/collector/observations.db ~/coding-projects/cs2-skin-pricer/data/collector/
EOF
