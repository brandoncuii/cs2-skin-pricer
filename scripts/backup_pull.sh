#!/usr/bin/env bash
# Daily pull-backup of the Pi's collector DB onto the Mac (run by launchd —
# see scripts/launchd/com.cs2pricer.backup.plist).
#
# Pull (not push) because the Mac's Remote Login is off. The Pi side takes a
# crash-consistent snapshot first via Python's sqlite3 backup API (safe while
# the collector is mid-write; no sqlite3 CLI needed on the Pi), then we rsync
# the snapshot and keep 30 dated gzipped copies for point-in-time recovery.
#
# Prereq (one-time): passwordless SSH to the Pi — ssh-copy-id pi@10.0.0.241
set -euo pipefail

PI="pi@10.0.0.241"
PI_REPO="cs2-skin-pricer"
BACKUP_DIR="$HOME/coding-projects/cs2-skin-pricer/data/collector/backups"
STAMP="$(date +%Y%m%d)"

mkdir -p "$BACKUP_DIR"

ssh -o BatchMode=yes -o ConnectTimeout=10 "$PI" \
    "cd $PI_REPO && python3 -" <<'PY'
import sqlite3
src = sqlite3.connect("data/collector/observations.db")
dst = sqlite3.connect("data/collector/observations-snapshot.db")
with dst:
    src.backup(dst)
dst.close()
src.close()
PY

rsync -a "$PI:$PI_REPO/data/collector/observations-snapshot.db" \
      "$BACKUP_DIR/observations-$STAMP.db"
gzip -f "$BACKUP_DIR/observations-$STAMP.db"

# Keep the last 30 dated copies.
find "$BACKUP_DIR" -name 'observations-*.db.gz' -mtime +30 -delete

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] backup ok: observations-$STAMP.db.gz" \
     "($(du -h "$BACKUP_DIR/observations-$STAMP.db.gz" | cut -f1))"
