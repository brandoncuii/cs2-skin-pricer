#!/usr/bin/env bash
# Daily backup of data/collector/observations.db to a private GitHub repo.
# Designed to run from cron on the Raspberry Pi.
#
# Env vars (loaded from .env if present):
#   BACKUP_REPO_URL   — remote URL of the private backup repo
#                       (e.g. https://github.com/brandoncuii/cs2-collector-backup.git)
#   BACKUP_LOCAL_DIR  — local clone path (default: ~/cs2-backup-repo)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DB_FILE="$REPO_DIR/data/collector/observations.db"
LOG_FILE="$REPO_DIR/data/collector/backup.log"

# ---------- helpers ----------------------------------------------------------
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$LOG_FILE"; }

die() { log "ERROR: $*"; exit 1; }

# ---------- load config ------------------------------------------------------
if [[ -f "$REPO_DIR/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "$REPO_DIR/.env"; set +a
fi

BACKUP_REPO_URL="${BACKUP_REPO_URL:-}"
BACKUP_LOCAL_DIR="${BACKUP_LOCAL_DIR:-$HOME/cs2-backup-repo}"

[[ -n "$BACKUP_REPO_URL" ]] || die "BACKUP_REPO_URL is not set. Add it to $REPO_DIR/.env"
[[ -f "$DB_FILE" ]]         || die "Database not found at $DB_FILE"

# ---------- ensure local clone exists ----------------------------------------
if [[ ! -d "$BACKUP_LOCAL_DIR/.git" ]]; then
    log "First run — cloning $BACKUP_REPO_URL → $BACKUP_LOCAL_DIR"
    git clone "$BACKUP_REPO_URL" "$BACKUP_LOCAL_DIR"
fi

cd "$BACKUP_LOCAL_DIR"

# Make sure we're up-to-date (handles remote changes / force-pushes).
git fetch origin
git reset --hard origin/"$(git rev-parse --abbrev-ref HEAD)" 2>/dev/null || true

# ---------- copy DB ----------------------------------------------------------
cp "$DB_FILE" "$BACKUP_LOCAL_DIR/observations.db"

# ---------- commit only if something changed ---------------------------------
if git diff --quiet -- observations.db && git diff --cached --quiet -- observations.db; then
    log "No changes in observations.db — skipping commit."
    exit 0
fi

ROW_COUNT="$(sqlite3 "$DB_FILE" 'SELECT COUNT(*) FROM observations;' 2>/dev/null || echo '?')"
TIMESTAMP="$(date -u '+%Y-%m-%d %H:%M UTC')"
COMMIT_MSG="backup $TIMESTAMP — $ROW_COUNT rows"

git add observations.db
git commit -m "$COMMIT_MSG"
git push

log "Backup committed: $COMMIT_MSG"
