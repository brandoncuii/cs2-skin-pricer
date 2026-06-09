# Raspberry Pi Collector Setup

The Pi runs two cron jobs:

| Job | Schedule | Script |
|-----|----------|--------|
| **Collector** | Every 12 h (`17 */12 * * *`) | `scripts/collect.py` |
| **DB Backup** | Daily 06:00 UTC (`0 6 * * *`) | `scripts/pi/backup_db.sh` |

## First-time setup

```bash
git clone https://github.com/brandoncuii/cs2-skin-pricer.git && cd cs2-skin-pricer
bash scripts/pi/setup.sh      # installs venv, deps, both cron jobs
```

Then follow the manual steps printed by the script (API key, DB migration,
backup repo creation, etc.).

## Collector

`scripts/collect.py` fetches recent sold-price data from CSFloat and appends
rows to `data/collector/observations.db` (SQLite). Logs go to
`data/collector/collect.log`.

## Daily Backup

`scripts/pi/backup_db.sh` copies `observations.db` into a dedicated private
GitHub repo and pushes a timestamped commit. This gives you point-in-time
history — not just the latest snapshot — so you can recover from SD card
failure or corruption.

### Configuration

Add these to `.env` in the repo root:

```bash
BACKUP_REPO_URL=https://github.com/brandoncuii/cs2-collector-backup.git
BACKUP_LOCAL_DIR=~/cs2-backup-repo   # optional, this is the default
```

### How it works

1. Sources `.env` for `BACKUP_REPO_URL` and `BACKUP_LOCAL_DIR`.
2. On first run, clones the backup repo to `BACKUP_LOCAL_DIR`.
3. Copies `observations.db` into the clone.
4. If the file changed, commits with a message like
   `backup 2025-06-10 06:00 UTC — 48230 rows` and pushes.
5. If nothing changed, exits silently (no empty commits).
6. Logs to `data/collector/backup.log`.

### One-time backup repo setup

1. Create a private repo:
   ```bash
   gh repo create cs2-collector-backup --private --clone=false
   ```
2. Generate a fine-grained PAT scoped to that repo with **Contents: Read and
   write** permission:
   https://github.com/settings/personal-access-tokens/new
3. On the Pi, store credentials so pushes work unattended:
   ```bash
   git config --global credential.helper store
   git clone https://github.com/brandoncuii/cs2-collector-backup.git ~/cs2-backup-repo
   # git will prompt for username + PAT once, then remember
   ```
4. Smoke-test:
   ```bash
   bash scripts/pi/backup_db.sh
   ```

## Pulling the DB to your Mac for training

```bash
rsync <pi-host>:cs2-skin-pricer/data/collector/observations.db \
      ~/coding-projects/cs2-skin-pricer/data/collector/
```
