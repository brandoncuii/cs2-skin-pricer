# Migrating the collector from the Raspberry Pi to a VPS

## Why

The collector lives on the Pi at `10.0.0.241` — on your home LAN, so it shares
your household's public IP. CSFloat's website throttles by IP, so the
collector's ~4,300 requests/day get *your own browsing* rate-limited. Moving the
collector to a VPS gives it its own IP: your home connection only carries your
browsing and the local app, and the collector gets a full API key's budget to
itself.

**What this fixes:** the IP/browsing throttle, and the app-vs-collector
contention on your home connection.

**What this does NOT fix:** the API *key* limit (~200 requests/window, very
roughly 5,000–8,000/day). That ceiling is per-key and unchanged by the move —
one key still caps total throughput. Comprehensive all-knife *sold*-price
collection eventually needs multiple keys on multiple IPs; this migration is the
first step toward that, not the whole way there.

## Target

- A small **non-AWS** VPS (CLAUDE.md rules out AWS specifically). Hetzner
  CX22 (~€4/mo) or DigitalOcean ($6/mo) is far more than enough — the Pi already
  handles this. 1 vCPU / 1–2 GB RAM / Ubuntu 24.04 LTS.
- The collector is the only thing that moves. Training, the API, and the
  Streamlit app stay on the Mac. The VPS only accumulates
  `data/collector/observations.db`.

## Cutover principle: one canonical DB, no gap, no divergence

The DB on the collector host is canonical (the Mac only keeps backups). Two
collectors writing two DBs diverge — the same trap as the original Mac→Pi move.
So the cutover order matters: **stop the Pi collecting, snapshot its DB, seed the
VPS from that snapshot, then start the VPS.** Keep the Pi intact (just disabled)
until the VPS is proven, so rollback is trivial.

---

## 1. Provision and harden the VPS

On the VPS provider, create the Ubuntu 24.04 instance and add your SSH public
key. Then, from the Mac:

```bash
VPS=root@<vps-ip>            # use the IP the provider gave you
ssh "$VPS"
```

On the VPS, create a non-root user and install prerequisites:

```bash
adduser --disabled-password --gecos "" collector
usermod -aG sudo collector
install -d -m 700 -o collector -g collector /home/collector/.ssh
cp ~/.ssh/authorized_keys /home/collector/.ssh/ && chown collector:collector /home/collector/.ssh/authorized_keys

apt-get update
apt-get install -y python3-venv python3-pip git rsync cron util-linux
systemctl enable --now cron

# Minimal firewall: SSH in, everything out.
apt-get install -y ufw
ufw allow OpenSSH && ufw --force enable
```

From now on connect as `collector@<vps-ip>`. (Optional but recommended:
`apt-get install -y unattended-upgrades` for automatic security patches.)

## 2. Deploy the collector

The repo is public, so the VPS clones anonymously. `scripts/pi/setup.sh` is not
actually Pi-specific — it builds the collection-only venv and installs the dual
cron on any Debian/Ubuntu host.

```bash
ssh collector@<vps-ip>
git clone https://github.com/brandoncuii/cs2-skin-pricer.git
cd cs2-skin-pricer
echo 'CSFLOAT_API_KEY=<your key>' > .env      # same key; new IP
bash scripts/pi/setup.sh                       # venv + dual cron + flock guard
```

`setup.sh` installs exactly the schedule the Pi runs today (flock-guarded):

| Job   | Cron          | Scope                          |
|-------|---------------|--------------------------------|
| Locked | `17 */3 * * *` | 4 locked skins, 3 pages each  |
| Full   | `47 */12 * * *`| all knives, `--min-qty 1` gate |

Note: a default VPS runs cron in **UTC** (the Pi was Pacific), so the sweeps fire
at different wall-clock hours. The cadence is what matters, so leave it — or
`timedatectl set-timezone America/Los_Angeles` if you want it to match.

## 3. Seed the canonical DB (no history loss) and run the cutover

Do these in order, in one sitting, to avoid a collection gap or two divergent DBs:

```bash
# --- on the PI: stop collecting and take a crash-consistent snapshot ---
ssh pi@10.0.0.241 '
  crontab -l | grep -vF scripts/collect.py | crontab -    # disable Pi cron
  cd cs2-skin-pricer && python3 - <<"PY"
import sqlite3
s=sqlite3.connect("data/collector/observations.db")
d=sqlite3.connect("data/collector/observations-cutover.db")
with d: s.backup(d)
d.close(); s.close()
PY'

# --- copy the snapshot Pi -> Mac -> VPS (Mac bridges; it can reach both) ---
rsync -a pi@10.0.0.241:cs2-skin-pricer/data/collector/observations-cutover.db /tmp/
rsync -a /tmp/observations-cutover.db collector@<vps-ip>:cs2-skin-pricer/data/collector/observations.db

# --- on the VPS: smoke-test one collection run, then regenerate the gate ---
ssh collector@<vps-ip> 'cd cs2-skin-pricer && PYTHONPATH=. .venv/bin/python scripts/collect.py'
ssh collector@<vps-ip> 'cd cs2-skin-pricer && PYTHONPATH=. .venv/bin/python scripts/scan_full_names.py'
```

The smoke-test run should print `collector run start` and a non-error summary.
`scan_full_names.py` rebuilds `data/collector/full_names.json` (the liquidity
gate) on the new host; rerun it ~monthly as before.

## 4. Repoint the Mac's daily backup to the VPS

The backup currently hardcodes the Pi. Two edits in `scripts/backup_pull.sh`:

```bash
PI="collector@<vps-ip>"     # was pi@10.0.0.241
PI_REPO="cs2-skin-pricer"   # unchanged (clone path is the same)
```

Then set up passwordless SSH so the launchd job runs unattended, and verify:

```bash
ssh-copy-id collector@<vps-ip>
ssh -o BatchMode=yes collector@<vps-ip> true && echo "key auth OK"
bash scripts/backup_pull.sh        # should write data/collector/backups/observations-<today>.db.gz
```

The launchd job `com.cs2pricer.backup` runs this script daily at 10:00 — no plist
change needed once the script points at the VPS. Also update the one-off
"pull DB before training" command you use on the Mac:

```bash
rsync collector@<vps-ip>:cs2-skin-pricer/data/collector/observations.db \
      ~/coding-projects/cs2-skin-pricer/data/collector/
```

## 5. Verify, then decommission the Pi

After the first VPS sweep and a successful backup:

```bash
# VPS is collecting: row count should climb between two checks a few hours apart
ssh collector@<vps-ip> 'cd cs2-skin-pricer && PYTHONPATH=. .venv/bin/python -c "
import sqlite3; c=sqlite3.connect(\"data/collector/observations.db\")
print(\"observations:\", c.execute(\"SELECT COUNT(*) FROM observations\").fetchone()[0])"'

# VPS no longer shares your IP: browse csfloat.com while a sweep runs — no throttle.
ssh collector@<vps-ip> 'tail -2 cs2-skin-pricer/data/collector/collect_full.log'
```

The Pi's cron is already disabled (step 3). Leave the Pi as-is for a week as a
warm rollback target; once you trust the VPS, you can wipe the Pi or repurpose it.

## Rollback

If the VPS misbehaves, the Pi DB is untouched and only a `crontab` away:

```bash
ssh pi@10.0.0.241 'cd cs2-skin-pricer && bash scripts/pi/setup.sh'   # re-installs Pi cron
# revert scripts/backup_pull.sh PI="pi@10.0.0.241"
```

## Effort and cost

- **Cost:** ~$5–6/mo for the VPS. No other recurring cost; same single API key.
- **Setup:** ~30–45 min, mostly waiting on the snapshot copy and the first
  `scan_full_names.py` (~1 hour to run, but it runs unattended).
- **Ongoing:** identical to today — the Mac pulls a daily backup; you `rsync` the
  DB before training. Only the host changed.

## When you outgrow one VPS

This frees your home IP but keeps the single-key throughput ceiling. The next
step for genuine all-knife *sold*-price coverage is parallel collection: a second
API key (second CSFloat account) on a second VPS, each polling a disjoint slice
of the name list, writing to its own DB, merged on the Mac before training. That
is the commercial-grade tier — only worth it once one VPS is provably maxed out.
