# Production streaming operations

Status: **owner accepted the hosted preview and authorized production rollout;
production streaming active, October 3 CDT / October 4 UTC, 2026**. Preparation passed
on Amazon Linux at `2026-10-04 04:47:57.130018 UTC`, after 14 promotion tests and
11 exclusivity-guard tests passed. The durable promotion unit began its full
600-second preview-result retention wait at **04:48:13.198694 UTC** and emitted
`production_streaming_active` at **2026-10-04 04:58:27.430940 UTC**. Production
PID **58714** is active/enabled; preview is inactive with PID 0. An independent
check at **04:59:00.504526 UTC** passed canonical HTTPS streaming-ready health,
the test-domain 307/`no-store` redirect, exact source/override/snapshot
preservation, sole-model ownership, and unchanged chess/Caddy PIDs. Two
certified canonical HTTPS generation jobs then passed incremental SSE,
disconnect/reconnect, canonical token/settings/archive/import-export and
regeneration checks, plus exact-origin/query/redirect checks. Independent
after-verification at **05:01:30.491948 UTC** passed with no warnings and the
same production/chess/Caddy PIDs. Exact reproduced evidence belongs in
[`VERIFICATION.md`](../VERIFICATION.md).

The active frozen application is **`5389a30`** at
`/home/gpt-2/GPT-2-streaming-preview`. The original production checkout
`/home/gpt-2/GPT-2-tools` stays at **`e5971db`**, and its installed base unit and
`/etc/gpt2-public.env` remain unchanged. Promotion installed only the owned
`/etc/systemd/system/gpt2-public.service.d/90-gpt2-streaming.conf`. It selects the
tested checkout through `WorkingDirectory` and an explicit
`/usr/bin/env PYTHONPATH=/home/gpt-2/GPT-2-streaming-preview/src` launch wrapper,
then runs the same pinned Python/model with `--enable-streaming` on loopback
8899 and the existing exact `https://gpt2-chat.com` origin. CPU FP32, two Torch
threads, the 256-token cap, 8 GiB startup floor, and 8.5/9 GiB limits remain.
The installed startup guard is
`/usr/local/libexec/gpt2-exclusive-preflight.py`; it refuses a running,
transitional, or nonempty opposing preview service/cgroup.

## Prepared paths and completed promotion

The root-owned tooling is separate from both application checkouts:

| Item | Path |
| --- | --- |
| Helper | `/home/ec2-user/gpt2-promotion-tooling/scripts/lightsail-promote.py` |
| New integrity/backups stage | `/home/astra/.config/astra-chess/gpt2-production-stage-20261004` |
| Original hosted-preview stage | `/home/astra/.config/astra-chess/gpt2-preview-stage-20261004` |

Preparation stages immutable complete Caddy candidates, the production drop-in,
unit-validation candidate, and preserved-file snapshots, with a private
manifest. It changes no live configuration or services. This command already
completed; `--prepare` requires a **fresh** stage directory, so do not repeat it
against this existing stage:

```bash
sudo /usr/bin/python3.12 /home/ec2-user/gpt2-promotion-tooling/scripts/lightsail-promote.py \
  --prepare --stage /home/astra/.config/astra-chess/gpt2-production-stage-20261004 \
  --preview-stage /home/astra/.config/astra-chess/gpt2-preview-stage-20261004
```

The following launch is historical and **completed successfully**; do not repeat:

```bash
sudo systemd-run --unit=gpt2-production-promote-20261004-0448 --collect \
  --property=Type=exec /usr/bin/python3.12 \
  /home/ec2-user/gpt2-promotion-tooling/scripts/lightsail-promote.py \
  --promote --stage /home/astra/.config/astra-chess/gpt2-production-stage-20261004
sudo journalctl -u gpt2-production-promote-20261004-0448 --no-pager
```

Ctrl+C exits only the log viewer. The helper gates preview generation POSTs
with 503/`accepted: false`, `no-store`, and `Retry-After: 60`, retains GETs while
accepted work drains, then waits the entire 600-second result TTL before
stopping preview. Only then does it install the override, verify loaded
commands/settings, recheck memory and sole-model guards, and start production.
Direct and canonical TLS health must advertise streaming before admission opens.
Production was kept in maintenance until ready; admission is now open.
`test.gpt2-chat.com` redirects to the canonical site with **307 and `no-store`**,
preserving path/query so the test origin remains reusable. JSON export/import
crosses the browser-storage boundary. Original public/chess Caddy blocks,
including existing 308 aliases, remain byte-for-byte intact.

The terminal journal event, canonical HTTPS health, redirect, sole-model
ownership, unchanged chess/configuration checks, and real canonical HTTPS
generation/SSE/archive verification passed. Keep their reproduced timings and
resource measurements in `VERIFICATION.md`.
The helper uses the shared Caddy lock, full validation/backups, inode-preserving
fsync writes and fresh-cursor SIGUSR1 confirmation. Never force-stop
undrainable accepted jobs. Promotion failure attempts safe recovery of the
accepted preview; uncertain/drifting state remains gated for operator review.

## Roll back to the original polling production build

**The old preview helper's `--restore` is obsolete after promotion.** Use the
promotion helper's `--rollback` with the same production stage and a fresh
durable unit; an SSH disconnect must not interrupt the retention window:

```bash
rollback_unit="gpt2-production-rollback-$(date -u +%Y%m%d-%H%M%S)"
sudo systemd-run --unit="$rollback_unit" --collect --property=Type=exec \
  /usr/bin/python3.12 /home/ec2-user/gpt2-promotion-tooling/scripts/lightsail-promote.py \
  --rollback --stage /home/astra/.config/astra-chess/gpt2-production-stage-20261004
sudo journalctl -u "$rollback_unit" -f
```

Rollback gates production generation, drains and retains results for 600
seconds, then stops production, including any scheduled on-failure retry. It
removes **only the byte-matching owned drop-in**, daemon-reloads, and starts the
original `e5971db` build under maintenance. After readiness, it restores the
original canonical Caddy bytes plus the temporary test-domain redirect. The
base unit, env, old checkout, and guard files remain intact. Require
`original_polling_production_restored`, canonical TLS health, redirect behavior,
and unchanged chess/one-model invariants. Keep the stage and frozen tooling;
configuration/source drift or a changed override requires review, not a blind
stop, overwrite, or rerun.
