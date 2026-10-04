# Production seed policy update

Status: **production active**, October 4, 2026. The durable unit
`gpt2-seed-update-20261004-1925` retained results for 600 seconds and emitted
`production_version_updated` at **19:35:22.787953 UTC / 2:35 p.m. CDT**.
Production PID **91398** uses the accepted seed-policy release; preview is
inactive. Exact independent and live HTTPS evidence is in `VERIFICATION.md`.

The owner accepted the streaming laptop preview on October 4, 2026 and
authorized deploying its automatic seed advancement policy. The application
release is pinned to `78ebbdc254249562187c06db8ec9c3cb7dd0e87b` in
`/home/gpt-2/GPT-2-releases/seed-78ebbdc`, separately from both previous builds.
The complete release directory is root-owned and read-only. The service still
runs as `gpt-2`; its existing runtime and model cache stay outside that directory.

Both a new message and regeneration use the selected or stored next seed, then
save its successor modulo `2**63` after successful generation and validation.
Historical generation records keep the seed actually used. Failed requests,
previews, provisional stream updates, reconnects and repeated result retrieval
do not advance the seed. Imports preserve the saved seed as the next seed.
This changes neither the model nor its sampling algorithm.

The update uses the existing production service on port 8899 and replaces only
its owned `90-gpt2-streaming.conf` override. The base unit, environment,
singleton startup guard, pinned CPU FP32 GPT-2 XL, two inference threads,
256-token reply cap and memory limits stay unchanged. The preview service stays
inactive. The original streaming checkout remains at `5389a30`; the older
polling checkout remains at `e5971db`.

The update helper lives separately in
`/home/ec2-user/gpt2-update-tooling/scripts/lightsail-update.py`. Its stage is
`/home/astra/.config/astra-chess/gpt2-seed-update-stage-20261004`. Preparation
validates and freezes snapshots and complete Caddy candidates without changing
the running service. Activation gates new generation POSTs, drains accepted
work, and retains completed results for their entire 600-second lifetime before
stopping the old process. It starts the new release behind the gate and requires
model identity, streaming, direct health and certified HTTPS readiness before
reopening admission. Only one model process runs at a time.

The final Caddy bytes match those before the update. Temporary admission gates
use the shared Caddy maintenance lock, preserve its bind-mounted inode and
reload with SIGUSR1. Both chess services and the Caddy process remain running.
The `test.gpt2-chat.com` origin keeps its temporary 307/no-store redirect to
the canonical service, and canonical requests continue to ignore query strings.

## Rollback

Use the update helper's `--rollback` with this update stage to restore the
previous streaming version and its exact saved override. Run it in a fresh
durable systemd unit. It gates, drains and retains results before replacing the
owned override, and checks readiness before reopening the old release:

```bash
sudo systemd-run --unit=gpt2-seed-rollback-UNIQUE --collect \
  --property=Type=exec /usr/bin/python3.12 \
  /home/ec2-user/gpt2-update-tooling/scripts/lightsail-update.py \
  --rollback --stage /home/astra/.config/astra-chess/gpt2-seed-update-stage-20261004
```

Replace `UNIQUE` with a fresh operation label. Inspect that unit's journal and
independently verify the terminal event, exact release/override, sole-model
ownership, certified canonical HTTPS and unchanged chess services. Do not
force-stop undrainable work or bypass a drift/protection check. Failure recovery
uses the same drain/retention sequence; uncertain state stays gated for review.

For a further rollback to original polling behavior, first complete this
streaming rollback. Only then use the original promotion helper's `--rollback`
with `/home/astra/.config/astra-chess/gpt2-production-stage-20261004`, as described
in [PRODUCTION-STREAMING.md](PRODUCTION-STREAMING.md). The original helper's
integrity checks require the previous streaming override and source. Neither
the old preview helper's `--restore` nor directly checking out a different
branch is a supported production switch.

Deployment events and independently verified results are recorded in
`VERIFICATION.md`. Raw prompts, bearer job IDs, archives and evidence remain
under ignored `outputs/` and are never published.
