# Public GPT-2 XL on Amazon Linux

This deployment uses the original pinned GPT-2 XL checkpoint on CPU in FP32,
one resident model, and one generation at a time. Browser-held archives preserve
anonymous chats between page visits and can be exported/imported. This is a
modest beta service; CPU throughput, available RAM, queue capacity, and request
caps limit its capacity. The benchmark measures the same backend independently
of the web server.

## Current host and deployment gate

The initial read-only inspection on October 2, 2026 found Amazon Linux 2023.9,
x86_64, two logical CPUs reporting one core/two threads on an Intel Xeon Platinum
8175M at 2.5 GHz, approximately 7.6 GiB total RAM and 4.8 GiB available, no swap,
and about 100 GiB free disk. These observations are a snapshot, not a current
resource guarantee. The host also serves the existing Astra and OpenRouter chess
applications and their Caddy proxy; ports 80/443 and loopback 8788/8792 are already
in use. Preserve their users, services, files, environments, and listeners.

The approximately 6 GiB FP32 checkpoint plus runtime, activations, and up to
600 MiB of full-window KV tensors does **not** safely fit in the observed
4.8 GiB available RAM. The public server and benchmark default to requiring
8 GiB available physical RAM before loading. Resolve the memory plan first,
such as a larger or separate instance. Installing packages and downloading the
checkpoint do not establish that inference fits. Swap is not counted as
available RAM; do not reduce the guard merely to force startup on this shared
host. The unit's example memory limits also assume that the memory plan has
been resolved.

No Amazon Linux model-load or generation measurements have been reproduced yet.
The files here are deployment instructions and templates, not evidence that a
public service is installed or running.

## Install an isolated runtime

Use `ssh lightsail` to access the administrator account. Amazon Linux keeps
`/usr/bin/python3` at its system Python 3.9. Install a separate Python 3.12;
leave that symlink intact, as required by the
[Amazon Linux Python documentation](https://docs.aws.amazon.com/linux/al2023/ug/python.html).
The lock was resolved on Python 3.12; the observed Python 3.11 installation is
not a substitute for reproducing it.

```bash
# As ec2-user: install only this project's prerequisites, without a full update.
sudo dnf install -y python3.12 python3.12-pip git
sudo -iu gpt-2
```

Clone the repository as `gpt-2`, using its existing GitHub SSH configuration.
This work starts on local branch `codex/gpt2-public`, split from chat commit
`3657a81`; that branch has not yet been published. Once a reviewed deployment
commit has been published, select it explicitly. A transferred release checkout
is also possible when its source commit/files are recorded. The local chat
branch is not itself a public service. The unit assumes this checkout path:

```bash
git clone git@github.com:mikepfrank/GPT-2-tools.git /home/gpt-2/GPT-2-tools
cd /home/gpt-2/GPT-2-tools
# git switch <reviewed-deployment-branch>, then inspect git status and rev-parse HEAD.
bash scripts/setup-amazon-linux.sh --dry-run
bash scripts/setup-amazon-linux.sh --download-model
```

The setup refuses to run as root or `ec2-user`. It creates only this project's
`/home/gpt-2/.venvs/gpt2-public` and `/home/gpt-2/.cache/gpt2-public`, installs
`torch==2.14.1+cpu` and `requirements.lock.txt`, and installs this checkout
editable without changing the laptop's WSL environment. Binary-only package
installation avoids an accidental source build on the small host. If the
namespaced Python packages or pinned wheels are unavailable, stop and diagnose;
do not change the model, precision, dependency lock, or system Python silently.
System packages and pip itself are not pinned, so this is not a hermetic build.

The optional model download fetches only the needed config/tokenizer files and
safetensors checkpoint at revision
`15ea56dee5df4983c59b2538573817e1667135e2`. It verifies the full checkpoint
SHA-256 against
`0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a`.
Model weights, environments, and caches stay outside Git.

Use this cache for every later command:

```bash
export HF_HOME=/home/gpt-2/.cache/gpt2-public/huggingface
PYTHON=/home/gpt-2/.venvs/gpt2-public/bin/python
"$PYTHON" -m gpt2_local.cli --info
"$PYTHON" -m pip check
```

`--info` reports installed versions without loading the model. Retain that
output and the deployed Git commit alongside any benchmark results.

## Short throughput measurements

Begin with a plan that performs no inference, then resolve any failed memory
guard before proceeding. Stop the GPT-2 web service if it is already running;
benchmarking must not load a second copy of its model. Leave unrelated services
running and record their activity when interpreting performance.

```bash
"$PYTHON" scripts/benchmark-host.py --dry-run --threads 1
# Once the host has sufficient available physical RAM:
"$PYTHON" scripts/benchmark-host.py --threads 1
# Optional, separately named short comparison with the same resident model:
"$PYTHON" scripts/benchmark-host.py --threads 1 2 --samples 1 --max-new-tokens 32
# Optional context-cost comparison; this adds a substantial prompt prefill:
"$PYTHON" scripts/benchmark-host.py --threads 1 --samples 1 --near-full-context
```

The default is two warm samples of at most 32 output tokens, greedy decoding,
seed 42, and one CPU thread. Each requested thread count gets a one-token
warm-up recorded separately; only one model is loaded for the entire run.
Requests are capped at 64 tokens and four samples per prompt/thread count.
`--near-full-context` adds an exact tokenized prompt that fills the remaining
1,024-token context window. An EOS stop may yield fewer than the requested
output tokens. These short trials do not establish sustained performance after
burst capacity depletion or under concurrent traffic.

JSON evidence is checkpointed under ignored `outputs/benchmarks/`. It retains
the exact prompt/token IDs, output/token IDs, text hashes, settings, model and
runtime identity, UTC timing, warm-ups, model-load duration, memory snapshots,
and process CPU user/system time. The model digest is labeled **expected**;
the benchmark does not replace the download's actual file-integrity check.
Keep raw generated content private until it has been reviewed for publication.
Interrupted or failed runs retain completed measurements and their status.

Generation TPS includes prompt prefill and decoding. It measures neither
time-to-first-token nor isolated decode speed. Model construction wall time,
including initial runtime imports, is reported separately; the first warm-up
pays for lazy page-in and initial kernel setup.
`process_peak_rss_mib` is the Linux process-lifetime maximum resident set,
converted from KiB to MiB. Process CPU percentages report both one-logical-CPU
and all-host-logical-CPUs conventions: 100% of one logical CPU is 50% of a
two-vCPU host. These are process measurements, not the whole-instance CPU
metric shown in AWS. Other applications, steal time, throttling, and cgroup
limits can alter results.

The laptop's 14-thread measurements cannot be scaled linearly to this host's
two vCPUs; the CPU, memory bandwidth, SMT topology, and credit state differ.
Start with one thread and compare two empirically. A numeric Lightsail TPS
estimate remains unverified until the guarded benchmark can run.

For an intentionally isolated experiment, `--allow-low-memory` explicitly
bypasses the benchmark's pre-load check. It can trigger swapping or OOM kills,
including harm to unrelated services. It is not the suggested remedy for the
inspected shared host. The public service should retain its memory guard.

## Lightsail burst capacity

Lightsail describes sustainable and burstable CPU zones. Sustained use above
the baseline spends burst capacity; when that capacity is exhausted, performance
returns toward the sustainable zone. Monitor CPU utilization and remaining
burst capacity before/after trials in the dashboard. This is the behavior in
[AWS's Lightsail burst documentation](https://docs.aws.amazon.com/lightsail/latest/userguide/identifying-instance-burst.html).

AWS lists a 30% averaged baseline for the two-vCPU, 8 GB Lightsail bundle.
The inspected hardware is consistent with that size, but the account's actual
bundle and dashboard baseline still need confirmation. One fully occupied
logical CPU alone would average about 50% on two vCPUs before other workloads;
it can therefore spend credits even with one inference thread. See the
[Lightsail baseline table and averaging explanation](https://docs.aws.amazon.com/lightsail/latest/userguide/baseline-cpu-performance.html).

Do not infer EC2 Unlimited CPU-surplus charges for a Lightsail bundle. Those
charges are a separately documented behavior of
[EC2 Unlimited instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-performance-instances-unlimited-mode-concepts.html).
Confirm the actual product, plan, and billing in the AWS account before making
a cost estimate. The benchmark deliberately performs short trials rather than
a prolonged credit-depletion test.

## Test privately, then install the service

After the memory check and short model tests pass, run in the foreground as
`gpt-2`. The backend binds to loopback; inference remains offline after setup.

```bash
"$PYTHON" -m gpt2_local.public_chat \
  --host 127.0.0.1 --port 8899 --offline --threads 1 \
  --public-origin http://localhost:8899 \
  --max-reply-tokens 256 --minimum-available-memory-gib 8
```

From a local terminal, establish an SSH tunnel:

```powershell
ssh -N -L 8899:127.0.0.1:8899 lightsail
```

Open `http://localhost:8899/` and test generation, page reload, transcript
export/import, regeneration, a second browser, and queue/busy behavior. The
origin must match exactly; `http://127.0.0.1:8899` is a different origin.
Do not open an inbound firewall port for 8899. The SSH tunnel also avoids
transmitting trial transcripts in plain HTTP across the public network.

Before installing the unit, review its checkout path, cache path, origin,
thread count, request cap, and memory limits. The example uses low CPU weight
and niceness to give other work priority during contention; these are not
absolute CPU caps and do not avoid burst-capacity consumption. `MemoryHigh=8G`
and `MemoryMax=9G` assume a host with sufficient spare RAM. Use an explicit
systemd drop-in for a measured budget on the chosen host; the pre-load check,
process limit, and unrelated services' needs must agree. `MemorySwapMax=0`
does not enable system swap.

As `ec2-user`, after review:

```bash
sudo install -o root -g root -m 0600 /home/gpt-2/GPT-2-tools/deploy/gpt2-public.env.example /etc/gpt2-public.env
# Edit the exact external origin and settings before starting.
sudo install -o root -g root -m 0644 /home/gpt-2/GPT-2-tools/deploy/gpt2-public.service.example /etc/systemd/system/gpt2-public.service
sudo systemd-analyze verify /etc/systemd/system/gpt2-public.service
sudo systemctl daemon-reload
sudo systemctl start gpt2-public.service
sudo systemctl status gpt2-public.service --no-pager
sudo journalctl -u gpt2-public.service -n 50 --no-pager
# Match the configured origin's host; change this header with the origin.
curl --fail --header 'Host: localhost:8899' http://127.0.0.1:8899/api/health
```

Enable autostart only after the service and preservation checks pass:
`sudo systemctl enable gpt2-public.service`. A repeated startup failure is a
diagnostic condition, not a reason to bypass the memory guard. Stop only this
service to roll back: `sudo systemctl disable --now gpt2-public.service`.

The unit runs as `gpt-2`, reads the code/environment from their dedicated
paths, and permits writes only to its own cache and private temporary directory.
Logs should contain operational metadata, not conversation text. Browser
storage is local to that browser profile/origin; exporting structured transcripts
is the portability mechanism. Clearing browser storage or changing origin can
lose local recovery unless the user exports first.

## Integrate the existing proxy later

The active `astra-caddy` proxy already owns 80/443. Do not install a competing
listener or replace its Caddyfile. For a dedicated domain, configure DNS, set
`GPT2_PUBLIC_ORIGIN` to its exact HTTPS origin, and merge the site block in
`deploy/Caddyfile.gpt2.example` into the actual existing proxy configuration.
Keep the old configuration backed up and validate the complete candidate with
the installed Caddy executable and the active configuration format before a
reload. The inspected configuration is
`/home/astra/.config/astra-chess/Caddyfile` and its executable is
`/home/astra/.local/bin/caddy`; confirm these against the live unit before use.
The inspected `astra-caddy.service` has no `ExecReload`, so use Caddy's admin
API reload command after validation, in the `astra` account with the same
environment/admin endpoint as the live proxy. For example, once those details
are confirmed:

```bash
sudo -iu astra /home/astra/.local/bin/caddy validate --config /home/astra/.config/astra-chess/Caddyfile --adapter caddyfile
sudo -iu astra /home/astra/.local/bin/caddy reload --config /home/astra/.config/astra-chess/Caddyfile --adapter caddyfile
```

An explicitly configured admin endpoint may require `--address` on reload;
do not use an unverified default. See
[Caddy's reload command](https://caddyserver.com/docs/command-line#caddy-reload).
Do not overwrite any chess site blocks or restart the live proxy merely to
apply this new site.

For testing through the current instance address, choose an explicit unused
origin/site route on the existing proxy and set that origin in the backend.
The current UI/API expects the site root, so do not assume a `/gpt2` path prefix
works. Use the SSH tunnel for private tests. For an initial public beta before
procuring a domain, the example site block can instead begin with
`http://YOUR_PUBLIC_IP`, with `GPT2_PUBLIC_ORIGIN=http://YOUR_PUBLIC_IP`.
Confirm the existing proxy has no conflicting IP site before merging it.
This temporary HTTP origin sends chat content without transport encryption.
When moving to the dedicated HTTPS domain, export chats first and import them
at the new origin, because browser storage is scoped to each origin.

The reverse proxy must overwrite `X-Real-IP` with the actual connecting client's
address and retain the requested host. The example clears visitor-supplied
`Forwarded` and `X-Forwarded-For`; the backend does not use them for identity.
The installed unit trusts only the loopback proxy peer `127.0.0.1` for
`X-Real-IP`. Requests sent directly to the foreground test server do not trust
forwarded identity. Origin validation remains an exact allowlist; forwarded
headers cannot expand it. Caddy's `header_up` directives overwrite/delete
upstream headers as described in its
[reverse-proxy documentation](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#headers).

After starting or reloading, verify the GPT-2 page/API through the intended
origin, then check that `astra-chess`, `or-chess`, and `astra-caddy` are still
active, their existing listeners/health checks succeed, and their existing site
blocks and application configurations remain unchanged. Record the deployed Git commit, runtime versions, chosen
memory budget, benchmark results, and measured CPU/burst-capacity observations
in `VERIFICATION.md` only after reproducing them.
