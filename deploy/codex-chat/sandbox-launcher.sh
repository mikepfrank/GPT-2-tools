#!/bin/bash
# Root-owned constrained sudo target. Never grant the service Docker access.
set -euo pipefail
[[ $# -ge 4 ]] || exit 64
session=$1
codex_home=$2
workspace=$3
base_url=$4
shift 4
[[ $session =~ ^[a-f0-9]{32}$ ]] || exit 64
root="/home/gpt-2/.local/share/gpt2-codex/agents/players/$session"
[[ $codex_home == "$root/codex-home" && $workspace == "$root/workspace" ]] || exit 64
[[ $base_url == http://172.30.97.1:8902/v1 ]] || exit 64
[[ ${GPT2_PROXY_TOKEN:-} =~ ^[A-Za-z0-9_-]{30,150}$ ]] || exit 64
[[ $# -eq 3 && $1 == app-server && $2 == --stdio && $3 == --strict-config ]] || exit 64
[[ $(realpath -e "$codex_home") == "$codex_home" && $(realpath -e "$workspace") == "$workspace" ]] || exit 64
[[ $(stat -c %u "$codex_home") == 1015 && $(stat -c %u "$workspace") == 1015 ]] || exit 64
name="gpt2-codex-$session"
cleanup() { /usr/bin/docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
cleanup
/usr/bin/timeout --signal=TERM --kill-after=5s 190s /usr/bin/docker run --rm --interactive --stop-timeout 5 --name "$name" \
  --label io.gpt2-codex=sandbox \
  --network gpt2-codex-isolated --user 1015:1018 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  --security-opt seccomp=/usr/local/libexec/gpt2-codex-seccomp.json \
  --memory 768m --memory-swap 768m --cpus 1 --pids-limit 96 \
  --tmpfs /tmp:rw,nosuid,nodev,size=128m \
  --mount "type=bind,source=$codex_home,target=$codex_home" \
  --mount "type=bind,source=$workspace,target=$workspace" \
  --env "CODEX_HOME=$codex_home" --env "HOME=$workspace" \
  --env "TMPDIR=/tmp" --env GPT2_PROXY_TOKEN --env NO_COLOR=1 \
  --workdir "$workspace" gpt2-codex-runtime:0.154.0 "$@" <&0 &
child=$!
wait "$child"
