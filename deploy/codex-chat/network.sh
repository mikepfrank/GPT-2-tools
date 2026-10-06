#!/bin/bash
# Idempotent setup for this beta's own network and input chain only.
set -euo pipefail
if ! docker network inspect gpt2-codex-isolated >/dev/null 2>&1; then
  docker network create --internal --subnet 172.30.97.0/24 --gateway 172.30.97.1 \
    --opt com.docker.network.bridge.enable_icc=false gpt2-codex-isolated
fi
docker network inspect gpt2-codex-isolated | python3 -c '
import json, sys
n = json.load(sys.stdin)[0]
assert n["Internal"] is True, "Codex network is not internal"
assert n["Options"].get("com.docker.network.bridge.enable_icc") == "false", "Container isolation is disabled"
assert n["IPAM"]["Config"] == [{"Subnet":"172.30.97.0/24", "Gateway":"172.30.97.1"}], "Unexpected Codex subnet"
'
# No access to SSH, metadata or another host service through the bridge gateway.
iptables -N GPT2-CODEX-INPUT 2>/dev/null || true
iptables -C GPT2-CODEX-INPUT -p tcp --dport 8902 -j ACCEPT 2>/dev/null || \
  iptables -A GPT2-CODEX-INPUT -p tcp --dport 8902 -j ACCEPT
iptables -C GPT2-CODEX-INPUT -j DROP 2>/dev/null || iptables -A GPT2-CODEX-INPUT -j DROP
iptables -C INPUT -s 172.30.97.0/24 -j GPT2-CODEX-INPUT 2>/dev/null || \
  iptables -I INPUT 1 -s 172.30.97.0/24 -j GPT2-CODEX-INPUT
