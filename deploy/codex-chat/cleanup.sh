#!/bin/bash
set -euo pipefail
while read -r container; do
  [[ $container =~ ^[a-f0-9]{12,64}$ ]] || exit 64
  /usr/bin/docker rm -f "$container" >/dev/null
done < <(/usr/bin/docker ps -aq --filter label=io.gpt2-codex=sandbox)
