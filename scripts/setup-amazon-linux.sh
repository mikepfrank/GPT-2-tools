#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DOWNLOAD_MODEL=false
DRY_RUN=false
for argument in "$@"; do
  case "$argument" in
    --download-model) DOWNLOAD_MODEL=true ;;
    --dry-run) DRY_RUN=true ;;
    --help|-h)
      echo "Usage: bash scripts/setup-amazon-linux.sh [--download-model] [--dry-run]"
      echo "Run as gpt-2 on Amazon Linux 2023; see docs/DEPLOYMENT.md for prerequisites."
      exit 0 ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done

if [[ "$(id -un)" != "gpt-2" || "$EUID" -eq 0 ]]; then
  echo "Run this script in the gpt-2 login account, not as root or ec2-user." >&2
  exit 1
fi
ACCOUNT_HOME="$(getent passwd gpt-2 | cut -d: -f6)"
if [[ -z "$ACCOUNT_HOME" || "$HOME" != "$ACCOUNT_HOME" ]]; then
  echo "HOME must be the gpt-2 account's own home. Use sudo -iu gpt-2." >&2
  exit 1
fi
if [[ ! -r /etc/os-release ]]; then
  echo "Cannot identify this Linux distribution." >&2
  exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "$ID" != "amzn" || "$VERSION_ID" != "2023" || "$(uname -m)" != "x86_64" ]]; then
  echo "This script targets Amazon Linux 2023 x86_64 and the pinned CPU baseline." >&2
  exit 1
fi
if ! command -v python3.12 >/dev/null 2>&1 || ! python3.12 -c 'import ensurepip' >/dev/null 2>&1; then
  echo "Python 3.12 with venv/ensurepip support is required." >&2
  echo "From ec2-user, install the namespaced prerequisites: sudo dnf install -y python3.12 python3.12-pip" >&2
  echo "Leave /usr/bin/python3 and other services' environments unchanged." >&2
  exit 1
fi

VENV_DIR="$ACCOUNT_HOME/.venvs/gpt2-public"
export HF_HOME="$ACCOUNT_HOME/.cache/gpt2-public/huggingface"
export PIP_CACHE_DIR="$ACCOUNT_HOME/.cache/gpt2-public/pip"
if [[ "$DRY_RUN" == true ]]; then
  echo "Prerequisites OK. Would create/reuse $VENV_DIR and install the pinned CPU runtime."
  echo "Hugging Face cache: $HF_HOME"
  echo "Download and SHA-256 verification requested: $DOWNLOAD_MODEL"
  exit 0
fi

mkdir -p "$ACCOUNT_HOME/.venvs" "$HF_HOME" "$PIP_CACHE_DIR"
python3.12 -m venv "$VENV_DIR"
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
"$PYTHON" -m pip install --only-binary=:all: -r "$PROJECT_DIR/requirements.lock.txt"
"$PYTHON" -m pip install --no-build-isolation --no-deps -e "$PROJECT_DIR"
"$PYTHON" -m pip check

if [[ "$DOWNLOAD_MODEL" == true ]]; then
  "$PYTHON" -m gpt2_local.cli --download-only
fi
echo "Installed the isolated CPU FP32 runtime as $(id -un)."
echo "Set HF_HOME=$HF_HOME when using this runtime."
echo "Model loading is separate; check available memory before benchmarking or starting the service."
