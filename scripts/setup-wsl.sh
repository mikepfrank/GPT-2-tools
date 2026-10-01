#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$HOME/.venvs/gpt2-xl"

run_as_root() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "This setup needs root access to install python3-venv and python3-pip." >&2
    exit 1
  fi
}

if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
  echo "Installing Ubuntu's Python virtual-environment support..."
  run_as_root apt-get update
  run_as_root apt-get install -y python3-venv python3-pip
else
  echo "Ubuntu's Python virtual-environment support is already installed."
fi

echo "Creating virtual environment at $VENV_DIR..."
mkdir -p "$(dirname -- "$VENV_DIR")"
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "Installing CPU-only PyTorch..."
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.14.1+cpu'

echo "Installing the local GPT-2 XL runner..."
python -m pip install -r "$PROJECT_DIR/requirements.lock.txt"
python -m pip install --no-deps -e "$PROJECT_DIR"
python -m pip check

echo
echo "Runtime installation complete."
echo "Activate it with: source '$VENV_DIR/bin/activate'"
echo "Download the model with: gpt2-xl --download-only"
