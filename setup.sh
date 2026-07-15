#!/bin/bash
# Setup script for OpenClaw + Qwen Local on a new machine
# Run this after cloning/openclaw and setting up dependencies

set -e

echo "=== OpenClaw + Qwen Local Setup ==="

# Check if ~/.openclaw/workspace exists
if [ ! -d "$HOME/.openclaw/workspace" ]; then
    echo "Creating ~/.openclaw/workspace..."
    mkdir -p "$HOME/.openclaw/workspace"
fi

# Create workspace directory structure
mkdir -p "$HOME/.openclaw/workspace/memories"

echo "=== Copying OpenClaw settings ==="
cp -r /home/aldenpark/dev/qwen_local_openclaw/openclaw_settings/* "$HOME/.openclaw/workspace/"

echo "=== Copying memories ==="
cp -r /home/aldenpark/dev/qwen_local_openclaw/memories/* "$HOME/.openclaw/workspace/memories/"

echo "=== Copying Qwen local config ==="
cp /home/aldenpark/dev/qwen_local_openclaw/models/qwen-model-presets.json "$HOME/models/"
cp /home/aldenpark/dev/qwen_local_openclaw/models/qwen-local.sh "$HOME/models/"
cp /home/aldenpark/dev/qwen_local_openclaw/models/qwen-openclaw-optuna-tune.py "$HOME/models/"

echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "1. Start qwen-local: cd $HOME/models && ./qwen-local.sh fast"
echo "2. Start OpenClaw: openclaw gateway start"
echo "3. Verify model is running: nc -zv 127.0.0.1 18080"

install_packages() {
  if [ "$(uname -s)" = "Darwin" ]; then
    echo "== Installing macOS packages =="

    # Verify Xcode is installed
    if ! xcode-select -p >/dev/null 2>&1; then
      xcode-select --install
      echo "Complete the Xcode CLI Tools installation, then rerun this script."
      return 1
    fi

    # Handle standard Apple Silicon and Intel Homebrew locations.
    if ! command -v brew >/dev/null 2>&1; then
      if [ -x /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
      elif [ -x /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
      else
        echo "Homebrew is required: https://brew.sh"
        return 1
      fi
    fi

    brew install git cmake pkg-config python curl wget psmisc sqlite3 node
    return 0
  fi

  echo "== Installing Linux packages =="
  sudo apt update
  sudo apt install -y \
    git cmake build-essential curl wget psmisc \
    python3 python3-pip python3-venv python3-full \
    libcurl4-openssl-dev libsqlite3-dev pkg-config sqlite3
}
