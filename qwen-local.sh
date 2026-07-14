#!/usr/bin/env bash
set -euo pipefail
umask 077

# --- Runtime tuning notes ----------------------------------------------------
#
# These knobs control memory use, speed, and how much context llama-server can
# handle. For OpenClaw routing, prefer stable/fast settings over huge context.
#
# CTX:
#   Usable context window.
#   Raise this last. Larger context increases KV-cache memory usage.
#   Router models usually only need 4096–8192.
#   Planning/review models benefit from 16384–32768.
#
# MAX_TOKENS:
#   Maximum generated tokens per response.
#   Use 1024–2048 for router/classifier output.
#   Use 4096 for planning, review, or longer explanations.
#
# NGL:
#   Number of attention layers to offload to GPU/Metal.
#   Use 999 when the model fits; llama.cpp will offload as much as possible.
#   On NVIDIA, reduce this only when VRAM OOMs.
#   On Apple Silicon/Metal, usually leave this at 999 because memory is unified.
#   Partial GPU offload (only if GPU is saturated)
#     NGL="16"   # All attention layers for 4B model
#     NGL="8"    # Half attention layers on GPU, half on CPU
#
# PARALLEL:
#   Number of concurrent request slots.
#   Keep this at 1 for OpenClaw unless serving multiple clients.
#   Higher values increase memory use.
#
# BATCH:
#   Prompt processing batch size.
#   Higher can improve prompt ingestion speed but uses more memory.
#   Reduce this after UBATCH if startup or prompt processing OOMs.
#   Good fallback values: 256, 128, 64.
#
# UBATCH:
#   Physical micro-batch size.
#   Lower this first when you hit OOM or instability.
#   Good fallback values: 256, 128, 64.
#
# FLASH_ATTN:
#   Attention memory/performance optimization.
#   Use auto by default. Try on if supported and stable.
#   Turn off only if llama.cpp reports problems or output becomes unstable.
#
# Memory rule of thumb:
#   Total available memory should exceed the quantized GGUF file size plus
#   KV-cache/runtime overhead. If it does not, llama.cpp may still run by
#   offloading between CPU/GPU, but it will be slower.
#
# NVIDIA vs Apple Silicon:
#   NVIDIA laptops are limited mostly by dedicated VRAM. An RTX 4070 laptop
#   with 8 GB VRAM may need lower CTX/BATCH/UBATCH or reduced NGL.
#   Apple Silicon uses unified memory, so a MacBook Pro with 24 GB or 32 GB
#   unified memory can often run larger GGUFs than an 8 GB NVIDIA GPU, even
#   if raw GPU speed is lower.
#
# Tuning order when something fails:
#   1. Lower UBATCH.
#   2. Lower BATCH.
#   3. Lower CTX.
#   4. On NVIDIA only, lower NGL.
#   5. Lower MAX_TOKENS if long generations are causing memory pressure.
#
# /home/aldenpark/
# ├── .local/bin/openclaw-start # Main launcher
# ├── .local/bin/openclaw-stop
# ├── .local/bin/openclaw-status
# ├── .convig/qwen-local/openclaw.env   # settings for OpenClaw
# ├── models/                                                                                                                                       
# │   ├── qwen-local.sh          # Main qwen sh program 
# │   ├── qwen3.5-4b-q5km/   # Model directory                                                                                                      
# │   │   └── Qwen3.5-4B-Q5_K_M.gguf                                                                                                                
# │   └── qwen-model-presets.json  # Config catalog                                                                                                 
# └── .openclaw/                                                                                                                                    
#     └── workspace/                                                                                                                                    
#         └── openclaw.json   # OpenClaw integration config  
#
# https://apxml.com/tools/vram-calculator
# https://huggingface.co/unsloth/models?sort=alphabetical&p=30 qwen does not offer gguf quantized varients
# ~/.local/share/qwen-local/hf-venv/bin/pip install -U huggingface_hub
#

# --- Model presets -----------------------------------------------------------

MODE="${1:-help}"
CATALOG="${QWEN_MODEL_CATALOG:-$HOME/models/qwen-model-presets.json}"

catalog_python() {
  python3 - "$@"
}

require_catalog() {
  [ -f "$CATALOG" ] || {
    echo "ERROR: model catalog not found: $CATALOG"
    echo "Install/copy qwen-model-presets.json to: $HOME/models/qwen-model-presets.json"
    exit 1
  }
}

load_known_model_presets() {
  require_catalog
  local preset
  local preset_output
  preset_output="$(catalog_python "$CATALOG" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    data = json.load(f)
for name in data.get('models', {}):
    print(name)
PY
)"

  KNOWN_MODEL_PRESETS=()
  if type mapfile >/dev/null 2>&1; then
    mapfile -t KNOWN_MODEL_PRESETS <<< "$preset_output"
  else
    while IFS= read -r preset; do
      [ -n "$preset" ] && KNOWN_MODEL_PRESETS+=("$preset")
    done <<< "$preset_output"
  fi
}

load_known_model_presets

host_memory_gb() {
  if [ "$(uname -s)" = "Darwin" ]; then
    local bytes
    bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    echo $((bytes / 1024 / 1024 / 1024))
    return 0
  fi

  awk '/MemTotal:/ {print int($2 / 1024 / 1024)}' /proc/meminfo 2>/dev/null || echo 0
}

nvidia_vram_gb() {
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo 0
    return 0
  }

  local mib
  mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -n 1 | tr -d ' ' || true)"
  [[ "$mib" =~ ^[0-9]+$ ]] || {
    echo 0
    return 0
  }

  echo $((mib / 1024))
}

detect_model_hardware() {
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    echo "metal $(host_memory_gb)"
    return 0
  fi

  local vram_gb
  vram_gb="$(nvidia_vram_gb)"
  if [ "$vram_gb" -gt 0 ]; then
    echo "cuda $vram_gb"
    return 0
  fi

  echo "cpu $(host_memory_gb)"
}

model_recommendation() {
  local preset="$1"
  local backend="$2"
  local memory_gb="$3"

  case "$preset" in
    qwen35-4b-*)
      echo "recommended: fast router/planner baseline"
      ;;
    qwen35-9b-q4km|qwen35-9b-iq4nl|qwen35-9b-ud-q4kxl)
      if [ "$backend" = "cuda" ] && [ "$memory_gb" -lt 12 ]; then
        echo "recommended with partial offload: tune NGL 16-32"
      elif [ "$backend" = "cpu" ] && [ "$memory_gb" -lt 24 ]; then
        echo "maybe: CPU will be slow"
      else
        echo "recommended"
      fi
      ;;
    qwen35-9b-q5km)
      if [ "$backend" = "cuda" ] && [ "$memory_gb" -lt 12 ]; then
        echo "maybe: quality test; expect lower NGL/CPU offload"
      elif [ "$backend" = "cpu" ] && [ "$memory_gb" -lt 24 ]; then
        echo "maybe: CPU will be slow"
      else
        echo "recommended"
      fi
      ;;
    qwen35-2b-*|qwen35-0.8b-*)
      echo "fast but weaker; use only if 4B/9B are too slow"
      ;;
    qwen3-4b-*)
      echo "legacy fallback"
      ;;
    qwen3-8b-*)
      echo "legacy fallback; consider qwen35-9b-* instead"
      ;;
    qwen36-35b-a3b)
      if [ "$backend" = "cuda" ] && [ "$memory_gb" -ge 24 ]; then
        echo "maybe: high-end GPU/server model"
      elif [ "$backend" = "metal" ] && [ "$memory_gb" -ge 48 ]; then
        echo "maybe: high-memory Apple Silicon only"
      else
        echo "not recommended: high-end server model"
      fi
      ;;
    *)
      echo "unknown"
      ;;
  esac
}


default_model_choice() {
  local backend="$1"
  local memory_gb="$2"

  case "$backend" in
    cuda)
      # Default to the 9B Q4_K_M Unsloth GGUF because this setup uses
      # Qwen as an orchestrator/planner and can use lower NGL for RAM offload.
      if [ "$memory_gb" -ge 8 ]; then
        echo 3
      else
        echo 1
      fi
      ;;
    metal)
      if [ "$memory_gb" -ge 24 ]; then
        echo 4
      elif [ "$memory_gb" -ge 16 ]; then
        echo 3
      else
        echo 1
      fi
      ;;
    *)
      echo 1
      ;;
  esac
}


default_tune_profile() {
  local backend="$1"
  local memory_gb="$2"

  case "$backend" in
    cuda)
      if [ "$memory_gb" -ge 24 ]; then
        echo "4090-24gb"
      else
        echo "4070-8gb"
      fi
      ;;
    metal)
      if [ "$memory_gb" -ge 32 ]; then
        echo "macbook-pro-large"
      elif [ "$memory_gb" -ge 24 ]; then
        echo "macbook-pro"
      else
        echo "macbook-pro-small"
      fi
      ;;
    *)
      echo "auto"
      ;;
  esac
}

print_model_presets() {
  local backend="${1:-}"
  local memory_gb="${2:-0}"
  local default_choice="${3:-0}"
  local i=1

  for preset in "${KNOWN_MODEL_PRESETS[@]}"; do
    local marker="  "
    if [ "$i" -eq "$default_choice" ]; then
      marker="=>"
    fi

    if [ -n "$backend" ]; then
      printf "%s %d. %-22s [%s]\n" "$marker" "$i" "$preset" "$(model_recommendation "$preset" "$backend" "$memory_gb")"
    else
      printf "%s %d. %s\n" "$marker" "$i" "$preset"
    fi
    i=$((i + 1))
  done
}

model_override_set() {
  [ -n "${MODEL_PRESET+x}" ] ||
    [ -n "${MODEL_REPO+x}" ] ||
    [ -n "${MODEL_FILE+x}" ] ||
    [ -n "${MODEL_DIR+x}" ] ||
    [ -n "${MODEL+x}" ]
}

choose_model_preset() {
  case "$MODE" in
    install|update) ;;
    *) return 0 ;;
  esac

  model_override_set && return 0

  if [ ! -t 0 ]; then
    local backend
    local memory_gb
    local default_choice
    read -r backend memory_gb <<< "$(detect_model_hardware)"
    default_choice="$(default_model_choice "$backend" "$memory_gb")"
    MODEL_PRESET="${KNOWN_MODEL_PRESETS[$((default_choice - 1))]}"
    echo "No interactive input available; using MODEL_PRESET=$MODEL_PRESET"
    return 0
  fi

  if [ "$MODE" = "update" ]; then
    echo "Choose model to update:"
  else
    echo "Choose model to install:"
  fi

  local backend
  local memory_gb
  local default_choice
  read -r backend memory_gb <<< "$(detect_model_hardware)"
  default_choice="$(default_model_choice "$backend" "$memory_gb")"

  case "$backend" in
    cuda)
      echo "Detected hardware: NVIDIA CUDA (${memory_gb}GB VRAM)"
      ;;
    metal)
      echo "Detected hardware: Apple Silicon/Metal (${memory_gb}GB unified memory)"
      ;;
    *)
      echo "Detected hardware: CPU-only (${memory_gb}GB system memory)"
      ;;
  esac
  echo
  print_model_presets "$backend" "$memory_gb" "$default_choice"
  echo
  printf "Model [%s]: " "$default_choice"

  local choice
  read -r choice
  choice="${choice:-$default_choice}"

  if [[ "$choice" =~ ^[0-9]+$ ]] &&
    [ "$choice" -ge 1 ] &&
    [ "$choice" -le "${#KNOWN_MODEL_PRESETS[@]}" ]; then
    MODEL_PRESET="${KNOWN_MODEL_PRESETS[$((choice - 1))]}"
  else
    MODEL_PRESET="$choice"
  fi
}

choose_model_preset
MODEL_PRESET="${MODEL_PRESET:-qwen35-4b-q5km}"

set_model_defaults_for_preset() {
  local preset="$1"
  require_catalog

  local assignments
  assignments="$(catalog_python "$CATALOG" "$preset" <<'PY'
import json, os, shlex, sys
catalog_path, preset = sys.argv[1], sys.argv[2]
with open(catalog_path, encoding='utf-8') as f:
    data = json.load(f)
models = data.get('models', {})
if preset not in models:
    print(f"__QWEN_CATALOG_ERROR__={shlex.quote('Unknown MODEL_PRESET: ' + preset)}")
    print("__QWEN_CATALOG_KNOWN__=" + shlex.quote("\n".join(models.keys())))
    raise SystemExit(0)
model = models[preset]
defaults = model.get('defaults', {})
def expand(v):
    if not isinstance(v, str):
        return v
    return os.path.expandvars(v.replace("$HOME", os.path.expanduser("~")))
def emit(k, v):
    print(f"{k}={shlex.quote(str(v))}")
emit('DEFAULT_MODEL_REPO', model['repo'])
emit('DEFAULT_MODEL_FILE', model['file'])
emit('DEFAULT_MODEL_DIR', expand(model['dir']))
emit('DEFAULT_CTX', defaults.get('ctx', 32768))
emit('DEFAULT_NGL', defaults.get('ngl', 999))
emit('DEFAULT_BATCH', defaults.get('batch', 256))
emit('DEFAULT_UBATCH', defaults.get('ubatch', 128))
emit('DEFAULT_MAX_TOKENS', defaults.get('predict', 4096))
emit('DEFAULT_PARALLEL', defaults.get('parallel', 1))
emit('DEFAULT_FLASH_ATTN', defaults.get('flash_attn', 'auto'))
emit('DEFAULT_REASONING', defaults.get('reasoning', 'off'))
PY
)"

  if grep -q '^__QWEN_CATALOG_ERROR__=' <<< "$assignments"; then
    eval "$assignments"
    echo "ERROR: $__QWEN_CATALOG_ERROR__"
    echo
    echo "Known presets:"
    while IFS= read -r p; do
      echo "  $p"
    done <<< "$__QWEN_CATALOG_KNOWN__"
    exit 1
  fi

  eval "$assignments"
}


set_model_defaults_for_preset "$MODEL_PRESET"

# Model download location. Override these when testing a different GGUF.
MODEL_REPO="${MODEL_REPO:-$DEFAULT_MODEL_REPO}"
MODEL_FILE="${MODEL_FILE:-$DEFAULT_MODEL_FILE}"
MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"
MODEL="${MODEL:-$MODEL_DIR/$MODEL_FILE}"

# Optional integrity check. Leave empty to skip checksum verification.
MODEL_SHA256="${MODEL_SHA256:-}"

# llama.cpp source/build location and installed binary paths.
LLAMA_DIR="${LLAMA_DIR:-$HOME/src/llama.cpp}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-}"
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-}"
SERVER="${SERVER:-$HOME/.local/bin/llama-server}"
CLI="${CLI:-$HOME/.local/bin/llama-cli}"

# Runtime server binding and model-performance knobs.
# HOST currently binds to LAN. Use 127.0.0.1 if you only want local access.
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
CTX="${CTX:-${DEFAULT_CTX:-32768}}"

# NGL=999 asks llama.cpp to offload as many layers as possible to GPU/Metal.
NGL="${NGL:-${DEFAULT_NGL:-999}}"
REASONING_BUDGET="${REASONING_BUDGET:-512}"

# -n, -np, -b, and -ub flags passed to llama-server.
MAX_TOKENS="${MAX_TOKENS:-${DEFAULT_MAX_TOKENS:-4096}}"
PARALLEL="${PARALLEL:-${DEFAULT_PARALLEL:-1}}"
BATCH="${BATCH:-${DEFAULT_BATCH:-256}}"
UBATCH="${UBATCH:-${DEFAULT_UBATCH:-128}}"
FLASH_ATTN="${FLASH_ATTN:-${DEFAULT_FLASH_ATTN:-auto}}"
REASONING="${REASONING:-${DEFAULT_REASONING:-off}}"

# Daemon bookkeeping. Keep the stable PID path outside world-writable /tmp;
# write_pid_file uses mktemp and an atomic rename when the daemon starts.
PID_FILE_IS_DEFAULT=0
if [ -z "${PID_FILE:-}" ]; then
  PID_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/qwen-local/qwen-local-${PORT}.pid"
  PID_FILE_IS_DEFAULT=1
fi
LOG_FILE="${LOG_FILE:-$HOME/llm-services/logs/qwen-local-${PORT}.log}"
TUNE_TIMEOUT="${TUNE_TIMEOUT:-120}"
OPENCLAW_QWEN_MODE="${OPENCLAW_QWEN_MODE:-daemon}"
MODEL_ID="$(basename "$MODEL")"
OPENCLAW_MODEL_REF="qwen-local/${MODEL_ID}"

canonical_path() {
  local path="$1"

  if readlink -f / >/dev/null 2>&1; then
    readlink -f "$path"
  else
    python3 - "$path" <<'PY'
import os, sys
print(os.path.realpath(sys.argv[1]))
PY
  fi
}

sha256_file() {
  local path="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$path" | awk '{print $NF}'
  else
    echo "ERROR: no SHA-256 tool found (sha256sum, shasum, or openssl)." >&2
    return 1
  fi
}

# --- Install/update helpers -------------------------------------------------

check_nvidia() {
  echo "== Checking NVIDIA GPU =="
  command -v nvidia-smi >/dev/null || {
    echo "ERROR: nvidia-smi not found. Install NVIDIA driver first."
    exit 1
  }
  nvidia-smi
}

check_apple_silicon_metal() {
  echo "== Checking Apple Silicon/Metal =="

  [ "$(uname -s)" = "Darwin" ] || {
    echo "ERROR: Apple Silicon/Metal requires macOS."
    exit 1
  }

  [ "$(uname -m)" = "arm64" ] || {
    echo "ERROR: Apple Silicon/Metal requires an arm64 Mac."
    exit 1
  }

  clang --version >/dev/null 2>&1 || {
    echo "ERROR: Xcode Command Line Tools compiler not found."
    echo "Run: xcode-select --install"
    exit 1
  }

  echo "Apple Silicon/Metal OK"
  sysctl -n machdep.cpu.brand_string 2>/dev/null || true
}

detect_llama_backend() {
  if [ "${LLAMA_BACKEND:-auto}" != "auto" ]; then
    echo "$LLAMA_BACKEND"
    return 0
  fi

  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    echo "metal"
    return 0
  fi

  if [ "$(nvidia_vram_gb)" -gt 0 ]; then
    echo "cuda"
    return 0
  fi

  echo "cpu"
}

check_llama_backend() {
  local backend="$1"

  case "$backend" in
    cuda)
      check_nvidia
      ;;
    metal)
      check_apple_silicon_metal
      ;;
    cpu)
      echo "== No GPU backend detected; building CPU-only llama.cpp =="
      ;;
    *)
      echo "ERROR: Unknown LLAMA_BACKEND: $backend"
      echo "Expected one of: auto, cuda, metal, cpu"
      exit 1
      ;;
  esac
}

install_packages() {
  if [ "$(uname -s)" = "Darwin" ]; then
    echo "== Installing macOS packages =="

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

    brew install \
      git cmake pkg-config python curl wget psmisc sqlite3 node

    return 0
  fi

  echo "== Installing Linux packages =="
  sudo apt update
  sudo apt install -y \
    git cmake build-essential curl wget psmisc \
    python3 python3-pip python3-venv python3-full \
    libcurl4-openssl-dev libsqlite3-dev pkg-config sqlite3
}

build_llamacpp() {
  local backend="$1"
  local cmake_backend_flags=()

  echo "== Cloning/updating llama.cpp =="
  mkdir -p "$HOME/src"
  if [ -d "$LLAMA_DIR/.git" ]; then
    if [ -n "$LLAMA_CPP_REF" ]; then
      git -C "$LLAMA_DIR" fetch --tags origin "$LLAMA_CPP_REF"
      git -C "$LLAMA_DIR" checkout --detach FETCH_HEAD
    else
      git -C "$LLAMA_DIR" pull --ff-only
    fi
  else
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    if [ -n "$LLAMA_CPP_REF" ]; then
      git -C "$LLAMA_DIR" fetch --tags origin "$LLAMA_CPP_REF"
      git -C "$LLAMA_DIR" checkout --detach FETCH_HEAD
    fi
  fi

  if [ -n "$LLAMA_CPP_COMMIT" ]; then
    local actual_commit
    actual_commit="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
    case "$actual_commit" in
      "$LLAMA_CPP_COMMIT"*) echo "Verified llama.cpp commit: $actual_commit" ;;
      *)
        echo "ERROR: llama.cpp commit verification failed."
        echo "Expected: $LLAMA_CPP_COMMIT"
        echo "Actual:   $actual_commit"
        exit 1
        ;;
    esac
  fi

  case "$backend" in
    cuda)
      echo "== Building llama.cpp with CUDA =="
      cmake_backend_flags=(-DGGML_CUDA=ON)
      ;;
    metal)
      echo "== Building llama.cpp with Metal =="
      cmake_backend_flags=(-DGGML_METAL=ON)
      ;;
    cpu)
      echo "== Building llama.cpp CPU-only =="
      ;;
    *)
      echo "ERROR: Unknown llama.cpp backend: $backend"
      exit 1
      ;;
  esac

  # LLAMA_CURL keeps remote/model URL support available.
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    "${cmake_backend_flags[@]}" \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release

  cmake --build "$LLAMA_DIR/build" --config Release -j"$(build_jobs)" --target llama-server llama-cli llama-bench

echo "== Installing llama.cpp binaries =="
mkdir -p "$(dirname "$SERVER")" "$(dirname "$CLI")"

install_binary() {
  local src="$1"
  local dst="$2"

  if [ "$(canonical_path "$src")" = "$(canonical_path "$dst" 2>/dev/null || true)" ]; then
    echo "Already installed: $dst"
    return 0
  fi

  cp "$src" "$dst"
}

install_binary "$LLAMA_DIR/build/bin/llama-server" "$SERVER"
install_binary "$LLAMA_DIR/build/bin/llama-cli" "$CLI"
install_binary "$LLAMA_DIR/build/bin/llama-bench" "$HOME/.local/bin/llama-bench"
}

build_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  else
    sysctl -n hw.ncpu 2>/dev/null || echo 4
  fi
}

hf_venv_dir() {
  echo "$HOME/.local/share/qwen-local/hf-venv"
}

hf_bin() {
  echo "$(hf_venv_dir)/bin/hf"
}

install_hf_cli() {
  echo "== Installing Hugging Face downloader in isolated venv =="

  local HF_VENV
  HF_VENV="$(hf_venv_dir)"
  mkdir -p "$(dirname "$HF_VENV")"

  if [ ! -x "$HF_VENV/bin/python" ]; then
    python3 -m venv "$HF_VENV"
  fi

  "$HF_VENV/bin/python" -m pip install --upgrade pip
  "$HF_VENV/bin/python" -m pip install -U huggingface_hub hf_transfer

  [ -x "$(hf_bin)" ] || {
    echo "ERROR: hf CLI was not installed in $HF_VENV"
    exit 1
  }
}

verify_checksum() {
  if [ -n "$MODEL_SHA256" ] && [ -f "$MODEL" ]; then
    echo "== Verifying model checksum =="
    local actual
    actual="$(sha256_file "$MODEL")"
    if [ "$actual" != "$MODEL_SHA256" ]; then
      echo "ERROR: Model checksum mismatch."
      echo "Expected: $MODEL_SHA256"
      echo "Actual:   $actual"
      exit 1
    fi
    echo "Checksum OK"
  fi
}

download_model() {
  echo "== Downloading model if missing =="
  echo "MODEL_PRESET=$MODEL_PRESET"
  echo "MODEL_REPO=$MODEL_REPO"
  echo "MODEL_FILE=$MODEL_FILE"
  echo "MODEL_DIR=$MODEL_DIR"
  echo "MODEL=$MODEL"
  echo

  mkdir -p "$MODEL_DIR"

  if [ ! -f "$MODEL" ]; then
    echo "Model missing. Downloading expected file..."
    HF_HUB_ENABLE_HF_TRANSFER=1 "$(hf_bin)" download "$MODEL_REPO" "$MODEL_FILE" \
      --local-dir "$MODEL_DIR"
  else
    echo "Model already exists: $MODEL"
  fi

  if [ ! -f "$MODEL" ]; then
    echo
    echo "ERROR: Expected model file was not found after download:"
    echo "  $MODEL"
    echo
    echo "Files found under MODEL_DIR:"
    find "$MODEL_DIR" -maxdepth 5 -type f | sort || true
    echo
    echo "Likely causes:"
    echo "  1. MODEL_FILE does not exactly match the Hugging Face filename."
    echo "  2. You ran a different qwen-local.sh than the one you edited."
    echo "  3. The repo downloaded metadata or partial files but not the expected GGUF."
    echo
    echo "Check available repo files with:"
    echo "  $(hf_bin) repo files $MODEL_REPO"
    exit 1
  fi

  echo "Model found:"
  ls -lh "$MODEL"

  verify_checksum
}

install_all() {
  # Full first-time setup: OS packages, llama.cpp build, HF tools, and model file.
  local backend
  backend="$(detect_llama_backend)"
  install_packages
  check_llama_backend "$backend"
  build_llamacpp "$backend"
  install_hf_cli
  download_model
  echo
  echo "Install complete."
  print_cline
  print_openclaw_review
}

update_all() {
  # Refresh llama.cpp, HF downloader packages, and the model file if needed.
  local backend
  backend="$(detect_llama_backend)"
  check_llama_backend "$backend"
  build_llamacpp "$backend"
  install_hf_cli
  download_model
  echo "Update complete."
  print_openclaw_review
}

require_runtime_files() {
  [ -x "$SERVER" ] || {
    echo "ERROR: $SERVER not found. Run: $0 install"
    exit 1
  }

  [ -f "$MODEL" ] || {
    echo "ERROR: $MODEL not found. Run: $0 install"
    exit 1
  }
}

check_vram() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  local free_mb
  free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ' || true)"

  if [ -n "$free_mb" ]; then
    echo "VRAM free: ${free_mb} MiB"
  fi
}

write_pid_file() {
  local pid="$1"
  local pid_dir
  local pid_tmp

  pid_dir="$(dirname "$PID_FILE")"
  mkdir -p "$pid_dir"
  if [ "$PID_FILE_IS_DEFAULT" -eq 1 ]; then
    chmod 700 "$pid_dir"
  fi
  pid_tmp="$(mktemp "${PID_FILE}.XXXXXX")"
  printf '%s\n' "$pid" > "$pid_tmp"
  mv -f "$pid_tmp" "$PID_FILE"
}

DAEMON_PID=""
DAEMON_PID_FILE=""

cleanup_owned_daemon() {
  local status=$?
  local recorded_pid=""

  trap - EXIT INT TERM HUP
  if [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    kill "$DAEMON_PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
      kill -9 "$DAEMON_PID" 2>/dev/null || true
    fi
  fi

  if [ -n "$DAEMON_PID_FILE" ] && [ -f "$DAEMON_PID_FILE" ]; then
    recorded_pid="$(cat "$DAEMON_PID_FILE" 2>/dev/null || true)"
    [ "$recorded_pid" = "$DAEMON_PID" ] && rm -f "$DAEMON_PID_FILE"
  fi

  return "$status"
}

arm_daemon_cleanup() {
  DAEMON_PID="$1"
  DAEMON_PID_FILE="$PID_FILE"
  trap cleanup_owned_daemon EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
}

disarm_daemon_cleanup() {
  trap - EXIT INT TERM HUP
  DAEMON_PID=""
  DAEMON_PID_FILE=""
}

kill_port() {
  # Keep startup deterministic by clearing anything already bound to this port.
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null \
      | while IFS= read -r pid; do
          [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
        done
  else
    echo "WARNING: cannot clear port $PORT; install fuser or lsof." >&2
  fi
  sleep 2
}

show_port_status() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp | grep ":${PORT}"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN
  elif command -v netstat >/dev/null 2>&1; then
    netstat -an | grep -E "[.:]${PORT}[[:space:]].*LISTEN"
  else
    return 1
  fi
}

# --- Server lifecycle -------------------------------------------------------

start_server() {
  local reasoning="$1"
  local args=()

  if [ "$reasoning" = "on" ]; then
    args=(--reasoning on --reasoning-budget "$REASONING_BUDGET")
  else
    args=(--reasoning off)
  fi

  require_runtime_files
  check_vram
  kill_port

  local log_dir="$HOME/llm-services/logs"
  mkdir -p "$log_dir"
  local log_file="$log_dir/qwen3-$(date +%F-%H%M%S).log"

  echo "Starting llama-server:"
  echo "  MODEL=$MODEL"
  echo "  HOST=$HOST"
  echo "  PORT=$PORT"
  echo "  CTX=$CTX"
  echo "  MAX_TOKENS=$MAX_TOKENS"
  echo "  NGL=$NGL"
  echo "  PARALLEL=$PARALLEL"
  echo "  BATCH=$BATCH"
  echo "  UBATCH=$UBATCH"
  echo "  FLASH_ATTN=$FLASH_ATTN"
  echo "  REASONING=$reasoning"
  echo "  LOG=$log_file"

  "$SERVER" \
    -m "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    -ngl "$NGL" \
    -c "$CTX" \
    -n "$MAX_TOKENS" \
    -np "$PARALLEL" \
    -b "$BATCH" \
    -ub "$UBATCH" \
    --flash-attn "$FLASH_ATTN" \
    "${args[@]}" \
    2>&1 | tee "$log_file"
}

start_daemon() {
  local reasoning="$1"
  local args=()

  require_runtime_files

  # PID files only guard daemon mode. Foreground starts still clear by port.
  if [ -f "$PID_FILE" ]; then
    local old_pid
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Server already running. PID: $old_pid"
      echo "Stop with: $0 stop"
      exit 1
    fi
    # Stale PID file - try to remove it
    rm -f "$PID_FILE"
    # Double-check after removal
    if [ -f "$PID_FILE" ]; then
      local stale_pid
      stale_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
      if [ -n "$stale_pid" ] && ! kill -0 "$stale_pid" 2>/dev/null; then
        echo "Removing stale PID file"
        rm -f "$PID_FILE"
      fi
    fi
  fi

  if [ "$reasoning" = "on" ]; then
    args=(--reasoning on --reasoning-budget "$REASONING_BUDGET")
  else
    args=(--reasoning off)
  fi

  check_vram
  kill_port
  mkdir -p "$(dirname "$LOG_FILE")"

  # Use nohup so the server survives the shell that launched it.
  echo "Starting qwen server in background..."
  nohup "$SERVER" \
    -m "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    -ngl "$NGL" \
    -c "$CTX" \
    -n "$MAX_TOKENS" \
    -np "$PARALLEL" \
    -b "$BATCH" \
    -ub "$UBATCH" \
    --flash-attn "$FLASH_ATTN" \
    "${args[@]}" \
    >> "$LOG_FILE" 2>&1 &

  local pid=$!
  arm_daemon_cleanup "$pid"
  write_pid_file "$pid"

  # Wait until the OpenAI-compatible model endpoint responds before returning.
  echo "Waiting for server to become ready..."
  local attempt=0
  local max_attempts=2
  local delay=1

  while [ "$attempt" -lt "$max_attempts" ]; do
    sleep "$delay"
    attempt=$((attempt + 1))
    if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "Server started. PID: $pid"
      echo "Log: $LOG_FILE"
      disarm_daemon_cleanup
      return 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: server exited during startup."
      tail -n 40 "$LOG_FILE" || true
      exit 1
    fi

    delay=2
  done

  echo "ERROR: server did not become ready after $max_attempts attempts (3 seconds)."
  tail -n 40 "$LOG_FILE" || true
  exit 1
}

stop_server() {
  # Prefer the daemon PID when present, then clear the port as a fallback.
  if [ -f "$PID_FILE" ]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ]; then
      kill "$pid" 2>/dev/null || true
      for _ in {1..10}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi

  kill_port
  echo "Stopped/cleared port $PORT"
}

status_server() {
  echo "Configured endpoint: http://${HOST}:${PORT}"
  echo "PID file: $PID_FILE"
  if [ -f "$PID_FILE" ]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "PID running: $pid"
    else
      echo "PID file exists but process is not running"
    fi
  fi

  show_port_status || echo "qwen server is not running on port $PORT"
}

test_server() {
  echo "Testing server endpoint..."
  local tmp_dir
  local response
  local http_status
  tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/qwen-test.XXXXXX")"

  if ! http_status=$(curl -fsS -o "$tmp_dir/response" -w '%{http_code}' \
    "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL_ID}\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"Say only: qwen works\"}
      ],
      \"temperature\": 0.2
    }" 2>"$tmp_dir/error"); then
    echo "ERROR: Server test failed"
    [ -s "$tmp_dir/error" ] && cat "$tmp_dir/error"
    [ -s "$tmp_dir/response" ] && cat "$tmp_dir/response"
    rm -rf "$tmp_dir"
    return 1
  fi

  case "$http_status" in
    2??) ;;
    *)
      echo "ERROR: Server returned HTTP $http_status"
      cat "$tmp_dir/response"
      rm -rf "$tmp_dir"
      return 1
      ;;
  esac

  response="$(cat "$tmp_dir/response")"
  rm -rf "$tmp_dir"
  echo "Server responding OK"
  echo "$response"
  return 0
}

print_cline() {
  echo
  echo "Cline settings:"
  echo "  Provider: OpenAI Compatible"
  echo "  Base URL: http://${HOST}:${PORT}/v1"
  echo "  API Key: local"
  echo "  Model ID: ${MODEL_ID}"
  echo "  Context Window: ${CTX}"
  echo "  Max Output Tokens: ${MAX_TOKENS}"
  echo "  Parallel Slots: ${PARALLEL}"
  echo
}

print_vscode() {
  echo
  echo "llama-vscode localStartCommand:"
  echo "bash -lc 'cd $(pwd) && CTX=${CTX} NGL=${NGL} MAX_TOKENS=${MAX_TOKENS} PARALLEL=${PARALLEL} BATCH=${BATCH} UBATCH=${UBATCH} PORT=${PORT} $0 reasoning'"
  echo
  echo "llama-vscode endpoint:"
  echo "http://${HOST}:${PORT}"
  echo
}

json_string() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/^/"/; s/$/"/'
}

node_version_ok() {
  local version="${1#v}"
  local major minor patch
  IFS=. read -r major minor patch <<< "$version"
  major="${major:-0}"
  minor="${minor:-0}"
  patch="${patch:-0}"

  [ "$major" -ge 24 ] ||
    { [ "$major" -eq 22 ] && [ "$minor" -gt 19 ]; } ||
    { [ "$major" -eq 22 ] && [ "$minor" -eq 19 ] && [ "$patch" -ge 0 ]; }
}

check_openclaw_dependencies() {
  echo "== Checking OpenClaw dependencies =="

  command -v node >/dev/null || {
    echo "ERROR: node not found. Install Node 24 or Node 22.19+ first."
    exit 1
  }

  command -v npm >/dev/null || {
    echo "ERROR: npm not found. Install npm first."
    exit 1
  }

  local node_version
  node_version="$(node --version)"
  if ! node_version_ok "$node_version"; then
    echo "ERROR: OpenClaw requires Node 24 or Node 22.19+."
    echo "Current node: $node_version"
    exit 1
  fi

  echo "node: $node_version"
  echo "npm:  $(npm --version)"
}

global_npm_install() {
  local package="$1"
  local npm_prefix

  npm_prefix="$(npm prefix -g)"
  if [ -w "$npm_prefix" ]; then
    npm install -g "$package"
  else
    echo "WARNING: npm will run as root to install $package globally." >&2
    sudo npm install -g "$package"
  fi
}

install_or_update_openclaw_package() {
  echo "== Installing/updating OpenClaw =="

  global_npm_install openclaw@latest

  command -v openclaw >/dev/null || {
    echo "ERROR: openclaw was not found on PATH after npm install."
    exit 1
  }
}

install_openclaw_tool_dependencies() {
  echo "== Installing OpenClaw tool dependencies =="

  command -v codex >/dev/null || {
    echo "Installing Codex CLI..."
    global_npm_install @openai/codex
  }

  command -v codex >/dev/null || {
    echo "ERROR: codex was not found on PATH after install."
    exit 1
  }

  echo "Checking Context7 MCP package..."
  npx -y @upstash/context7-mcp@latest --help >/dev/null

  echo "Installing Playwright Chromium..."
  npx playwright install chromium
}

openclaw_provider_json() {
  local script_path
  script_path="$(canonical_path "$0")"

  cat <<EOF
{"baseUrl":$(json_string "http://${HOST}:${PORT}/v1"),"apiKey":"local","api":"openai-completions","timeoutSeconds":300,"localService":{"command":$(json_string "$script_path"),"args":[$(json_string "$OPENCLAW_QWEN_MODE")],"cwd":$(json_string "$(dirname "$script_path")"),"env":{"MODEL_PRESET":$(json_string "$MODEL_PRESET"),"QWEN_MODEL_CATALOG":$(json_string "$CATALOG"),"MODEL_REPO":$(json_string "$MODEL_REPO"),"MODEL_FILE":$(json_string "$MODEL_FILE"),"MODEL_DIR":$(json_string "$MODEL_DIR"),"MODEL":$(json_string "$MODEL"),"HOST":$(json_string "$HOST"),"PORT":$(json_string "$PORT"),"CTX":$(json_string "$CTX"),"NGL":$(json_string "$NGL"),"PARALLEL":$(json_string "$PARALLEL"),"BATCH":$(json_string "$BATCH"),"UBATCH":$(json_string "$UBATCH"),"FLASH_ATTN":$(json_string "$FLASH_ATTN"),"REASONING_BUDGET":$(json_string "$REASONING_BUDGET"),"REASONING":$(json_string "$REASONING"),"LLAMA_BACKEND":$(json_string "${LLAMA_BACKEND:-auto}")},"healthUrl":$(json_string "http://${HOST}:${PORT}/v1/models"),"readyTimeoutMs":180000,"idleStopMs":0},"models":[{"id":$(json_string "$MODEL_ID"),"name":$(json_string "$MODEL_PRESET local llama.cpp"),"reasoning":true,"input":["text"],"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0},"contextWindow":$CTX,"maxTokens":$MAX_TOKENS}]}
EOF
}

# Validate JSON output
validate_json() {
  local json_str="$1"
  echo "$json_str" | jq . >/dev/null 2>&1
  return $?
}

openclaw_allowlist_json() {
  cat <<EOF
{$(json_string "$OPENCLAW_MODEL_REF"):{"alias":$(json_string "$MODEL_PRESET")}}
EOF
}

configure_openclaw_for_qwen() {
  command -v openclaw >/dev/null || {
    echo "ERROR: openclaw not found. Run: $0 openclaw-install"
    exit 1
  }

  local script_path
  local plugin_dir
  script_path="$(canonical_path "$0")"
  plugin_dir="$(dirname "$script_path")/plugins/codex-exec"

  echo "== Configuring OpenClaw local model provider =="
  if [ -f "$plugin_dir/package.json" ] && [ -f "$plugin_dir/openclaw.plugin.json" ]; then
    echo "== Installing linked OpenClaw codex-exec plugin =="
    openclaw plugins install --link "$plugin_dir"
    openclaw config set 'plugins.entries["codex-exec"].config.defaultCwd' "$(json_string "$(dirname "$script_path")")" --strict-json
  fi
  openclaw config set models.providers.qwen-local "$(openclaw_provider_json)" --strict-json --merge
  openclaw config set agents.defaults.models "$(openclaw_allowlist_json)" --strict-json --merge
  openclaw config set agents.defaults.model.primary "$(json_string "$OPENCLAW_MODEL_REF")" --strict-json
  # Keep the Codex MCP server installed, but force the local Qwen coordinator
  # to use the repo-owned codex_exec OpenClaw plugin instead of codex__*.
  openclaw config set 'tools.byProvider["qwen-local"].deny' '["codex__*"]' --strict-json
}

print_openclaw_review() {
  echo
  echo "OpenClaw local model settings:"
  echo "  Provider ID: qwen-local"
  echo "  Model ref:   $OPENCLAW_MODEL_REF"
  echo "  Catalog:     $CATALOG"
  echo "  Base URL:    http://${HOST}:${PORT}/v1"
  echo "  Health URL:  http://${HOST}:${PORT}/v1/models"
  echo "  Start mode:  $OPENCLAW_QWEN_MODE"
  echo "  Model file:  $MODEL"
  echo "  CTX:         $CTX"
  echo "  MAX_TOKENS:     $MAX_TOKENS"
  echo "  NGL:         $NGL"
  echo "  PARALLEL:    $PARALLEL"
  echo "  BATCH:       $BATCH"
  echo "  UBATCH:      $UBATCH"
  echo "  FLASH_ATTN:  $FLASH_ATTN"
  echo
  echo "OpenClaw commands:"
  echo "  Install/update OpenClaw with these settings:"
  echo "    $0 openclaw-install"
  echo "  Review these settings:"
  echo "    $0 openclaw-review"
  echo "  Start OpenClaw onboarding daemon setup:"
  echo "    openclaw onboard --install-daemon"
  echo "  Check OpenClaw:"
  echo "    openclaw doctor"
  echo "    openclaw models status"
  echo "  Tool dependencies installed by openclaw-install:"
  echo "    codex"
  echo "    npx -y @upstash/context7-mcp@latest --help"
  echo "    npx playwright install chromium"
  echo "    linked OpenClaw plugin: plugins/codex-exec"
  echo
  echo "Codex routing:"
  echo "  qwen-local denies codex__* tools so OpenClaw uses the codex_exec plugin tool"
  echo "  Codex MCP stays installed for direct Codex management outside the local coordinator"
  echo
}

install_openclaw() {
  check_openclaw_dependencies
  install_or_update_openclaw_package
  install_openclaw_tool_dependencies
  configure_openclaw_for_qwen
  print_openclaw_review

  echo "OpenClaw install/update complete."
  echo "Run onboarding when ready: openclaw onboard --install-daemon"
}

# --- Tuning -----------------------------------------------------------------

openclaw_prompt_estimate() {
  local fallback="${1:-11000}"
  local log_file="$HOME/.openclaw/logs/manual-gateway.log"
  local best="$fallback"

  if [ -f "$log_file" ]; then
    local detected_tokens
    detected_tokens="$(
      grep -oE 'estimatedPromptTokens=[0-9]+' "$log_file" 2>/dev/null \
        | tail -n 1 \
        | cut -d= -f2 \
        || true
    )"
    if [[ "$detected_tokens" =~ ^[0-9]+$ ]] && [ "$detected_tokens" -gt "$best" ]; then
      best="$detected_tokens"
    fi

    local overflow_tokens
    overflow_tokens="$(
      grep -oE 'request \([0-9]+ tokens\)' "$log_file" 2>/dev/null \
        | tail -n 1 \
        | grep -oE '[0-9]+' \
        || true
    )"
    if [[ "$overflow_tokens" =~ ^[0-9]+$ ]] && [ "$overflow_tokens" -gt "$best" ]; then
      best="$overflow_tokens"
    fi
  fi

  echo "$best"
}

openclaw_config_value() {
  local query="$1"
  local fallback="$2"
  local config="$HOME/.openclaw/openclaw.json"

  if command -v jq >/dev/null 2>&1 && [ -f "$config" ]; then
    jq -r "$query // empty" "$config" 2>/dev/null | head -n 1
    return 0
  fi

  echo "$fallback"
}

next_context_candidate() {
  local required="$1"
  local candidates="${2:-8192,12288,16384,24576,32768,49152,65536}"
  local candidate

  IFS=',' read -ra candidate_list <<< "$candidates"
  for candidate in "${candidate_list[@]}"; do
    candidate="${candidate//[[:space:]]/}"
    if [[ "$candidate" =~ ^[0-9]+$ ]] && [ "$candidate" -ge "$required" ]; then
      echo "$candidate"
      return 0
    fi
  done

  echo ""
}

list_prompt_files_with_sizes() {
  local workspace="$1"

  if find "$workspace" -maxdepth 0 -printf '' >/dev/null 2>&1; then
    find "$workspace" -maxdepth 2 -type f \
      \( -name 'AGENTS.md' -o -name 'SOUL.md' -o -name 'TOOLS.md' -o -name 'HEARTBEAT.md' -o -name 'MEMORY.md' -o -path '*/memory/*.md' \) \
      -printf '%s %p\n' 2>/dev/null
  else
    python3 - "$workspace" <<'PY'
import os, sys

root = os.path.abspath(sys.argv[1])
names = {'AGENTS.md', 'SOUL.md', 'TOOLS.md', 'HEARTBEAT.md', 'MEMORY.md'}
for current, dirs, files in os.walk(root):
    relative_dir = os.path.relpath(current, root)
    depth = 0 if relative_dir == '.' else relative_dir.count(os.sep) + 1
    if depth >= 1:
        dirs[:] = []
    for name in files:
        path = os.path.join(current, name)
        relative = os.path.relpath(path, root)
        if name in names or (relative.startswith('memory' + os.sep) and name.endswith('.md')):
            try:
                print(f'{os.path.getsize(path)} {path}')
            except OSError:
                pass
PY
  fi
}

diagnose_openclaw() {
  local config="$HOME/.openclaw/openclaw.json"
  local workspace="$HOME/.openclaw/workspace"
  local startup_tokens
  local reserve_tokens
  local margin_tokens
  local required_ctx
  local suggested_ctx

  startup_tokens="$(openclaw_prompt_estimate "${OPENCLAW_STARTUP_TOKENS:-11000}")"
  reserve_tokens="$(openclaw_config_value '.agents.defaults.compaction.reserveTokens' "${OPENCLAW_RESERVE_TOKENS:-2048}")"
  reserve_tokens="${reserve_tokens:-${OPENCLAW_RESERVE_TOKENS:-2048}}"
  margin_tokens="${OPENCLAW_MARGIN_TOKENS:-1024}"
  required_ctx=$((startup_tokens + reserve_tokens + margin_tokens))
  suggested_ctx="$(next_context_candidate "$required_ctx" "${CTX_CANDIDATES:-8192,12288,16384,24576,32768}")"

  echo "OpenClaw prompt/context diagnosis"
  echo
  echo "Prompt budget:"
  echo "  startup estimate: $startup_tokens"
  echo "  reserve:          $reserve_tokens"
  echo "  margin:           $margin_tokens"
  echo "  required ctx:     $required_ctx"
  if [ -n "$suggested_ctx" ]; then
    echo "  suggested ctx:    $suggested_ctx"
  else
    echo "  suggested ctx:    none in CTX_CANDIDATES"
  fi
  echo

  echo "Largest local prompt files:"
  if [ -d "$workspace" ]; then
    list_prompt_files_with_sizes "$workspace" \
      | sort -nr \
      | head -n 12 \
      | while read -r size path; do
          printf "  %-8s %s\n" "$size" "${path#$workspace/}"
        done
  else
    echo "  workspace not found: $workspace"
  fi
  echo

  echo "Enabled configurable context sources:"
  if command -v jq >/dev/null 2>&1 && [ -f "$config" ]; then
    local tools_profile
    local web_enabled
    local memory_enabled

    tools_profile="$(jq -r '.tools.profile // "unset"' "$config" 2>/dev/null)"
    web_enabled="$(jq -r '.tools.web.search.enabled // false' "$config" 2>/dev/null)"
    memory_enabled="$(jq -r '.agents.defaults.memorySearch.enabled // false' "$config" 2>/dev/null)"
    echo "  tools.profile=$tools_profile"
    echo "  web.search.enabled=$web_enabled"
    echo "  memorySearch.enabled=$memory_enabled"

    jq -r '.mcp.servers // {} | keys[] | "  mcp.\(.) enabled"' "$config" 2>/dev/null || true
    jq -r '.plugins.entries // {} | to_entries[] | select(.value.enabled == true) | "  plugin.\(.key) enabled"' "$config" 2>/dev/null || true
    jq -r '.skills.entries // {} | to_entries[] | select(.value.enabled == true) | "  skill.\(.key) enabled"' "$config" 2>/dev/null || true
  else
    echo "  jq or config unavailable; config=$config"
  fi
  echo

  echo "Recommended actions:"
  if [ -n "$suggested_ctx" ]; then
    echo "  use CTX >= $suggested_ctx for the current OpenClaw prompt load"
  else
    echo "  reduce OpenClaw prompt/tool context or add a larger CTX_CANDIDATES value"
  fi
  if command -v jq >/dev/null 2>&1 && [ -f "$config" ]; then
    if jq -e '.mcp.servers.context7? != null' "$config" >/dev/null 2>&1; then
      echo "  disable mcp.context7 unless this session needs docs lookup"
    fi
    if jq -e '.mcp.servers.playwright? != null' "$config" >/dev/null 2>&1; then
      echo "  disable mcp.playwright unless this session needs browser checks"
    fi
    if [ "$(jq -r '.tools.profile // empty' "$config" 2>/dev/null)" = "coding" ]; then
      echo "  consider tools.profile=minimal for the local controller model"
    fi
    if jq -e '.mcp.servers.codex? != null' "$config" >/dev/null 2>&1; then
      echo "  keep mcp.codex enabled for coding delegation"
    fi
  fi
}

tune_settings() {
  require_runtime_files

  local best_ctx=""
  local best_ngl=""

  local candidates=(
    # Largest known-good context first, then smaller fallbacks.
    "65536 999"
    "49152 999"
    "32768 999"
    "24576 999"
    "16384 999"
    "32768 120"
    "32768 100"
    "32768 80"
    "16384 80"
  )

  echo "Testing settings..."
  echo

  for candidate in "${candidates[@]}"; do
    local test_ctx
    local test_ngl
    local tune_log
    test_ctx="$(awk '{print $1}' <<< "$candidate")"
    test_ngl="$(awk '{print $2}' <<< "$candidate")"
    tune_log="$(mktemp "${TMPDIR:-/tmp}/qwen-tune-${test_ctx}-${test_ngl}.log.XXXXXX")"

    echo "== Trying CTX=${test_ctx}, NGL=${test_ngl} =="
    kill_port

    "$SERVER" \
      -m "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      -ngl "$test_ngl" \
      -c "$test_ctx" \
      --reasoning off \
      > "$tune_log" 2>&1 &

    local pid=$!
    local ready=0
    local elapsed=0

    while [ "$elapsed" -lt "$TUNE_TIMEOUT" ]; do
      if curl -s "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
        ready=1
        break
      fi

      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi

      sleep 1
      elapsed=$((elapsed + 1))
    done

    if [ "$ready" -ne 1 ]; then
      echo "FAILED startup after ${elapsed}s"
      tail -n 12 "$tune_log" || true
      kill "$pid" 2>/dev/null || true
      rm -f "$tune_log"
      echo
      continue
    fi

    local response
    response="$(curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"${MODEL_ID}\",
        \"messages\": [
          {\"role\": \"user\", \"content\": \"Say only: qwen works\"}
        ],
        \"temperature\": 0.2
      }" || true)"

    if grep -q "qwen works" <<< "$response"; then
      echo "PASS"
      best_ctx="$test_ctx"
      best_ngl="$test_ngl"
      kill_port
      rm -f "$tune_log"
      break
    fi

    echo "FAILED chat test"
    echo "$response"
    kill_port
    rm -f "$tune_log"
    echo
  done

  [ -n "$best_ctx" ] || {
    echo "No settings passed."
    exit 1
  }

  echo
  echo "Best passing settings:"
  echo "  CTX=$best_ctx"
  echo "  NGL=$best_ngl"
  echo
  echo "Start command:"
  echo "  CTX=$best_ctx NGL=$best_ngl $0 fast"
  echo
  CTX="$best_ctx" NGL="$best_ngl" print_cline
  CTX="$best_ctx" NGL="$best_ngl" print_vscode
}

ensure_llama_bench() {
  local bench="${LLAMA_BENCH:-$HOME/.local/bin/llama-bench}"

  if [ -x "$bench" ]; then
    echo "$bench"
    return 0
  fi

  if command -v llama-bench >/dev/null 2>&1; then
    command -v llama-bench
    return 0
  fi

  echo "llama-bench not found; building it from llama.cpp..." >&2

  local backend
  backend="$(detect_llama_backend)"
  check_llama_backend "$backend" >&2

  [ -d "$LLAMA_DIR" ] || {
    echo "ERROR: llama.cpp source not found:" >&2
    echo "  $LLAMA_DIR" >&2
    echo "Run: $0 install" >&2
    exit 1
  }

  local cmake_backend_flags=()
  case "$backend" in
    cuda)
      cmake_backend_flags=(-DGGML_CUDA=ON)
      ;;
    metal)
      cmake_backend_flags=(-DGGML_METAL=ON)
      ;;
    cpu)
      cmake_backend_flags=()
      ;;
    *)
      echo "ERROR: Unknown llama.cpp backend: $backend" >&2
      exit 1
      ;;
  esac

  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    "${cmake_backend_flags[@]}" \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release >&2

  cmake --build "$LLAMA_DIR/build" --config Release -j"$(build_jobs)" --target llama-bench >&2

  mkdir -p "$(dirname "$bench")"

  if [ "$(canonical_path "$LLAMA_DIR/build/bin/llama-bench")" = "$(canonical_path "$bench" 2>/dev/null || true)" ]; then
    echo "Already installed: $bench" >&2
  else
    cp "$LLAMA_DIR/build/bin/llama-bench" "$bench"
  fi

  [ -x "$bench" ] || {
    echo "ERROR: llama-bench build/install failed:" >&2
    echo "  $bench" >&2
    exit 1
  }

  echo "$bench"
}

tune_openclaw_settings() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  local tuner="${QWEN_OPENCLAW_TUNER:-}"

  if [ -z "$tuner" ]; then
    if [ -f "$script_dir/qwen-openclaw-optuna-tune.py" ]; then
      tuner="$script_dir/qwen-openclaw-optuna-tune.py"
    elif [ -f "$HOME/models/qwen-openclaw-optuna-tune.py" ]; then
      tuner="$HOME/models/qwen-openclaw-optuna-tune.py"
    elif [ -f "$HOME/models/qwen-optuna-tune.py" ]; then
      echo "ERROR: found old tuner, but the bench-based tuner is missing:"
      echo "  old:     $HOME/models/qwen-optuna-tune.py"
      echo "  missing: $HOME/models/qwen-openclaw-optuna-tune.py"
      echo
      echo "Install the new tuner:"
      echo "  cp ~/Downloads/qwen-openclaw-optuna-tune.py $HOME/models/"
      echo "  chmod +x $HOME/models/qwen-openclaw-optuna-tune.py"
      exit 1
    else
      echo "ERROR: OpenClaw bench tuner not found."
      echo "Checked:"
      echo "  $script_dir/qwen-openclaw-optuna-tune.py"
      echo "  $HOME/models/qwen-openclaw-optuna-tune.py"
      echo
      echo "Install it with:"
      echo "  cp ~/Downloads/qwen-openclaw-optuna-tune.py $HOME/models/"
      echo "  chmod +x $HOME/models/qwen-openclaw-optuna-tune.py"
      exit 1
    fi
  fi

  [ -f "$tuner" ] || {
    echo "ERROR: OpenClaw bench tuner not found:"
    echo "  $tuner"
    exit 1
  }

  [ -x "$SERVER" ] || {
    echo "ERROR: llama-server not found:"
    echo "  $SERVER"
    echo "Run: $0 install"
    exit 1
  }

  local bench
  bench="$(ensure_llama_bench)"

  echo "Tuning OpenClaw with llama-bench..."
  echo "  tuner:       $tuner"
  echo "  llama-server:$SERVER"
  echo "  llama-bench: $bench"
  echo "  host:        ${TUNE_HOST:-127.0.0.1}"
  echo "  port:        $PORT"
  echo

  python3 "$tuner" \
    --catalog "$CATALOG" \
    --qwen-local "$(canonical_path "$0")" \
    --llama-server "$SERVER" \
    --llama-bench "$bench" \
    --host "${TUNE_HOST:-127.0.0.1}" \
    --port "$PORT" \
    --openclaw-config "${OPENCLAW_CONFIG:-$HOME/.openclaw/openclaw.json}" \
    --env-file "${OPENCLAW_ENV_FILE:-$HOME/.config/qwen-local/openclaw.env}"
}

# --- Command dispatch -------------------------------------------------------

case "$MODE" in
  install)
    install_all
    ;;
  update)
    update_all
    ;;
  fast)
    start_server off
    ;;
  reasoning)
    start_server on
    ;;
  daemon)
    start_daemon off
    ;;
  daemon-reasoning)
    start_daemon on
    ;;
  stop)
    stop_server
    ;;
  status)
    status_server
    ;;
  test)
    test_server
    ;;
  cline)
    print_cline
    ;;
  vscode)
    print_vscode
    ;;
  tune)
    tune_settings
    ;;
  diagnose-openclaw)
    diagnose_openclaw
    ;;
  tune-openclaw)
    tune_openclaw_settings
    ;;
  openclaw-install)
    install_openclaw
    ;;
  openclaw-review)
    print_openclaw_review
    ;;
  *)
    echo "Usage: $0 {install|update|fast|reasoning|daemon|daemon-reasoning|stop|status|test|cline|vscode|tune|diagnose-openclaw|tune-openclaw|openclaw-install|openclaw-review}"
    echo
    echo "Install/update:"
    echo "  $0 install                 Choose model, install deps, build llama.cpp, download model"
    echo "  $0 update                  Choose model, update llama.cpp/HF CLI/model"
    echo
    echo "Start:"
    echo "  $0 fast                    Start server in foreground, reasoning off"
    echo "  $0 reasoning               Start server in foreground, reasoning on"
    echo "  $0 daemon                  Start server in background, reasoning off"
    echo "  $0 daemon-reasoning        Start server in background, reasoning on"
    echo
    echo "Manage/test:"
    echo "  $0 stop                    Stop server on configured port"
    echo "  $0 status                  Show server status"
    echo "  $0 test                    Send simple chat request"
    echo
    echo "Config:"
    echo "  $0 cline                   Print Cline config"
    echo "  $0 vscode                  Print llama-vscode command/endpoint"
    echo "  $0 tune                    Try CTX/NGL settings"
    echo "  $0 diagnose-openclaw       Diagnose OpenClaw prompt/context load"
    echo "  $0 tune-openclaw           Tune with OpenClaw startup prompt headroom"
    echo "  $0 openclaw-review         Print OpenClaw local model settings"
    echo "  $0 openclaw-install        Install/update OpenClaw and apply local model settings"
    echo
    echo "Environment overrides:"
    echo "  QWEN_MODEL_CATALOG MODEL_PRESET MODEL_REPO MODEL_FILE MODEL_DIR MODEL MODEL_SHA256"
    echo "  LLAMA_BACKEND LLAMA_DIR LLAMA_CPP_REF LLAMA_CPP_COMMIT SERVER CLI OPENCLAW_QWEN_MODE"
    echo "  HOST PORT CTX NGL REASONING_BUDGET MAX_TOKENS PARALLEL BATCH UBATCH FLASH_ATTN"
    echo "  PID_FILE LOG_FILE TUNE_TIMEOUT"
    ;;
esac
