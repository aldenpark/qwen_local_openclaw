#!/usr/bin/env python3
"""
qwen-openclaw-optuna-tune.py

Replacement tuner for llama.cpp + OpenClaw.

Hard rule: a candidate is not valid just because llama.cpp starts or a synthetic
chat works. The useful constraint is OpenClaw's real request size. This script
parses OpenClaw/llama-server overflow logs, refuses contexts below the observed
need, then uses Optuna to tune CTX/BATCH/UBATCH/etc. around viable settings.

Optional: pass --openclaw-smoke-cmd with a real noninteractive OpenClaw command.
If provided, candidates only pass if that command exits 0 and the gateway logs do
not report a new context overflow.

"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import hashlib
import re
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VENV = Path.home() / ".local/share/qwen-local/hf-venv"
VENV_PY = VENV / "bin/python"

def in_managed_venv() -> bool:
    return os.environ.get("QWEN_OPTUNA_IN_VENV") == "1"

def ensure_venv_and_deps() -> None:
    if in_managed_venv():
        return

    if not VENV_PY.exists():
        subprocess.check_call([os.sys.executable, "-m", "venv", str(VENV)])

    subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-U", "pip", "optuna"])

    env = os.environ.copy()
    env["QWEN_OPTUNA_IN_VENV"] = "1"
    os.execve(str(VENV_PY), [str(VENV_PY), __file__, *os.sys.argv[1:]], env)

ensure_venv_and_deps()

try:
    import optuna
except ImportError:
    print("ERROR: optuna is not installed in the managed venv.", file=os.sys.stderr)
    print(f"Venv Python: {VENV_PY}", file=os.sys.stderr)
    print(f"Try: {VENV_PY} -m pip install -U optuna", file=os.sys.stderr)
    raise SystemExit(2)


@dataclasses.dataclass(frozen=True)
class ModelPreset:
    name: str
    path: Path
    size_gb: float


@dataclasses.dataclass(frozen=True)
class Candidate:
    ctx: int
    ngl: int
    predict: int
    parallel: int
    batch: int
    ubatch: int
    flash_attn: str
    reasoning_budget: int
    mode: str


@dataclasses.dataclass
class TrialResult:
    ok: bool
    score: float
    error: str = ""
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    elapsed_s: float = 0.0
    overflow_tokens: int = 0
    server_log: str = ""


REQUEST_OVERFLOW_RE = re.compile(
    r"request\s+\((\d+)\s+tokens\)\s+exceeds\s+the\s+available\s+context\s+size\s+\((\d+)\s+tokens\)",
    re.IGNORECASE,
)
PRECHECK_RE = re.compile(
    r"estimatedPromptTokens=(\d+).*?promptBudgetBeforeReserve=(\d+).*?reserveTokens=(\d+)",
    re.IGNORECASE,
)
CTX_TIERS = [
    8192, 12288, 16384, 18432, 20480, 22528, 24576, 28672,
    32768, 40960, 49152, 65536, 98304, 131072, 196608, 262144,
]


def eprint(*args: object) -> None:
    print(*args, file=os.sys.stderr)


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                time.sleep(0.2)
    return False


def terminate_process(proc: subprocess.Popen[Any], timeout_s: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    with contextlib.suppress(Exception):
        proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout_s)


def http_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, Any, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if "application/json" in resp.headers.get("Content-Type", ""):
                with contextlib.suppress(Exception):
                    return resp.status, json.loads(body), body
            return resp.status, body, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        with contextlib.suppress(Exception):
            return e.code, json.loads(body), body
        return e.code, body, body
    except Exception as e:
        return 0, None, repr(e)


def canonical_model_preset_for_path(path: Path) -> str:
    """Return the qwen-local.sh preset name for a known GGUF filename.

    The tuner must write MODEL_PRESET back to openclaw.env because qwen-local.sh
    validates the preset before applying MODEL overrides, and OpenClaw's
    localService env also uses it. Do not use broad discovery labels like
    qwen35-4b here; return the exact launcher preset.
    """
    name = path.name.lower()
    full = str(path).lower()

    mapping = [
        ("qwen3.5-4b-q5_k_m", "qwen35-4b-q5km"),
        ("qwen3.5-4b-q4_k_m", "qwen35-4b-q4km"),
        ("qwen3.5-9b-q4_k_m", "qwen35-9b-q4km"),
        ("qwen3.5-9b-ud-q4_k_xl", "qwen35-9b-ud-q4kxl"),
        ("qwen3.5-9b-iq4_nl", "qwen35-9b-iq4nl"),
        ("qwen3.5-9b-q5_k_m", "qwen35-9b-q5km"),
        ("qwen3.5-2b-q5_k_m", "qwen35-2b-q5km"),
        ("qwen3.5-0.8b-q5_k_m", "qwen35-0.8b-q5km"),
        ("qwen3-4b-q5_k_m", "qwen3-4b-q5"),
        ("qwen3-8b-q4_k_m", "qwen3-8b-q4"),
        ("qwen3.6-35b-a3b-ud-q4_k_m", "qwen36-35b-a3b"),
    ]
    for needle, preset in mapping:
        if needle in name or needle in full:
            return preset
    return path.parent.name


def expand_catalog_path(value: str) -> Path:
    return Path(os.path.expandvars(value.replace("$HOME", str(Path.home())))).expanduser()


def load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"ERROR: model catalog not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"ERROR: failed to read model catalog {path}: {e}") from e


def catalog_models(catalog: dict[str, Any]) -> dict[str, Any]:
    models = catalog.get("models", {})
    if not isinstance(models, dict) or not models:
        raise SystemExit("ERROR: model catalog has no models")
    return models


def catalog_preset_for_path(catalog: dict[str, Any], path: Path) -> str:
    aliases = catalog.get("filename_aliases", {})
    if path.name in aliases:
        return str(aliases[path.name])
    low_name = path.name.lower()
    for preset, info in catalog_models(catalog).items():
        if str(info.get("file", "")).lower() == low_name:
            return preset
    for preset in catalog_models(catalog):
        if preset.lower() in str(path).lower():
            return preset
    return path.parent.name


def model_from_catalog(catalog: dict[str, Any], preset: str) -> ModelPreset:
    info = catalog_models(catalog).get(preset)
    if not info:
        raise SystemExit(f"ERROR: unknown model preset in catalog: {preset}")
    path = expand_catalog_path(str(info["dir"])) / str(info["file"])
    size_gb = path.stat().st_size / (1024 ** 3) if path.exists() else 0.0
    return ModelPreset(name=preset, path=path, size_gb=size_gb)


def apply_catalog_search_defaults(args: argparse.Namespace, catalog: dict[str, Any], model: ModelPreset) -> None:
    info = catalog_models(catalog).get(model.name, {})
    defaults = dict(info.get("defaults", {}))
    profile_name = args.hardware_profile
    profile = dict(catalog.get("hardware_profiles", {}).get(profile_name, {})) if profile_name else {}
    model_profile = dict(info.get("hardware_profiles", {}).get(profile_name, {})) if profile_name else {}
    merged_profile = {**profile, **model_profile}

    if args.max_ctx is None:
        args.max_ctx = int(merged_profile.get("max_ctx", defaults.get("ctx", 65536)))
    if args.ctx_choices == "catalog":
        args.ctx_choices = ",".join(str(x) for x in merged_profile.get("ctx_choices", [defaults.get("ctx", 32768)]))
    if args.ngl_choices == "catalog":
        args.ngl_choices = ",".join(str(x) for x in merged_profile.get("ngl_choices", [defaults.get("ngl", 999)]))
    if args.batch_choices == "catalog":
        args.batch_choices = ",".join(str(x) for x in merged_profile.get("batch_choices", [defaults.get("batch", 256)]))
    if args.ubatch_choices == "catalog":
        args.ubatch_choices = ",".join(str(x) for x in merged_profile.get("ubatch_choices", [defaults.get("ubatch", 128)]))


def discover_models(models_root: Path, catalog: dict[str, Any]) -> list[ModelPreset]:
    found: dict[Path, str] = {}
    for preset, info in catalog_models(catalog).items():
        path = expand_catalog_path(str(info["dir"])) / str(info["file"])
        if path.is_file():
            found[path.resolve()] = preset
    for path in models_root.glob("**/*.gguf"):
        if path.is_file():
            found.setdefault(path.resolve(), catalog_preset_for_path(catalog, path))
    return [
        ModelPreset(name=preset, path=path, size_gb=path.stat().st_size / (1024 ** 3))
        for path, preset in sorted(found.items(), key=lambda x: str(x[0]))
    ]


def choose_model(models: list[ModelPreset], requested: str | None, catalog: dict[str, Any]) -> ModelPreset:
    if requested:
        req = Path(requested).expanduser()
        if req.exists():
            return ModelPreset(catalog_preset_for_path(catalog, req), req.resolve(), req.stat().st_size / (1024 ** 3))
        if requested in catalog_models(catalog):
            return model_from_catalog(catalog, requested)
        for m in models:
            if requested in m.name or requested in m.path.name or requested in str(m.path):
                return m
        raise SystemExit(f"ERROR: requested model not found: {requested}")

    preferred = list(catalog_models(catalog).keys())
    for preset in preferred:
        for m in models:
            if m.name == preset:
                return m
    if not models:
        raise SystemExit("ERROR: no GGUF models found")
    return models[0]


def read_tail(path: Path, max_bytes: int = 8_000_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_overflow_requirement(text: str) -> int:
    required = 0
    for m in REQUEST_OVERFLOW_RE.finditer(text):
        required = max(required, int(m.group(1)))
    for m in PRECHECK_RE.finditer(text):
        required = max(required, int(m.group(1)) + int(m.group(3)))
    return required


def default_gateway_logs() -> list[Path]:
    root = Path("/tmp/openclaw-1000")
    return sorted(root.glob("openclaw-*.log")) if root.exists() else []


def parse_logs(paths: list[Path]) -> int:
    return max([0] + [parse_overflow_requirement(read_tail(p)) for p in paths])


def snapshot_logs(paths: list[Path]) -> dict[Path, int]:
    out = {}
    for p in paths:
        with contextlib.suppress(Exception):
            if p.exists():
                out[p] = p.stat().st_size
    return out


def parse_new_log_overflow(paths: list[Path], before: dict[Path, int]) -> int:
    required = 0
    for p in paths:
        if not p.exists():
            continue
        try:
            size = p.stat().st_size
            start = before.get(p, 0)
            with p.open("rb") as f:
                if start and size >= start:
                    f.seek(start)
                elif size > 4_000_000:
                    f.seek(size - 4_000_000)
                text = f.read().decode("utf-8", errors="replace")
            required = max(required, parse_overflow_requirement(text))
        except Exception:
            pass
    return required


def parse_int_choices(raw: str) -> list[int]:
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.add(int(part))
        except ValueError as e:
            raise SystemExit(f"ERROR: invalid integer in --predict-choices: {part!r}") from e
    if not values:
        raise SystemExit("ERROR: --predict-choices produced no values")
    return sorted(values)


def read_env_predict(path: Path) -> int:
    text = read_tail(path, 256_000)
    for line in text.splitlines():
        if line.startswith(("MAX_TOKENS=", "PREDICT=")):
            with contextlib.suppress(ValueError):
                return int(line.split("=", 1)[1].strip())
    return 0


def read_openclaw_predict(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    values: list[int] = []

    defaults = data.get("agents", {}).get("defaults", {})
    for key in ("maxTokens", "predict", "reasoningBudget"):
        val = defaults.get(key)
        if isinstance(val, int):
            values.append(val)

    providers = data.get("models", {}).get("providers", {})
    qwen = providers.get("qwen-local", {})
    for m in qwen.get("models", []) if isinstance(qwen.get("models", []), list) else []:
        if isinstance(m, dict):
            for key in ("maxTokens", "predict"):
                val = m.get(key)
                if isinstance(val, int):
                    values.append(val)

    return max(values) if values else 0


def normalize_predict_choices(args: argparse.Namespace) -> list[int]:
    choices = parse_int_choices(args.predict_choices)
    floor = args.min_predict

    if args.keep_existing_predict:
        existing = max(
            read_openclaw_predict(Path(args.openclaw_config).expanduser()),
            read_env_predict(Path(args.env_file).expanduser()),
        )
        if existing:
            floor = max(floor, existing)
            eprint(f"Existing PREDICT/maxTokens floor: {floor}")

    if floor:
        choices = sorted({x for x in choices if x >= floor} | {floor})

    return choices


def backup_file(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path.with_name(path.name + f".bak-{stamp}").write_bytes(path.read_bytes())


def write_env(path: Path, model: ModelPreset, host: str, port: int, c: Candidate, catalog: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_preset = model.name if model.name in catalog_models(catalog) else catalog_preset_for_path(catalog, model.path)
    model_info = catalog_models(catalog).get(model_preset, {})
    path.write_text(f"""# Generated by qwen-openclaw-optuna-tune.py
# Generated at {now_utc_iso()}
QWEN_MODEL_CATALOG={catalog.get("_path", "")}
MODEL_PRESET={model_preset}
MODEL_REPO={model_info.get("repo", "")}
MODEL_FILE={model.path.name}
MODEL_DIR={model.path.parent}
MODEL={model.path}
MODEL_SHA256=
LLAMA_BACKEND=auto
LLAMA_DIR=$HOME/src/llama.cpp
LLAMA_CPP_REF=
LLAMA_CPP_COMMIT=
SERVER=$HOME/.local/bin/llama-server
CLI=$HOME/.local/bin/llama-cli
HOST={host}
PORT={port}
CTX={c.ctx}
NGL={c.ngl}
MAX_TOKENS={c.predict}
PARALLEL={c.parallel}
BATCH={c.batch}
UBATCH={c.ubatch}
FLASH_ATTN={c.flash_attn}
REASONING_BUDGET={c.reasoning_budget}
JINJA=on
CHAT_TEMPLATE=chatml
TUNE_TIMEOUT=120
""", encoding="utf-8")


def patch_openclaw_config(
    path: Path,
    model: ModelPreset,
    c: Candidate,
    base_url: str,
    catalog: dict[str, Any],
    qwen_local: Path,
) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    defaults = data.setdefault("agents", {}).setdefault("defaults", {})
    defaults["contextTokens"] = c.ctx
    model_decl = defaults.setdefault("model", {})
    if isinstance(model_decl, dict):
        model_decl["primary"] = f"qwen-local/{model.path.name}"

    providers = data.setdefault("models", {}).setdefault("providers", {})
    qwen = providers.setdefault("qwen-local", {})
    qwen["baseUrl"] = base_url

    model_preset = model.name if model.name in catalog_models(catalog) else catalog_preset_for_path(catalog, model.path)
    local_service = qwen.setdefault("localService", {})

    # Keep OpenClaw's local service launcher valid. The tuner owns the final
    # OpenClaw provider patch, so it must not leave stale/bad localService
    # command/args/cwd values behind.
    qwen_local = qwen_local.expanduser().resolve()
    local_service["command"] = str(qwen_local)
    local_service["args"] = ["fast"]
    local_service["cwd"] = str(qwen_local.parent)

    env = local_service.setdefault("env", {})
    env.update({
        "MODEL_PRESET": model_preset,
        "QWEN_MODEL_CATALOG": str(catalog.get("_path", "")),
        "MODEL_FILE": model.path.name,
        "MODEL_DIR": str(model.path.parent),
        "MODEL": str(model.path),
        "HOST": base_url.removeprefix("http://").removesuffix("/v1").rsplit(":", 1)[0],
        "PORT": base_url.removesuffix("/v1").rsplit(":", 1)[-1],
        "CTX": str(c.ctx),
        "NGL": str(c.ngl),
        "PREDICT": str(c.predict),
        "PARALLEL": str(c.parallel),
        "BATCH": str(c.batch),
        "UBATCH": str(c.ubatch),
        "FLASH_ATTN": c.flash_attn,
        "REASONING_BUDGET": str(c.reasoning_budget),
    })

    models = qwen.setdefault("models", [{}])
    if not models:
        models.append({})
    m = models[0]
    m["id"] = model.path.name
    m["name"] = model.path.name
    m.pop("model", None)
    m["contextWindow"] = c.ctx
    m["contextTokens"] = c.ctx
    m["maxTokens"] = c.predict

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_server_cmd(llama_server: Path, model: ModelPreset, host: str, port: int, c: Candidate) -> list[str]:
    return [
        str(llama_server),
        "-m", str(model.path),
        "--host", host,
        "--port", str(port),
        "-c", str(c.ctx),
        "-ngl", str(c.ngl),
        "-np", str(c.parallel),
        "-b", str(c.batch),
        "-ub", str(c.ubatch),
        "-n", str(c.predict),
        "-fa", c.flash_attn,
        "--reasoning", "off",
    ]


def start_server(llama_server: Path, model: ModelPreset, host: str, port: int, c: Candidate, log_path: Path, timeout: int) -> subprocess.Popen[Any] | None:
    cmd = build_server_cmd(llama_server, model, host, port, c)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")
    eprint("  start:", quote_cmd(cmd))
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    if wait_for_port(host, port, timeout):
        return proc
    terminate_process(proc)
    with contextlib.suppress(Exception):
        log_f.close()
    return None


def direct_chat_smoke(host: str, port: int, model: ModelPreset, c: Candidate) -> tuple[bool, str, float]:
    payload = {
        "model": model.path.name,
        "messages": [
            {"role": "system", "content": "Reply with only OK."},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": min(c.predict, 256),
        "temperature": 0,
        "stream": False,
    }
    started = time.time()

    # llama-server can open the TCP port before the model is ready.
    # Treat 503 "Loading model" as a readiness wait, not a failed trial.
    status, data, raw = 0, None, ""
    deadline = started + 90
    while True:
        status, data, raw = http_json(f"http://{host}:{port}/v1/chat/completions", payload, 60)
        if status != 503 or "Loading model" not in str(raw):
            break
        if time.time() >= deadline:
            break
        time.sleep(1.0)

    elapsed = max(time.time() - started, 0.001)
    if status != 200:
        return False, f"direct chat failed status={status} body={raw[:500]}", elapsed
    text = ""
    with contextlib.suppress(Exception):
        msg = data["choices"][0]["message"]
        text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    if not text:
        return False, f"direct chat returned empty body={raw[:500]}", elapsed
    return True, text, elapsed


def speed_probe(host: str, port: int, model: ModelPreset, c: Candidate) -> tuple[float, float, float, str]:
    prompt = "Write a concise explanation of binary search.\n" * 80
    payload = {
        "model": model.path.name,
        "messages": [
            {"role": "system", "content": "You are a concise coding assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(c.predict, 256),
        "temperature": 0,
        "stream": False,
    }
    started = time.time()
    status, data, raw = http_json(f"http://{host}:{port}/v1/chat/completions", payload, 180)
    elapsed = max(time.time() - started, 0.001)
    if status != 200:
        return 0.0, 0.0, elapsed, f"speed probe failed status={status} body={raw[:500]}"
    out = ""
    with contextlib.suppress(Exception):
        msg = data["choices"][0]["message"]
        out = msg.get("content") or msg.get("reasoning_content") or ""
    prompt_tps = max(1, len(prompt) // 4) / elapsed
    gen_tps = max(1, len(out) // 4) / elapsed
    return prompt_tps, gen_tps, elapsed, ""


def run_openclaw_cmd(cmd: str, timeout: int) -> tuple[bool, str, float]:
    started = time.time()
    try:
        proc = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        return False, f"openclaw smoke timeout; output={out[-1500:]}", time.time() - started
    elapsed = time.time() - started
    if proc.returncode != 0:
        return False, f"openclaw smoke failed rc={proc.returncode}; output={(proc.stdout or '')[-1500:]}", elapsed
    return True, (proc.stdout or "")[-1500:], elapsed


def next_ctx_tiers(min_required: int, max_ctx: int) -> list[int]:
    tiers = [x for x in CTX_TIERS if x > min_required and x <= max_ctx]
    if tiers:
        return tiers
    rounded = ((min_required + 2047) // 2048) * 2048
    return [rounded] if rounded <= max_ctx else []


def suggest_candidate(
    trial: optuna.Trial,
    min_required_ctx: int,
    max_ctx: int,
    ctx_choices: list[int],
    predict_choices: list[int],
    ngl_choices: list[int],
    batch_choices: list[int],
    ubatch_choices: list[int],
    param_prefix: str,
) -> Candidate:
    """Return one candidate without Optuna dynamic categorical collisions.

    Optuna binds each parameter name to one exact CategoricalDistribution
    inside a study. Two fixes are used here:

    1. Prefix parameter names with a search-space fingerprint so old stored
       studies cannot collide when hardware/model/profile choices change.
    2. Use a dependent microbatch parameter name per batch value, e.g.
       ``ubatch_b128``. That allows the valid UBATCH list to depend on BATCH
       without reusing the same ``ubatch`` parameter name with different
       choices.
    """
    ctx = trial.suggest_categorical(f"{param_prefix}ctx", ctx_choices)
    if ctx <= min_required_ctx or ctx > max_ctx:
        raise optuna.TrialPruned(f"ctx {ctx} outside required range ({min_required_ctx}, {max_ctx}]")

    ngl = trial.suggest_categorical(f"{param_prefix}ngl", ngl_choices)
    batch = trial.suggest_categorical(f"{param_prefix}batch", batch_choices)

    valid_ubatch_choices = [x for x in ubatch_choices if x <= batch]
    if not valid_ubatch_choices:
        raise optuna.TrialPruned(f"no valid ubatch choices for batch {batch}")
    ubatch = trial.suggest_categorical(f"{param_prefix}ubatch_b{batch}", valid_ubatch_choices)

    predict = trial.suggest_categorical(f"{param_prefix}predict", predict_choices)
    return Candidate(
        ctx=ctx,
        ngl=ngl,
        predict=predict,
        parallel=1,
        batch=batch,
        ubatch=ubatch,
        flash_attn="auto",
        reasoning_budget=512,
        mode="fast",
    )


class Tuner:
    def __init__(self, args: argparse.Namespace, model: ModelPreset, min_required_ctx: int):
        self.args = args
        self.model = model
        self.min_required_ctx = min_required_ctx
        self.best_candidate: Candidate | None = None
        self.best_result: TrialResult | None = None
        self.work_dir = Path(args.work_dir).expanduser()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def objective(self, trial: optuna.Trial) -> float:
        c = suggest_candidate(
            trial,
            self.min_required_ctx,
            self.args.max_ctx,
            self.args.ctx_choices,
            self.args.predict_choices,
            self.args.ngl_choices,
            self.args.batch_choices,
            self.args.ubatch_choices,
            self.args.param_prefix,
        )
        eprint(f"\n[trial {trial.number}] {c}")
        base_url = f"http://{self.args.host}:{self.args.port}/v1"

        if self.args.patch_during_trials:
            write_env(Path(self.args.env_file).expanduser(), self.model, self.args.host, self.args.port, c, self.args.catalog_data)
            patch_openclaw_config(Path(self.args.openclaw_config).expanduser(), self.model, c, base_url, self.args.catalog_data, Path(self.args.qwen_local).expanduser())

        server_log = self.work_dir / f"trial-{trial.number}-ctx{c.ctx}-b{c.batch}-ub{c.ubatch}.llama-server.log"
        proc = start_server(Path(self.args.llama_server).expanduser(), self.model, self.args.host, self.args.port, c, server_log, self.args.startup_timeout)
        if proc is None:
            raise optuna.TrialPruned("llama-server startup failed")

        gateway_logs = [Path(x).expanduser() for x in self.args.gateway_log] or default_gateway_logs()
        before = snapshot_logs(gateway_logs)

        try:
            ok, msg, direct_elapsed = direct_chat_smoke(self.args.host, self.args.port, self.model, c)
            if not ok:
                raise optuna.TrialPruned(msg)

            prompt_tps, gen_tps, probe_elapsed, err = speed_probe(self.args.host, self.args.port, self.model, c)
            if err:
                raise optuna.TrialPruned(err)

            openclaw_elapsed = 0.0
            if self.args.openclaw_smoke_cmd:
                ok, out, openclaw_elapsed = run_openclaw_cmd(self.args.openclaw_smoke_cmd, self.args.openclaw_timeout)
                overflow = parse_new_log_overflow(gateway_logs, before)
                if overflow:
                    raise optuna.TrialPruned(f"OpenClaw overflow required_tokens={overflow}")
                if not ok:
                    raise optuna.TrialPruned(out)
            else:
                overflow = parse_new_log_overflow(gateway_logs, before)
                if overflow:
                    raise optuna.TrialPruned(f"gateway overflow required_tokens={overflow}")

            elapsed = direct_elapsed + probe_elapsed + openclaw_elapsed
            score = (
                gen_tps * 1000.0
                + prompt_tps * 20.0
                - (c.ctx / 1024.0) * self.args.ctx_penalty
                - (c.batch / 256.0)
                - (c.ubatch / 128.0)
                - elapsed * 2.0
            )
            result = TrialResult(True, score, prompt_tps=prompt_tps, gen_tps=gen_tps, elapsed_s=elapsed, server_log=str(server_log))
            trial.set_user_attr("candidate", dataclasses.asdict(c))
            trial.set_user_attr("result", dataclasses.asdict(result))
            if self.best_result is None or score > self.best_result.score:
                self.best_candidate = c
                self.best_result = result
            eprint(f"  PASS score={score:.2f} pp_est={prompt_tps:.2f} tg_est={gen_tps:.2f} elapsed={elapsed:.2f}s")
            return score
        finally:
            terminate_process(proc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuner for llama.cpp + OpenClaw")
    p.add_argument("--models-root", default=str(Path.home() / "models"))
    p.add_argument("--catalog", default=str(Path.home() / "models/qwen-model-presets.json"), help="Shared qwen model catalog JSON")
    p.add_argument("--qwen-local", default=str(Path.home() / "models/qwen-local.sh"), help="qwen-local launcher path to write into OpenClaw localService.command")
    p.add_argument("--hardware-profile", default=os.environ.get("QWEN_HARDWARE_PROFILE", "cuda-4070-8gb"), help="Catalog hardware profile for search spaces")
    p.add_argument("--model", default=None, help="Model path or substring")
    p.add_argument("--llama-server", required=True)
    # Compatibility with qwen-local.sh tune_openclaw_settings(), which may still pass this.
    # This Optuna tuner does not use llama-bench as the acceptance test.
    p.add_argument("--llama-bench", default="", help=argparse.SUPPRESS)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--openclaw-config", default=str(Path.home() / ".openclaw/openclaw.json"))
    p.add_argument("--env-file", default=str(Path.home() / ".config/qwen-local/qwen-local.env"))
    p.add_argument("--gateway-log", action="append", default=[])
    p.add_argument("--openclaw-smoke-cmd", default="", help="Optional real OpenClaw hello command")
    p.add_argument("--openclaw-timeout", type=int, default=180)
    p.add_argument("--n-trials", type=int, default=24)
    p.add_argument("--max-ctx", type=int, default=None)
    p.add_argument("--ctx-choices", default="catalog", help="Comma-separated fixed CTX candidates, or catalog")
    p.add_argument("--ctx-penalty", type=float, default=8.0)
    p.add_argument("--predict-choices", default="1024,2048,3072,4096", help="Comma-separated llama-server -n / OpenClaw maxTokens candidates")
    p.add_argument("--min-predict", type=int, default=0, help="Minimum PREDICT/maxTokens to preserve")
    p.add_argument("--keep-existing-predict", action=argparse.BooleanOptionalAction, default=True, help="Do not tune below existing env/config PREDICT/maxTokens")
    p.add_argument("--ngl-choices", default="catalog", help="Comma-separated -ngl candidates, or catalog. Lower values force CPU/RAM offload.")
    p.add_argument("--batch-choices", default="catalog", help="Comma-separated -b candidates, or catalog")
    p.add_argument("--ubatch-choices", default="catalog", help="Comma-separated -ub candidates, or catalog")
    p.add_argument("--startup-timeout", type=int, default=45)
    p.add_argument("--work-dir", default=str(Path.home() / ".cache/qwen-openclaw-optuna"))
    p.add_argument("--study-name", default="qwen-openclaw")
    p.add_argument("--storage", default="", help="Optional storage URI, e.g. sqlite:///study.db")
    p.add_argument("--min-required-ctx", type=int, default=0)
    p.add_argument("--safety-margin", type=int, default=512)
    p.add_argument("--patch-during-trials", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--write-final", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--backup-final", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()



def search_space_fingerprint(args: argparse.Namespace, model: ModelPreset) -> str:
    payload = {
        "model": str(model.path),
        "ctx": args.ctx_choices,
        "predict": args.predict_choices,
        "ngl": args.ngl_choices,
        "batch": args.batch_choices,
        "ubatch": args.ubatch_choices,
        "max_ctx": args.max_ctx,
        "min_required_ctx": args.min_required_ctx,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def maybe_namespace_study(args: argparse.Namespace, model: ModelPreset) -> None:
    # Stored Optuna studies cannot reuse a parameter name with a different
    # CategoricalDistribution. Namespace the default study name by search space
    # so changing hardware/profile/model choices does not collide with old runs.
    if args.storage and args.study_name == "qwen-openclaw":
        args.study_name = f"qwen-openclaw-{search_space_fingerprint(args, model)}"


def main() -> int:
    args = parse_args()
    catalog_path = Path(args.catalog).expanduser()
    catalog = load_catalog(catalog_path)
    catalog["_path"] = str(catalog_path)
    args.catalog_data = catalog
    llama_server = Path(args.llama_server).expanduser()
    if not llama_server.exists():
        eprint(f"ERROR: llama-server not found: {llama_server}")
        return 2

    model = choose_model(discover_models(Path(args.models_root).expanduser(), catalog), args.model, catalog)
    apply_catalog_search_defaults(args, catalog, model)
    args.predict_choices = normalize_predict_choices(args)
    args.ctx_choices = parse_int_choices(args.ctx_choices)
    args.ngl_choices = parse_int_choices(args.ngl_choices)
    args.batch_choices = parse_int_choices(args.batch_choices)
    args.ubatch_choices = parse_int_choices(args.ubatch_choices)

    eprint(f"Catalog: {catalog_path}")
    eprint(f"Hardware profile: {args.hardware_profile}")
    eprint(f"Model: {model.name}")
    eprint(f"Path:  {model.path}")
    eprint(f"Size:  {model.size_gb:.2f} GB")
    eprint(f"PREDICT search space: {args.predict_choices}")
    eprint(f"CTX fixed search space: {args.ctx_choices}")
    eprint(f"NGL search space: {args.ngl_choices}")
    eprint(f"BATCH search space: {args.batch_choices}")
    eprint(f"UBATCH search space: {args.ubatch_choices}")

    gateway_logs = [Path(x).expanduser() for x in args.gateway_log] or default_gateway_logs()
    parsed_required = parse_logs(gateway_logs)
    min_required = max(args.min_required_ctx, parsed_required)
    if min_required:
        min_required += args.safety_margin
        eprint(f"OpenClaw/log required tokens + margin: {min_required}")
    else:
        min_required = 17452 + args.safety_margin
        eprint(f"No overflow logs found; using conservative floor: {min_required}")

    valid_ctx_choices = [x for x in args.ctx_choices if x > min_required and x <= args.max_ctx]
    if not valid_ctx_choices:
        eprint(f"ERROR: no context tier above {min_required} and <= {args.max_ctx}")
        eprint(f"Fixed CTX choices: {args.ctx_choices}")
        return 2
    args.ctx_choices = valid_ctx_choices
    eprint(f"CTX valid choices this run: {args.ctx_choices}")

    # Prefix every Optuna parameter with a fingerprint. This prevents
    # CategoricalDistribution collisions when reusing an old sqlite study or
    # changing hardware/model/profile search spaces.
    args.param_prefix = f"p_{search_space_fingerprint(args, model)}_"
    maybe_namespace_study(args, model)
    eprint(f"Optuna parameter prefix: {args.param_prefix}")
    if args.storage:
        eprint(f"Optuna study name: {args.study_name}")

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4)
    if args.storage:
        study = optuna.create_study(study_name=args.study_name, storage=args.storage, direction="maximize", load_if_exists=True, sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    tuner = Tuner(args, model, min_required)
    try:
        study.optimize(tuner.objective, n_trials=args.n_trials, gc_after_trial=True)
    except KeyboardInterrupt:
        eprint("Interrupted; using best result so far if any.")

    if tuner.best_candidate is None or tuner.best_result is None:
        eprint("ERROR: no passing candidate found")
        return 1

    best = tuner.best_candidate
    print("\nBest candidate:")
    print(json.dumps(dataclasses.asdict(best), indent=2))
    print("\nBest result:")
    print(json.dumps(dataclasses.asdict(tuner.best_result), indent=2))

    if args.write_final:
        env_path = Path(args.env_file).expanduser()
        cfg_path = Path(args.openclaw_config).expanduser()
        if args.backup_final:
            backup_file(env_path)
            backup_file(cfg_path)
        write_env(env_path, model, args.host, args.port, best, args.catalog_data)
        patch_openclaw_config(cfg_path, model, best, f"http://{args.host}:{args.port}/v1", args.catalog_data, Path(args.qwen_local).expanduser())
        print(f"\nWrote env:   {env_path}")
        print(f"Patched cfg: {cfg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
