# Qwen Local with llama.cpp, Codex, and OpenClaw

`qwen-local.sh` installs and runs a local Qwen GGUF through `llama-server`. It
supports CUDA, Apple Metal, and CPU builds, and can configure Codex or OpenClaw
to use the local OpenAI-compatible endpoint.

## Quick start

Create the persistent runtime configuration:

```bash
mkdir -p ~/.config/qwen-local
cp qwen-local.env.example ~/.config/qwen-local/qwen-local.env
```

Edit `~/.config/qwen-local/qwen-local.env` for the model and hardware, then run:

```bash
./qwen-local.sh install
./qwen-local.sh fast
```

`install` installs dependencies, builds llama.cpp, and downloads the selected
model. On macOS it uses Xcode Command Line Tools; the full Xcode application is
not required. The server runs in the foreground so its logs and failures remain
visible.

Use the reasoning mode when you want the model's reasoning channel enabled:

```bash
./qwen-local.sh reasoning
```

The two start commands only change reasoning on/off. Model, context, batching,
template behavior, and other server settings come from the env file.

## Runtime configuration

The default env file is:

```text
~/.config/qwen-local/qwen-local.env
```

Use a different file when needed:

```bash
QWEN_LOCAL_ENV_FILE=/path/to/qwen-local.env ./qwen-local.sh fast
```

Env entries must use plain `NAME=value` syntax without shell quotes or spaces.
The main runtime settings are:

```dotenv
MODEL=/path/to/model.gguf
HOST=127.0.0.1
PORT=18080
CTX=32768
NGL=999
MAX_TOKENS=4096
PARALLEL=1
BATCH=256
UBATCH=128
FLASH_ATTN=auto
REASONING_BUDGET=512
REASONING_FORMAT=deepseek
JINJA=on
OUTPUT_MODE=codex
CUSTOM_CHAT_TEMPLATE=
```

Restart `qwen-local.sh fast` or `qwen-local.sh reasoning` after changing the
env file. Output-template selection is server-wide, not per request.

## Output modes

`OUTPUT_MODE` selects how llama.cpp formats messages and reasoning:

| Mode | Behavior | Use case |
| --- | --- | --- |
| `codex` | Uses `templates/qwen3.5-codex.jinja`. Reasoning is returned separately from final text, and Codex message ordering is supported. | Codex CLI with `/v1/responses` |
| `native` | Passes no template override and uses the template embedded in the GGUF. | Other clients that expect the model's original format |
| `custom` | Uses the file set by `CUSTOM_CHAT_TEMPLATE`. | Client-specific or experimental output formats |

For Codex:

```dotenv
OUTPUT_MODE=codex
CUSTOM_CHAT_TEMPLATE=
```

For the original model output:

```dotenv
OUTPUT_MODE=native
CUSTOM_CHAT_TEMPLATE=
```

The native Qwen3.5 template can reject Codex parser-generation probes with
`System message must be at the beginning`; use `OUTPUT_MODE=codex` for Codex.

### Custom output template

Start from the bundled Codex-compatible template or create a Jinja template
for the target client:

```bash
cp templates/qwen3.5-codex.jinja templates/my-client.jinja
```

Then select it in the env file:

```dotenv
OUTPUT_MODE=custom
CUSTOM_CHAT_TEMPLATE=templates/my-client.jinja
```

Relative paths are resolved from the directory containing `qwen-local.sh`.
Absolute paths are also accepted. The script validates that the file exists
before stopping the current server.

Custom templates receive llama.cpp chat-template variables such as `messages`,
`tools`, `add_generation_prompt`, and `enable_thinking`. For Codex-compatible
reasoning, generated `<think>...</think>` text must be parsed into a Responses
API `reasoning` item while the final answer remains in `output_text`. Use the
bundled template as the working example.

After restarting, verify the selected mode in the startup output:

```text
OUTPUT_MODE=custom
CHAT_TEMPLATE=/absolute/path/to/templates/my-client.jinja
```

## Codex profile

Start the server with `OUTPUT_MODE=codex`, then create or refresh the profile:

```bash
./qwen-local.sh codex-profile
codex --profile local-llama
```

`codex-profile` verifies `/v1/responses` support and writes:

```text
~/.codex/local-llama.config.toml
~/.codex/model-catalogs/local-llama.json
```

The generated model catalog uses the active model ID and context size, avoiding
Codex's fallback model-metadata warning. The generated provider has its own name
so settings from another `qwen-local` provider do not leak into it.

Non-interactive verification:

```bash
codex exec --profile local-llama --skip-git-repo-check \
  'Reply with exactly: OK'
```

## Endpoint tests

Run the built-in Chat Completions test:

```bash
./qwen-local.sh test
```

With `OUTPUT_MODE=codex`, the response should have clean final content and a
separate `reasoning_content` field instead of `<think>` tags inside `content`.

Test the Responses endpoint directly:

```bash
curl -sS http://127.0.0.1:18080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-4B-Q5_K_M.gguf",
    "input": "Reply with exactly OK",
    "stream": false,
    "max_output_tokens": 768
  }' | jq .
```

Codex mode should return one `reasoning` output item and a separate `message`
whose `output_text` is `OK`.

## OpenClaw

Review the generated OpenClaw settings before applying them:

```bash
./qwen-local.sh openclaw-review
```

Install or update OpenClaw and apply the local provider settings:

```bash
./qwen-local.sh openclaw-install
```

Additional diagnostics and tuning commands:

```bash
./qwen-local.sh diagnose-openclaw
./qwen-local.sh tune-openclaw
./qwen-local.sh tune
```

Choose `OUTPUT_MODE=native` unless an OpenClaw integration specifically needs
the Codex-compatible Responses reasoning layout.

## Management

```bash
./qwen-local.sh status
./qwen-local.sh stop
./qwen-local.sh update
```

Server logs are written under `~/llm-services/logs/`.

## Troubleshooting

### `<think>` appears in final content

Set `OUTPUT_MODE=codex`, restart the server, and confirm startup prints the
bundled `qwen3.5-codex.jinja` path. `REASONING_FORMAT=deepseek` alone is not
enough when a generic ChatML template is used.

### `System message must be at the beginning`

The native model template is being used with Codex. Change to:

```dotenv
OUTPUT_MODE=codex
```

Then restart the server.

### Custom template parser error

llama.cpp automatically derives its request/response parser from the template.
An exception raised for one of its synthetic probe messages prevents server
requests from running. Remove overly strict role-order checks or compare the
template with `templates/qwen3.5-codex.jinja`.

### Out of memory

Reduce `CTX`, `PARALLEL`, `BATCH`, or `UBATCH` in the env file. Context and
parallel slots have the largest effect on KV-cache memory.
