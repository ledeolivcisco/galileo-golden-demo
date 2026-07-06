# Bank Assistant Load Generator

Drives the running Streamlit bank assistant with many concurrent, independent
browser sessions to generate realistic traffic and Galileo traces. Questions are
generated on the fly by **Ollama** (default `gemma4`) or optionally the
**Anthropic Claude API**.

Each browser context is a fresh Streamlit session, which the app maps to its own
`session_id` + per-session `GalileoLogger` — so every simulated user shows up as
a **separate session** in Galileo.

> This tool only interacts with the app through its UI. It does **not** modify
> the demo application.

## Scenarios

Each session is assigned one scenario and stays in-character for the whole chat:

| Scenario           | What it does |
|--------------------|--------------|
| `normal`           | Realistic banking questions (accounts, cards, balances, customer lookups like `C001`) that exercise tools + RAG |
| `rage`             | Angry/frustrated customer escalating over turns; tests tone handling |
| `prompt_injection` | Jailbreak attempts (ignore instructions, leak system prompt, misuse tools); tests Agent Control guardrails |
| `out_of_context`   | Off-topic questions (weather, recipes, code); tests scope adherence |

The scenario is printed in each session's log line, e.g. `[s004/rage]`, for easy
correlation.

## Prerequisites

- The demo app running at `http://localhost:8501` (bank domain at `/bank`).
- For `--provider ollama` (default): Ollama running with the chat model pulled (`gemma4`).
- For `--provider claude`: set `ANTHROPIC_API_KEY` in a `.env` file, or export it.

## Setup

```bash
cd demo/galileo-golden-demo/load_generator
pip install -r requirements.txt
playwright install chromium
```

For Claude, add your API key to a `.env` file (either here or in the project root):

```bash
# load_generator/.env  OR  ../.env
ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
# Default: Ollama — 6 concurrent sessions, 4-8 turns each, for 5 minutes
python loadgen.py

# Claude for question generation (faster, offloads Ollama for the bank assistant)
# ANTHROPIC_API_KEY is read from .env or the environment
python loadgen.py --provider claude --sessions 10 --concurrency 6

# Claude with a cheaper/faster model
python loadgen.py --provider claude --model claude-3-5-haiku-20241022 --sessions 20

# Fixed number of sessions instead of a duration
python loadgen.py --sessions 20 --concurrency 4

# Watch it happen (visible browsers)
python loadgen.py --sessions 3 --headed

# Only stress-test guardrails and off-topic handling
python loadgen.py --scenarios prompt_injection,out_of_context

# Weighted mix (mostly normal traffic with some abuse)
python loadgen.py --scenario-weights "normal=6,rage=2,prompt_injection=1,out_of_context=1"
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:8501/bank` | Target page (env: `LOADGEN_URL`) |
| `--concurrency` | `6` | Max concurrent sessions |
| `--messages-min` / `--messages-max` | `4` / `8` | Turns per session |
| `--duration` | `300` | Seconds to keep launching sessions (mutually exclusive with `--sessions`) |
| `--sessions` | – | Run exactly N sessions then stop |
| `--provider` | `ollama` | Question generation backend: `ollama` or `claude` (env: `LOADGEN_PROVIDER`) |
| `--ollama-url` | `http://localhost:11434` | Ollama endpoint (only when `--provider ollama`; env: `LOADGEN_OLLAMA_URL`) |
| `--model` | provider-specific | Ollama default: `gemma4`; Claude default: `claude-sonnet-4-20250514` |
| `--anthropic-api-key` | – | Anthropic API key (overrides `ANTHROPIC_API_KEY` from env or `.env`) |
| `--think-min` / `--think-max` | `1` / `4` | Random delay between turns (s) |
| `--headed` | off | Show browser windows |
| `--seed` | – | Seed RNG for reproducible mixes |
| `--scenarios` | `all` | Comma list of scenarios, or `all` |
| `--scenario-weights` | – | Weighted mix, e.g. `normal=6,rage=2,...` (overrides `--scenarios`) |
| `--response-timeout` | `180` | Max seconds to wait for one assistant reply |

## Docker

Run the load generator in a container against the Streamlit app on your host.
URL and API key are parameterized via environment variables (not baked into the image).

```bash
cd demo/galileo-golden-demo/load_generator
cp .env.example .env   # set LOADGEN_URL and ANTHROPIC_API_KEY
docker compose build
docker compose run --rm load-generator
```

### Environment variables

| Variable | Default (Docker) | Description |
|----------|------------------|-------------|
| `LOADGEN_URL` | `http://host.docker.internal:8501/bank` | Target Streamlit page |
| `ANTHROPIC_API_KEY` | (required for Claude) | Anthropic API key |
| `LOADGEN_PROVIDER` | `claude` | Question backend: `ollama` or `claude` |
| `LOADGEN_OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama endpoint when using `ollama` provider |

Override without editing `.env`:

```bash
LOADGEN_URL=http://host.docker.internal:8501/bank \
ANTHROPIC_API_KEY=sk-ant-... \
docker compose run --rm load-generator
```

Override run arguments (appended after entrypoint):

```bash
docker compose run --rm load-generator --sessions 5 --concurrency 2 --scenarios rage
```

Notes:
- `host.docker.internal` reaches services on the host (Mac/Windows; Linux uses `extra_hosts` in compose).
- Claude is recommended in Docker — it avoids routing question generation through host Ollama.
- The container exits when the load run finishes (one-shot).

## Notes

- With **Ollama**, the local instance serves both the app's agent and question
  generation, so real throughput is bound by Ollama. Keep concurrency moderate.
- With **Claude**, question generation no longer competes with Ollama — the bank
  assistant's response time becomes the main bottleneck, so you can run higher
  concurrency.
- If the chosen provider is slow or unavailable, the generator falls back to
  curated per-scenario questions so the run never stalls.
