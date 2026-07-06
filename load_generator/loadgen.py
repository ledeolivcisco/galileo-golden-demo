#!/usr/bin/env python3
"""
Load generator for the Galileo golden-demo bank assistant.

Spawns multiple concurrent browser sessions against the running Streamlit app
(default http://localhost:8501/bank) and drives multi-turn chats. Each browser
context is an independent Streamlit session, which the app maps to its own
`session_id` + per-session GalileoLogger, so Galileo records separate
sessions/traces per simulated user.

Questions are generated on the fly by Ollama (default gemma4) or optionally
the Anthropic Claude API. Each session is assigned a *scenario* (normal / rage /
prompt_injection / out_of_context) that shapes the whole conversation.

The demo application is NOT modified; this only interacts through the UI.

Usage:
    pip install -r requirements.txt
    playwright install chromium
    python loadgen.py --duration 300 --concurrency 6
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from playwright.async_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from question_generator import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OLLAMA_MODEL,
    PROVIDERS,
    SCENARIOS,
    QuestionGenerator,
    pick_scenario,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env from load_generator/ and project root (does not override existing env)."""
    here = Path(__file__).resolve().parent
    load_dotenv(here / ".env")
    load_dotenv(here.parent / ".env")


@dataclass
class Config:
    url: str = "http://localhost:8501/bank"
    concurrency: int = 6
    messages_min: int = 4
    messages_max: int = 8
    duration: Optional[float] = 300.0  # seconds; None if --sessions used
    sessions: Optional[int] = None
    ollama_url: str = "http://localhost:11434"
    provider: str = "ollama"
    model: str = DEFAULT_OLLAMA_MODEL
    anthropic_api_key: Optional[str] = None
    think_min: float = 1.0
    think_max: float = 4.0
    headless: bool = True
    seed: Optional[int] = None
    scenarios: List[str] = field(default_factory=lambda: list(SCENARIOS))
    scenario_weights: Optional[Dict[str, float]] = None
    response_timeout: float = 180.0  # per-turn max wait for assistant reply
    nav_timeout: float = 60.0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    sessions_started: int = 0
    sessions_completed: int = 0
    sessions_failed: int = 0
    turns_ok: int = 0
    turns_failed: int = 0
    latencies: List[float] = field(default_factory=list)
    by_scenario: Dict[str, int] = field(default_factory=dict)

    def record_scenario(self, scenario: str) -> None:
        self.by_scenario[scenario] = self.by_scenario.get(scenario, 0) + 1

    def summary(self) -> str:
        avg = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        lines = [
            "",
            "=" * 60,
            "LOAD RUN SUMMARY",
            "=" * 60,
            f"Sessions started   : {self.sessions_started}",
            f"Sessions completed : {self.sessions_completed}",
            f"Sessions failed    : {self.sessions_failed}",
            f"Turns OK / failed  : {self.turns_ok} / {self.turns_failed}",
            f"Avg response time  : {avg:.1f}s",
            f"Scenarios          : "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.by_scenario.items())),
            "=" * 60,
        ]
        return "\n".join(lines)


# Selectors for Streamlit's chat input and messages.
_CHAT_INPUT = "[data-testid='stChatInput'] textarea, textarea[data-testid='stChatInputTextArea']"
_CHAT_MESSAGE = "[data-testid='stChatMessage']"


# ---------------------------------------------------------------------------
# Single session
# ---------------------------------------------------------------------------

async def _wait_for_response(page: Page, prev_count: int, timeout: float) -> None:
    """Wait until a new assistant message has rendered and 'Thinking...' is gone."""
    deadline = time.monotonic() + timeout
    # First, wait for a new chat message bubble to appear (user + assistant both
    # render as stChatMessage; sending adds user then assistant, so wait for at
    # least prev_count + 2, but tolerate prev_count + 1 if layout differs).
    while time.monotonic() < deadline:
        count = await page.locator(_CHAT_MESSAGE).count()
        thinking = await page.get_by_text("Thinking...", exact=False).count()
        if count >= prev_count + 2 and thinking == 0:
            return
        await asyncio.sleep(0.5)
    raise PlaywrightTimeoutError("Timed out waiting for assistant response")


async def run_session(
    session_index: int,
    browser,
    cfg: Config,
    qgen: QuestionGenerator,
    stats: Stats,
    rng: random.Random,
) -> None:
    scenario = pick_scenario(cfg.scenario_weights, rng) if cfg.scenario_weights \
        else rng.choice(cfg.scenarios)
    n_turns = rng.randint(cfg.messages_min, cfg.messages_max)
    stats.sessions_started += 1
    stats.record_scenario(scenario)
    tag = f"[s{session_index:03d}/{scenario}]"
    print(f"{tag} start ({n_turns} turns)")

    context = None
    history: List[Dict[str, str]] = []
    try:
        context = await browser.new_context()
        page = await context.new_page()
        page.set_default_timeout(cfg.nav_timeout * 1000)
        await page.goto(cfg.url, wait_until="domcontentloaded",
                        timeout=cfg.nav_timeout * 1000)
        await page.wait_for_selector(_CHAT_INPUT, timeout=cfg.nav_timeout * 1000)

        for turn in range(n_turns):
            try:
                question = await qgen.generate(scenario, history, turn)
            except Exception as e:
                print(f"{tag} turn {turn}: qgen error: {e}")
                continue

            prev_count = await page.locator(_CHAT_MESSAGE).count()
            box = page.locator(_CHAT_INPUT).first
            await box.click()
            await box.fill(question)
            await box.press("Enter")
            history.append({"role": "user", "content": question})
            print(f"{tag} turn {turn}: {question[:80]}")

            t0 = time.monotonic()
            try:
                await _wait_for_response(page, prev_count, cfg.response_timeout)
                latency = time.monotonic() - t0
                stats.latencies.append(latency)
                stats.turns_ok += 1
                # Capture the last assistant bubble text for conversation context.
                try:
                    answer = await page.locator(_CHAT_MESSAGE).last.inner_text()
                    history.append({"role": "assistant", "content": answer.strip()})
                except Exception:
                    pass
                print(f"{tag} turn {turn}: responded in {latency:.1f}s")
            except PlaywrightTimeoutError:
                stats.turns_failed += 1
                print(f"{tag} turn {turn}: response TIMEOUT")

            # think time between turns
            await asyncio.sleep(rng.uniform(cfg.think_min, cfg.think_max))

        stats.sessions_completed += 1
        print(f"{tag} done")
    except Exception as e:
        stats.sessions_failed += 1
        print(f"{tag} FAILED: {e}")
    finally:
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run(cfg: Config) -> Stats:
    rng = random.Random(cfg.seed)
    stats = Stats()
    qgen = QuestionGenerator(
        provider=cfg.provider,
        ollama_url=cfg.ollama_url,
        model=cfg.model,
        anthropic_api_key=cfg.anthropic_api_key,
        rng=random.Random(cfg.seed + 1 if cfg.seed is not None else None),
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=cfg.headless)
        try:
            sem = asyncio.Semaphore(cfg.concurrency)
            tasks: set = set()
            session_index = 0
            start = time.monotonic()

            def should_continue() -> bool:
                if cfg.sessions is not None:
                    return session_index < cfg.sessions
                return (time.monotonic() - start) < (cfg.duration or 0)

            async def guarded(idx: int) -> None:
                async with sem:
                    await run_session(idx, browser, cfg, qgen, stats, rng)

            while should_continue():
                if len(tasks) >= cfg.concurrency:
                    done, tasks = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                session_index += 1
                tasks.add(asyncio.create_task(guarded(session_index)))
                await asyncio.sleep(0.05)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await browser.close()

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_weights(spec: Optional[str]) -> Optional[Dict[str, float]]:
    if not spec:
        return None
    weights: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"Invalid weight '{part}'. Use name=number,name=number"
            )
        name, val = part.split("=", 1)
        name = name.strip()
        if name not in SCENARIOS:
            raise argparse.ArgumentTypeError(
                f"Unknown scenario '{name}'. Valid: {', '.join(SCENARIOS)}"
            )
        weights[name] = float(val)
    return weights or None


def _parse_scenarios(spec: str) -> List[str]:
    if spec.strip().lower() == "all":
        return list(SCENARIOS)
    result = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in SCENARIOS:
            raise argparse.ArgumentTypeError(
                f"Unknown scenario '{part}'. Valid: {', '.join(SCENARIOS)}, all"
            )
        result.append(part)
    return result or list(SCENARIOS)


def _env_provider_default() -> str:
    val = os.environ.get("LOADGEN_PROVIDER", "ollama").strip().lower()
    return val if val in PROVIDERS else "ollama"


def parse_args(argv: Optional[List[str]] = None) -> Config:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Bank assistant load generator")
    ap.add_argument(
        "--url",
        default=os.environ.get("LOADGEN_URL", "http://localhost:8501/bank"),
        help="Target Streamlit page (env: LOADGEN_URL)",
    )
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--messages-min", type=int, default=4)
    ap.add_argument("--messages-max", type=int, default=8)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--duration", type=float, default=300.0,
                     help="Run for N seconds, launching sessions continuously")
    grp.add_argument("--sessions", type=int, default=None,
                     help="Run exactly N sessions then stop")
    ap.add_argument(
        "--ollama-url",
        default=os.environ.get("LOADGEN_OLLAMA_URL", "http://localhost:11434"),
        help="Ollama endpoint when --provider ollama (env: LOADGEN_OLLAMA_URL)",
    )
    ap.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=_env_provider_default(),
        help="LLM provider for question generation (env: LOADGEN_PROVIDER)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help=(
            "Model name (default: gemma4 for ollama, "
            f"{DEFAULT_CLAUDE_MODEL} for claude)"
        ),
    )
    ap.add_argument(
        "--anthropic-api-key",
        default=None,
        help="Anthropic API key (overrides ANTHROPIC_API_KEY from env or .env)",
    )
    ap.add_argument("--think-min", type=float, default=1.0)
    ap.add_argument("--think-max", type=float, default=4.0)
    ap.add_argument("--headed", action="store_true",
                    help="Show browser windows (default: headless)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scenarios", type=_parse_scenarios, default="all",
                    help="Comma list of scenarios or 'all'")
    ap.add_argument("--scenario-weights", type=_parse_weights, default=None,
                    help="Weighted mix, e.g. normal=6,rage=2,prompt_injection=1,out_of_context=1")
    ap.add_argument("--response-timeout", type=float, default=180.0)
    args = ap.parse_args(argv)

    provider = args.provider
    if args.model is None:
        model = DEFAULT_CLAUDE_MODEL if provider == "claude" else DEFAULT_OLLAMA_MODEL
    else:
        model = args.model

    anthropic_api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if provider == "claude" and not anthropic_api_key:
        ap.error(
            "ANTHROPIC_API_KEY is required when --provider claude. Set it in a "
            ".env file (load_generator/.env or project root .env), export it, "
            "or pass --anthropic-api-key"
        )

    duration = None if args.sessions is not None else args.duration
    scenarios = args.scenarios if isinstance(args.scenarios, list) else list(SCENARIOS)

    return Config(
        url=args.url,
        concurrency=args.concurrency,
        messages_min=args.messages_min,
        messages_max=args.messages_max,
        duration=duration,
        sessions=args.sessions,
        ollama_url=args.ollama_url,
        provider=provider,
        model=model,
        anthropic_api_key=anthropic_api_key,
        think_min=args.think_min,
        think_max=args.think_max,
        headless=not args.headed,
        seed=args.seed,
        scenarios=scenarios,
        scenario_weights=args.scenario_weights,
        response_timeout=args.response_timeout,
    )


def main() -> int:
    cfg = parse_args()
    print("Bank assistant load generator")
    print(f"  target      : {cfg.url}")
    print(f"  concurrency : {cfg.concurrency}")
    print(f"  provider    : {cfg.provider}")
    if cfg.provider == "claude":
        print(f"  model       : {cfg.model}")
    else:
        print(f"  model       : {cfg.model} @ {cfg.ollama_url}")
    if cfg.sessions is not None:
        print(f"  sessions    : {cfg.sessions}")
    else:
        print(f"  duration    : {cfg.duration}s")
    if cfg.scenario_weights:
        print(f"  weights     : {cfg.scenario_weights}")
    else:
        print(f"  scenarios   : {', '.join(cfg.scenarios)}")
    print()

    try:
        stats = asyncio.run(run(cfg))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    print(Stats.summary(stats) if isinstance(stats, Stats) else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
