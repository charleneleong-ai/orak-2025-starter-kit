# `agents/_harness/` — Shared Agent Harness

Generic infrastructure shared across every agent in this repo, **independent of cognitive architecture**.

## Layering

```
                    ┌─────────────────────────────────────────────┐
   Cognitive arch   │ MACLA   │ Vanilla │ Vector-memory  │ ...    │  vertical:
                    │ (proc.) │ (1-call)│  (5-module)    │        │  per-agent
                    └────┬────┴────┬────┴───────┬────────┴────────┘
                         │         │            │
                         ↓         ↓            ↓
                    ┌─────────────────────────────────────────────┐
   Harness          │              agents/_harness/                │  thin re-export
                    │   ┌───────────────────────────────────────┐  │  layer over
                    │   │  autoresearch.{trajectory, retry_utils,│  │  shared infra
                    │   │  prompt_caching}  (lifted upstream)    │  │
                    │   └───────────────────────────────────────┘  │
                    │   structured_invoke (langchain-coupled,      │
                    │   stays local)                               │
                    └────┬────────────────────────────────────────┘
                         ↓
                    ┌─────────────────────────────────────────────┐
   LLM client       │  langchain ChatOpenAI / ChatVertexAI / ...   │
                    └─────────────────────────────────────────────┘
```

The harness has **zero knowledge** of MACLA, procedure memory, sub-goal decomposition, vector retrieval, game adapters, or any cognitive concept. It only deals with: how the LLM call is made, what gets recorded, and how the prefix is cached.

## Where the code lives

| Symbol | Source |
|---|---|
| `TrajectoryWriter`, `StepRecord`, `convert_scratchpad_to_think`, `has_incomplete_scratchpad` | `autoresearch.trajectory` (lifted v0.17.0, autoresearch#32) |
| `extract_cache_stats` | `autoresearch.prompt_caching` (lifted v0.17.0, autoresearch#34) |
| `with_retries`, `classify`, `jittered_backoff`, `ClassifiedError`, `ErrorClass` | `autoresearch.retry_utils` (lifted v0.17.0, autoresearch#34) |
| `structured_invoke_with_usage` | `agents/_harness/structured_invoke.py` (still local — langchain-coupled) |

`agents/_harness/__init__.py` re-exports all of the above so call-sites don't need to know which symbol came from which package — `from agents._harness import TrajectoryWriter` keeps working.

## What each module does

### Trajectory (`autoresearch.trajectory`)

* `StepRecord` — dataclass for a single step (system/user/assistant turns + tokens + fallback flag).
* `TrajectoryWriter` — buffers `StepRecord`s, flushes per-episode to either:
  * `trajectory_samples.jsonl` — episodes with no fallbacks (clean for SFT)
  * `failed_trajectories.jsonl` — episodes with any fallback or crash
* Each step is rolled into a ShareGPT-shaped `conversations` array (`[{from: system|human|gpt, value: ...}]`) — directly consumable by SFT pipelines without reformatting.
* `convert_scratchpad_to_think` — replaces `<REASONING_SCRATCHPAD>` with `<think>` for thinking-mode dataset compatibility.

The legacy `raw_requests.jsonl` per-step log keeps working alongside this — `TrajectoryWriter` only emits the new rolled-up files.

### Prompt caching (`autoresearch.prompt_caching`)

`extract_cache_stats(usage)` pulls `cached_tokens` from a usage object regardless of shape:

* vLLM / OpenAI ChatCompletions `CompletionUsage` — `prompt_tokens_details.cached_tokens`
* OpenAI Responses `ResponseUsage` — `input_tokens_details.cached_tokens`
* Plain `dict` (some custom adapters)

Surfaces in `log_extras["tokens_cached"]` automatically via `BaseOrakAgent.get_action`.

### Retry (`autoresearch.retry_utils`)

* `jittered_backoff(attempt, base_delay=5.0, max_delay=120.0, jitter_ratio=0.5)` — decorrelated exponential backoff. Counter-seeded jitter prevents thundering-herd retries when autoresearch workers hit the same vLLM server.
* `classify(error) -> ClassifiedError` — categorises by HTTP status / message into `TRANSIENT` (retry), `TERMINAL` (raise), `UNKNOWN` (retry but log loudly).
* `with_retries(fn, max_attempts=3, label=...)` — runs `fn` with classified retries. The original exception is raised on terminal errors or exhaustion, with a `__classified__` attribute attached for cheap introspection.

### Structured invoke (`agents/_harness/structured_invoke.py`)

`structured_invoke_with_usage(llm, messages, output_schema)` wraps langchain's structured-output API to preserve `usage_metadata` so `extract_cache_stats` has something to read. Stays local because it's langchain-coupled and not yet on autoresearch's surface.

## Wiring status

Every `BaseOrakAgent` subclass automatically picks up:
* `extract_cache_stats` (called inside `get_action()`)
* `TrajectoryWriter` (set up in `set_log_dir()`, fed in `act()`, flushed in `record_episode_end()`)

Subclasses with try/except → fallback patterns around `structured_llm.invoke(messages)` have been wrapped with `with_retries()` and call `self._mark_fallback(reason)` so the trajectory writer can split successful episodes from failed ones:

| File | Status |
|---|---|
| `agents/super_mario/base.py` | wired (`super_mario.llm`) |
| `agents/twenty_fourty_eight/base.py` | wired (`twenty_fourty_eight.llm`) |
| `agents/starcraft/base.py` | wired (`starcraft.llm`) |
| `agents/pokemon_red/base.py` | wired (`pokemon_red.llm`) |
| `agents/macla/unified.py` | wired (`macla_unified.llm`) |
| `agents/pokemon_red/openai_pokemon_vector_memory.py` | **NOT WIRED** — bypasses `BaseOrakAgent`. Follow-up. |
| `agents/*/random_*.py` | N/A — no LLM calls |

## Adding a new agent

If your agent inherits from `BaseOrakAgent` (directly or transitively via `BaseOpenAIAgent` / `BaseGeminiAgent` / `BaseMaclaAgent`):

1. **Trajectory + cache stats**: free. Set `self.set_log_dir(log_dir)` and you're done.
2. **Retry**: wrap your LLM call site:
   ```python
   from agents._harness import with_retries
   try:
       response = with_retries(lambda: structured_llm.invoke(messages), label="my_game.llm")
       # ...success path
   except Exception as e:
       self._mark_fallback(f"llm_error: {type(e).__name__}: {str(e)[:200]}")
       # ...fallback path (existing behaviour)
   ```
3. **Prefix caching**: nothing to do — vLLM caches automatically. Just check `cached_tokens` is non-zero in your stats after a few turns.

If your agent does NOT inherit `BaseOrakAgent` (e.g. `OpenAIPokemonVectorMemoryAgent`), import from `autoresearch` (or `agents._harness`) directly — but trajectory/cache wiring needs to be done manually.

## Testing

* **Primitive unit tests** (`StepRecord.to_sharegpt`, `TrajectoryWriter.flush_episode`, `extract_cache_stats` shapes, `with_retries` retry/raise paths) live upstream in `autoresearch/tests/test_{trajectory,prompt_caching,retry_utils}.py` — not duplicated here.
* **Integration tests** (`tests/test_harness_integration.py`) cover the orak-specific wiring: `BaseOrakAgent.set_log_dir` → `TrajectoryWriter` instantiation, `_mark_fallback` → routing to `failed_trajectories.jsonl`, `cached_tokens` flow from usage object → `log_extras` → `StepRecord`. These are the only harness tests that need to live in this repo.

## What's intentionally NOT here

* Heavy error-classification machinery — replaced by an inline ~80-LOC classifier; expand only when a real failure mode demands it.
* Context compression / window management — not needed at 200-step game lengths.
* Memory provider abstractions — that's a cognitive-architecture concern, not a harness concern. Lives in `agents/_cognitive/`.
* Provider-agnostic LLM transports — orak uses langchain adapters; replacing that is a separate concern out of scope here.
