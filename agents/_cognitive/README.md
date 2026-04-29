# `agents/_cognitive/` — Shared Cognitive Primitives

Composable building blocks for agent memory and reasoning, sibling to `agents/_harness/`. The harness handles *how* the LLM is called; the cognitive layer handles *what the agent remembers and reasons about*.

## Layering

```
                ┌──────────────────────────────────────────────────────┐
Per-game        │ pokemon_red │ super_mario │ twenty_fourty_eight │ ...│  vertical:
agents          └──────┬──────┴──────┬──────┴──────┬──────────────┴────┘  per-game
                       ↓             ↓             ↓
                ┌──────────────────────────────────────────────────────┐
Cognitive       │              agents/_cognitive/                      │  horizontal:
primitives      │   MemoryProvider · VectorMemoryProvider · ...        │  shared
                └──────────────────────────────────────────────────────┘
                                       ↓
                ┌──────────────────────────────────────────────────────┐
Harness         │              agents/_harness/                        │  horizontal:
infra           │   prompt_caching · retry_utils · trajectory · ...    │  shared
                └──────────────────────────────────────────────────────┘
```

## Modules

### `memory_provider.py` — `MemoryProvider` (abstract base)

The interface every memory backend implements. Lifecycle:

| Method | When | Purpose |
|---|---|---|
| `is_available()` | agent setup | Provider is configured / ready (no network calls) |
| `initialize(session_id, **kw)` | session start | Connect, allocate, warm up |
| `prefetch(query)` | before each turn | Recall + format for prompt injection |
| `sync_turn(user, asst)` | after each turn | Persist the turn (optional) |
| `add_event(content, metadata)` | on significant events | Selective writes (badges, transitions, successes) |
| `on_session_end()` | episode end | Flush / summarize |
| `system_prompt_block()` | system prompt assembly | Inject usage notes for the model |
| `stats()` | observability | Memory count, retrieval hit rate, etc. |

### `vector_memory.py` — `VectorMemoryProvider`

Semantic recall via cosine similarity over text embeddings. Lifted from `agents/pokemon_red/openai_pokemon_vector_memory.py::VectorMemory` and reshaped to implement `MemoryProvider`.

* Default backend: `text-embedding-3-small` via the openai client (lazy-init, only when needed).
* Inject `embedding_fn=...` to use a different model — or a mock for tests.
* Sliding window of `max_memories` (default 100), drops oldest first.
* `add_event` deduplicates by similarity check before inserting (threshold 0.8).
* Pokemon's existing API surface (`add_memory`, `retrieve_similar`, `format_memories_for_prompt`) is preserved as compatibility shims, so refactoring an existing agent is one-line: `VectorMemory(...)` → `VectorMemoryProvider(...)`.

## Adding a new provider

Implement `MemoryProvider` and register in `__init__.py`. Keep the implementation focused — providers should not know about MACLA, game state, or specific agent classes. The agent decides what to store and what to retrieve; the provider decides how to embed, store, and search.

## What's intentionally NOT here yet

* Episodic memory (turn-by-turn replay buffer) — would be a separate provider
* Procedural memory (action sequence cache) — that's MACLA's domain
* Long-term cross-session storage — needs persistent backend; out of scope for this stage
* History summariser / reflector / subtask planner — separate cognitive modules, future stages
