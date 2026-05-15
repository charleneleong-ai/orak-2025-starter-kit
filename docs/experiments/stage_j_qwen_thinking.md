# Stage J — Qwen 3 Thinking-Mode (Extended Reasoning at Decision Time)

**Status:** scaffolded, queued after Stage H iter 3  •  **Branch:** `feat/stage-k-qwen-thinking`  •  **Sequenced before:** Stage K

## Hypothesis

Stages A→H converged at **57.14%** (4/7 milestones) on pokemon across model lineages:
- 6 Gemma variants (Stage A→G) with action/procedure/reflection layer interventions
- Qwen 3.5 35B-A3B-Int4 (Stage H, non-thinking variant): 57.14% × n=3, σ=0

Trajectory introspection shows the agent diagnoses stuck-state correctly at step 149 (*"You are stuck in a movement loop... no score gain"*) but the action layer doesn't change strategy. The diagnosis points at **LLM reasoning at the milestone boundary** as the bottleneck.

Stage J tests one specific intervention: **explicit thinking-mode reasoning budget before each tool-call**.

[`cyankiwi/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit`](https://hf.co/cyankiwi/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit) is a Qwen-trained always-thinking variant: it emits `<think>...</think>` chain-of-thought before every response. vLLM's `--reasoning-parser qwen3` strips these tags server-side and returns the reasoning in a separate `reasoning_content` field — so the agent harness sees the same clean tool-call output it always has. **The only variable changing is the LLM's reasoning budget per decision.**

## Direct comparison to Stage H

|  | Stage H | Stage J (this PR) |
|---|---|---|
| Model | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | `cyankiwi/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit` |
| Thinking mode | ❌ off | ✅ always-on |
| Total params | 35B (MoE) | 30B (MoE) |
| Active params | 3B | 3B |
| Quant | GPTQ-Int4 | AWQ-Int4 |
| Tool-call parser | `hermes` | `hermes` |
| Reasoning parser | none | `qwen3` (strips `<think>` blocks) |
| Agent stack | Stage D (vmem + planner + procedures + self-reflection) | identical |
| Pokemon n=3 result | 57.14% × n=3, σ=0 | _pending_ |

Same active-param count (3B) controls compute; ~30/35B total params controls total knowledge; same Int4 quant family; same MACLA stack. **The thinking-mode toggle is the only meaningful variable.**

## Mechanism — how thinking-mode integrates

```
LLM raw output:
  <think>
  The user wants me to navigate. I'm at (5, 11). Last 3 actions were all
  warp_with_warp_point with no state change — I'm in a loop. Looking at
  the map, the door at (7, 1) is to my up-right but there's a wall. Let
  me try down-then-east instead.
  </think>

  use_tool(move_to, x_dest=5, y_dest=12)

After vLLM's --reasoning-parser qwen3:
  response.content           = "use_tool(move_to, x_dest=5, y_dest=12)"
  response.reasoning_content = "The user wants me to navigate. I'm at (5, 11)..."

What the agent harness sees:
  content = "use_tool(move_to, x_dest=5, y_dest=12)"   ← parsed as tool call
  (reasoning_content is ignored unless explicitly accessed)
```

## Decision criteria (n=3)

| Stage J mean | Reading |
|---|---|
| ≥ 71.43% | **Thinking budget breaks the ceiling.** Pursue thinking-mode scaffold additions: port to Gemma via fine-tune or `<think>` block in prompt examples, or fine-tune our own reasoning model. |
| ~ 57.14% (within σ ≤ 7pp) | **Thinking doesn't lift.** Either same plateau for same reason (state-observation limit, action-vocabulary limit) OR gain from wrong reasoning style. **Stage K cumulative memory becomes the next test.** |
| ≤ 42% | Thinking interferes with tool emission (Qwen3-Thinking sometimes overthinks and never closes the `<think>` block before max_tokens) or 30B/3B is meaningfully weaker than 35B/3B. Diagnose via trajectory introspection. |

## Why this is the *right* next test (not Stage K)

- **Stage H is the direct precursor**: Stage J is one variable change. Cleanest experiment.
- **Falsifies the "more reasoning budget" hypothesis cheaply**: 3 iters × ~50 min (Int4 MoE 3B-active should be ~3× faster than Qwen 3.5 35B-A3B-Int4 due to smaller total + AWQ vs GPTQ). Wall-time: ~2.5 hours for n=3.
- **Result informs Stage K priority**: if thinking lifts past 57.14% → cumulative memory might compound the gain. If thinking doesn't lift → Stage K tests a totally different lever (state-carryover, not reasoning-depth).

## Run

After Stage H iter 3 finishes (~18:40Z), swap vLLM:

```bash
pkill -f 'vllm.entrypoints.openai.api_server'
nohup ./serving/qwen_serve.sh \
    cyankiwi/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit \
    >/tmp/qwen_thinking_serve.log 2>&1 & disown
until curl -s http://localhost:8000/v1/models | grep -qi 'thinking'; do sleep 5; done

# Launch Stage J n=3
nohup bash experiments/stage_j_qwen_thinking/run_pokemon_n3.sh \
    >/tmp/stage_k_thinking_n3.log 2>&1 & disown
```

Notes on Qwen3-Thinking specifics:
- vLLM serves it with `--reasoning-parser qwen3` (auto-enabled by `qwen_serve.sh` when model name contains `thinking`).
- `max_tokens=8192` per call (vs 4096 for non-thinking) to give `<think>` blocks room before the tool-call.
- `max_model_len=16384` keeps KV cache + thinking context fit.

## Out of scope

- **Cross-game thinking-mode** (mario/2048) — wait for pokemon result. If lift, expand.
- **Stage K cumulative memory** — queued AFTER Stage J. If Stage J lifts, Stage K becomes "does cumulative memory + thinking-mode compound?". If Stage J doesn't lift, Stage K becomes the standalone test of a different lever.
- **Gemma-with-thinking** — would need a Gemma-Thinking variant or fine-tune. None exist today.
