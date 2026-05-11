# Agentic RL Pipeline — Systematic Post-Training Guide

How to systematically fine-tune game-playing VLMs and update weights in production.
Grounded in the existing infrastructure: `gemma4_rl/train.py` (Unsloth + TRL GRPO),
`agents/macla/online_evaluator.py` (RewardShapers), and the autoresearch eval loop.

---

## Core principle

**The model is one lever. Architecture and prompting are the others.**
Fine-tuning is only worth doing when the model is the bottleneck — not MACLA's memory
system, not the reward shaper, not the eval harness. The gating experiment (run
`harness_check` with a better base model) should always precede any training run.

From `macla_findings.md`: mario hit 100% via checkpoint carry-over alone. Architecture was
the bottleneck, not model quality. Fine-tuning mario trajectories would have been wasted compute.

---

## Training paradigms — what exists and when to use each

All trainers are compiled at `/workspace/gemma4_rl/unsloth_compiled_cache/`:
`UnslothSFTTrainer`, `UnslothGRPOTrainer`, `UnslothDPOTrainer`, `UnslothKTOTrainer`,
`UnslothRLOOTrainer`. GRPO orchestration lives in `gemma4_rl/train.py`.

### 1. Rejection Sampling Fine-Tuning (RFT / SFT)

**What:** Collect N game episodes → keep top-K% by `final_score` → SFT on winning trajectories.
**Trainer:** `UnslothSFTTrainer`
**Cost:** ~1× baseline wall-clock.

Use when:
- First experiment after establishing a clean baseline — lowest risk, fastest signal
- Model makes systematic errors good trajectories would correct (wrong move ordering,
  ignoring corner tile in 2048, walking into walls in mario)
- Have ≥ 5K filtered trajectory steps (~50+ high-quality episodes)

Do NOT use when:
- Game is already solved (mario 100%)
- Eval harness is still being debugged — training on confounded data amplifies noise
- Bottleneck is architecture (MACLA memory/procedure system), not model capability

**Reward function needed:** None — SFT just imitates the filtered trajectories.

---

### 2. Group Relative Policy Optimization (GRPO)

**What:** Sample N rollouts per game-state prompt, normalize rewards within the group,
backprop the policy gradient. No value/critic network. DeepSeek-R1 style.
**Trainer:** `GRPOTrainer` (TRL) + Unsloth patches — wired in `gemma4_rl/train.py`.
**Cost:** ~10–20× SFT wall-clock (rollouts dominate).

Use when:
- SFT ceiling reached — imitation can't go beyond trajectory quality
- Reward is verifiable and dense enough to produce signal within N rollouts
- Want the model to discover strategies beyond what the trajectory corpus shows

**The 2048 case:** Strongest fit. Score is a clean verifiable reward. Corner-anchor
strategy can be discovered via GRPO exploration — SFT on human trajectories may not
demonstrate it consistently, but GRPO with a corner-bonus reward can find and reinforce it.

**Wiring to the existing RewardShaper API:**

GRPO reward functions need signature `(completions, **kwargs) -> list[float]`.
`RewardShaper` classes in `online_evaluator.py` already compute per-step rewards.
The bridge is a thin wrapper — pattern matches `dd_explainer_rewards.py` in `gemma4_rl/`:

```python
# experiments/training/game_reward_fns.py
from agents.macla.online_evaluator import TwentyFortyEightShaper

_shaper = TwentyFortyEightShaper(shaping=DEFAULT_2048_SHAPING)

def reward_2048_step(completions, game_states, **kwargs) -> list[float]:
    rewards = []
    for completion, (prev_state, cur_state) in zip(completions, game_states):
        is_fallback = "<fallback>" in completion or not completion.strip()
        prev = _shaper.extract_metrics(prev_state)
        cur  = _shaper.extract_metrics(cur_state)
        r = _shaper.compute_reward(prev, cur, success=False, is_fatal=is_fallback)
        if is_fallback:
            r = -1.0  # matches training_plan.md rule
        rewards.append(r)
    return rewards
```

Pass `reward_funcs=[reward_2048_step, ...]` to `GRPOTrainer` — same pattern as `REWARD_FUNCS`
in `gemma4_rl/dd_explainer_rewards.py`.

Key GRPO hyperparameters for game trajectories:
- `num_generations`: group size. Start at 4, use 8 if compute allows.
- `beta`: KL penalty. Start at 0.01. Increase if policy diverges early.
- `max_completion_length`: 512 tokens covers a full tool call with reasoning.
- `lora_rank`: 16–32 for E4B; 32–64 for 26B A4B (more capacity to steer the larger model).

---

### 3. Direct Preference Optimization (DPO)

**What:** Collect pairs of (winning trajectory, losing trajectory) from the same starting
state. Train the model to prefer the winner without a reward model.
**Trainer:** `UnslothDPOTrainer`
**Cost:** ~2–3× SFT.

Use when:
- Want contrastive learning without full RL rollout cost
- Pokemon: label "reached PalletTown" episodes as `chosen`, "stuck in house" as `rejected`
- Have paired rollouts from the same checkpoint (run ≥ 2 episodes per start state)

**Data format:** `{prompt, chosen, rejected}` — the trajectory JSONL already captures this
if you run multiple rollouts and compare by `final_score`.

---

### 4. KTO (Kahneman-Tversky Optimization)

**What:** Non-paired preference learning. Each trajectory is labelled "desirable" or
"undesirable" — no need for matched pairs from the same start state.
**Trainer:** `UnslothKTOTrainer`
**Cost:** ~1.5× SFT.

Use when:
- Pokemon sparse reward: label any episode that discovered a new map as desirable.
  No need for a matched failing episode — unpaired labels are enough.
- Large corpus of labelled trajectories but no paired data available

---

### 5. RLOO (REINFORCE Leave-One-Out)

**What:** REINFORCE with a leave-one-out baseline. Simpler than GRPO, no grouping.
**Trainer:** `UnslothRLOOTrainer`
**Cost:** ~5–10× SFT.

Use when:
- Debugging RL signal before committing to full GRPO (faster iteration loop)
- Prototyping new reward functions — RLOO converges faster per step

---

## Decision matrix — paradigm per game and scenario

Criteria:
- **Reward density:** steps per non-zero reward (dense = every step; sparse = rare events)
- **Data available:** quality filtered episodes on disk after `filter_top_k.py`
- **Model bottleneck:** does swapping to a better base model lift scores ≥ +10%?
- **Harness clean:** post-fix baselines established, no confounded signal

| Game | Scenario | Approach | Rationale |
|---|---|---|---|
| mario | 100% achieved | **Do nothing** | Architecture solved it. Training risks regression. |
| mario | New model, re-baselining | SFT on 1 good run | Distil carry-over trajectory into new model quickly |
| 2048 | Score < 10%, base plateau | **GRPO** | Clean reward, RL finds corner-anchor SFT can't |
| 2048 | First experiment, uncertain | RFT first | Fast sanity: does imitating top trajectories help? |
| pokemon | 14.29% post-fix, Stage A | **DPO or KTO** | Sparse reward; label map-discovery; no dense step signal |
| pokemon | Missing post-fix Stage C | **Baseline first** | Can't train until harness confirmed clean |
| any game | Model bottleneck confirmed | GRPO | Full RL with `RewardShaper` wired as reward functions |
| any game | Architecture bottleneck | No training | Fix MACLA, prompting, or swap to a better base model |
| any game | < 1K quality steps | No training yet | Insufficient signal — run more autoresearch sweeps first |

**How to determine which bottleneck:**
```
1. Swap to a better base model (e.g. Gemma 4 26B A4B AWQ)
2. Re-run harness_check (Stage A, no substrate)
3. Compare against current Gemma E4B Stage A baseline

   Δ > +10% on any game  →  model was bottleneck  →  training will compound
   Δ ≈ 0                 →  architecture bottleneck → improve MACLA first
   Δ < 0                 →  regression — debug model config
```

---

## Systematic weight update pipeline

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐   ┌────────┐
│ COLLECT  │→  │  FILTER  │→  │  TRAIN   │→  │   SAVE   │→  │ SERVE  │→  │  EVAL  │
│          │   │          │   │          │   │          │   │        │   │        │
│autores-  │   │filter_   │   │train.py  │   │LoRA      │   │vLLM    │   │autores-│
│earch     │   │top_k.py  │   │(GRPO/    │   │adapter   │   │--lora  │   │earch   │
│sweep     │   │(to build)│   │SFT/DPO)  │   │saved     │   │or      │   │harness │
│          │   │          │   │          │   │          │   │merge+  │   │_check  │
│game_logs/│   │filtered_ │   │gemma4_rl/│   │gemma4_rl/│   │AWQ     │   │        │
│*/logs/   │   │traject-  │   │gemma_4_  │   │gemma_4_  │   │        │   │compare │
│trajecto- │   │ories.    │   │lora/     │   │lora/     │   │        │   │vs      │
│ry_samples│   │jsonl     │   │          │   │adapter_  │   │        │   │baseline│
│.jsonl    │   │          │   │          │   │model.    │   │        │   │        │
│          │   │          │   │          │   │safe-     │   │        │   │        │
│          │   │          │   │          │   │tensors   │   │        │   │        │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └────────┘   └────────┘
     ↑                                                                         │
     └─────────────────────────── feedback loop ─────────────────────────────┘
```

### Step 1 — Collect rollouts

Run autoresearch sweep. Trajectories auto-saved to:
```
game_logs/<game>/<run_id>/logs/trajectory_samples.jsonl
```

Each row is ShareGPT-format with:
- `messages`: full conversation (system prompt + observation + tool call)
- `final_score`: episode evaluation score (0–100)
- `is_fallback`: step used fallback action (corrupted → training noise)

For GRPO specifically: need N rollouts from the same game-state prompt.
Run the same game config N times with the same start seed.

### Step 2 — Filter

Build `experiments/training/filter_top_k.py` (~50 LOC, no GPU needed):
```python
episodes = load_jsonl("game_logs/*/logs/trajectory_samples.jsonl")
episodes = [e for e in episodes if not e["is_fallback"]]   # drop corrupted steps
top_k = sorted(episodes, key=lambda e: e["final_score"], reverse=True)[:K]
save_jsonl(top_k, "experiments/training/filtered_trajectories.jsonl")
# Print stats: token count, score distribution, per-game breakdown
```

For DPO: pair episodes by game-start-state:
```python
pairs = [(high_score_ep, low_score_ep) for episodes sharing the same start state]
# Format: {"prompt": ..., "chosen": ..., "rejected": ...}
```

### Step 3 — Train

**RFT / SFT:**
```bash
cd /workspace/gemma4_rl
python train.py train \
  --model-name unsloth/gemma-4-E4B-it \
  --data-dir experiments/training/filtered_trajectories.jsonl \
  --save-path gemma_4_lora \
  --lora-rank 16 --max-steps 200
```

**GRPO** (requires `game_reward_fns.py` wired in — see §2 above):
```bash
python train.py train \
  --model-name unsloth/gemma-4-E4B-it \
  --save-path gemma_4_lora \
  --lora-rank 32 --num-generations 8 --max-steps 100
```

### Step 4 — Save and validate

`train.py` calls `_verify_adapter_nonzero(save_path)` post-training — confirms adapter
weights are non-zero. If this fails the training run was degenerate (reward invariant;
model never updated). Check reward function, group size, and learning rate.

Adapter output at `gemma4_rl/gemma_4_lora/`:
```
adapter_model.safetensors   ← delta weights only (~50–200 MB for rank 32)
adapter_config.json
<config_name>/exp_<N>/      ← per-experiment snapshots (auto-snapshotted by train.py)
```

### Step 5 — Serve: two deployment paths

**Path A — LoRA hot-swap (development, daily retraining):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model unsloth/gemma-4-E4B-it \
  --enable-lora \
  --lora-modules game_agent=/workspace/gemma4_rl/gemma_4_lora \
  --port 8000 --tool-call-parser pythonic
```
Use `model="game_agent"` in API calls. No merge step. ~5–10ms extra latency per forward pass.

**Path B — Merge + AWQ (production / Cloud Run L4):**
```bash
# 1. Merge (needs ~16GB VRAM or offload to CPU)
python -c "
from unsloth import FastModel
model, tok = FastModel.from_pretrained('unsloth/gemma-4-E4B-it', load_in_4bit=False)
model = model.load_adapter('/workspace/gemma4_rl/gemma_4_lora')
model = model.merge_and_unload()
model.save_pretrained('/workspace/merged_model'); tok.save_pretrained('/workspace/merged_model')
"
# 2. AWQ quantize (if targeting L4 VRAM budget)
python -c "
from awq import AutoAWQForCausalLM, AutoTokenizer
model = AutoAWQForCausalLM.from_pretrained('/workspace/merged_model')
tokenizer = AutoTokenizer.from_pretrained('/workspace/merged_model')
model.quantize(tokenizer, quant_config={'zero_point': True, 'q_group_size': 128, 'w_bit': 4})
model.save_quantized('/workspace/merged_model_awq')
"
# 3. Redeploy
vllm serve /workspace/merged_model_awq --quantization awq --port 8000
```

| Situation | Path | Rationale |
|---|---|---|
| Development / daily iteration | A (hot-swap) | No merge overhead; fast turnaround |
| Production / Cloud Run L4 | B (merge + AWQ) | No adapter overhead; fits L4 VRAM |
| A/B testing adapter vs base | A (two `--lora-modules`) | Serve both simultaneously |
| After Phase 1 validates lift | B | Bake improvements into production checkpoint |

### Step 6 — Evaluate

```bash
python experiments/autoresearch.py \
  --config gemma --tag harness_check_post_sft --n-iters 3
```

Pass criterion (`training_plan.md`): mean score > Stage A baseline + 10% on ≥ 1 game.
If flat: architecture is the bottleneck — improve MACLA before next training run.

---

## When to fine-tune vs serve a better base — decision tree

```
START: scores plateau / want improvement
│
├─ Eval harness validated post-fix?
│   └─ NO → Fix harness first. Training on confounded signal amplifies noise.
│
├─ Run gating experiment: swap to Gemma 4 26B A4B (AWQ) or Qwen3-VL-8B-FP8
│   └─ Score improves > +10% on any game?
│       ├─ YES → Model was bottleneck.
│       │         Fine-tune the new base. Training compounds on stronger foundations.
│       └─ NO  → Architecture / prompting bottleneck.
│                 Improve MACLA (procedures, memory, reward shaping).
│                 Fine-tuning the current model will not move scores.
│
IF fine-tuning is warranted:
│
├─ Clean, verifiable reward signal?
│   ├─ YES (2048, mario post-fix, pokemon post-fix) → GRPO once SFT plateaus
│   └─ NO (new game, confounded harness)           → SFT only until signal clean
│
├─ ≥ 5K quality filtered steps on disk?
│   ├─ YES → Proceed
│   └─ NO  → Run more autoresearch sweeps. < 1K steps overfits badly.
│
├─ Game already solved (score > 80%)?
│   ├─ YES → Do not train. Risk of regression. Lock config.
│   └─ NO  → Continue
│
└─ Select paradigm:
    ├─ Dense reward + want exploration beyond demos → GRPO
    ├─ Good paired trajectories, contrastive needed → DPO
    ├─ Sparse reward, unpaired labels available     → KTO
    └─ First experiment, lowest risk               → RFT / SFT
```

---

## Per-game RL strategy (post-ablation, May 2026 — updated post-PR #31)

> **Update (post-PR #31 merge):** All numbers below are now from the AWQ-26B
> A/B/C/D matrix shipped by PR #31. The PR #28 E4B verdicts are listed in
> parentheses as "legacy" — they no longer drive strategy.

### mario — do not train (verdict holds, mechanism updated)

PR #31 AWQ-26B ablation: A=44.50%, **B=61.26%** (planner +16.76 pp),
C=35.18% (**vmem hurts: −9.32 pp**), D=35.21% (vmem regression persists
even with planner). Legacy PR #28 E4B verdict was "vmem null"; AWQ-26B
shows vmem is actively negative on mario's reactive gameplay.

Bottleneck was MACLA procedure compounding via carry-over (PR #22), not
model quality. Config locked to **Stage B** (planner-only) post-PR #31 —
turn vmem OFF for mario. Training mario trajectories still risks
overfitting to W1-1 layout; do not train.

### 2048 — strategy pivots from "GRPO target" to "Stage D config-only"

PR #31 AWQ-26B ablation: A=54.55%, **B=63.64%**, C=54.55% (parity),
**D=63.64%** (vmem on top of planner adds nothing). Stage B = Stage D
both at max_tile=128 (log2=7, score 7/11=63.64%). Legacy PR #28 E4B
verdict was "Stage C wins for 2048 (vmem helps, planner hurts)" — the
26B + new outcome-tagged history wiring flips this completely.

GRPO is no longer the obvious next step — the planner alone delivers
the full lift, and the ceiling at 300 steps is max_tile=128 (the
Bayesian procedure selector is the harder problem, not reward shaping).
If/when GRPO is revisited, the reward design stays the same:

1. `TwentyFortyEightShaper.compute_reward()` per step (base signal)
2. `+0.5` if max tile in any corner after move (corner-anchor bonus)
3. `+0.2` per monotone row or column maintained (chain bonus)
4. `-1.0` for `is_fallback=True` steps (silent fallback becomes penalty)

Target now: consistent **max_tile=256** (would be 72.7%). Step budget
likely the binding constraint before reward shaping.

### pokemon — Stage D shipping; D++ for the headline; SFT only if D++ plateaus

PR #31 AWQ-26B ablation: A=28.57% (model only), **B=57.14%** (planner
alone), C=0.00% (vmem-only collapse — agent oscillates RedsHouse2f ↔ 1f
re-reading SIGN_REDSHOUSE1F_TV), **D=57.14%** (vmem adds nothing on
top of planner at 300 steps), **D++=71.43%** (Stage D at 600 steps +
LoopDetector grace, banks milestone 5 = Gym Brock approach).

The 14.29% plateau the original report identified is **broken**.
Pokemon Stage D-with-prompt-fix landed at 4× the pass criterion the
postscript set out. Per-game `gemma_26b.yaml` ships Stage D + the
adapter-specific `SUBTASK_PLANNER_SYSTEM` override (PR #31's planner
fix).

PR #31's diagnostics also flagged three secondary failures worth tracking:
- **Stage C-class variance is dominated by step-8 LLM sampling**
  (T=0.7 hits the wrong WarpPoint half the time). Temperature 0.3
  shifts spatial behaviour (move_to(3,7)×30 vs zero at T=0.7) but
  exposes a tool-selection failure (`move_to` instead of
  `warp_with_warp_point`). Not a training problem — a prompt /
  tool-validator one.
- **Obs preprocessor (PR #61)** is necessary but not sufficient — it
  accumulates explored map across steps but can't reveal tiles never
  observed. Pair with an exploration nudge or `WaypointGoalProvider`
  before any SFT attempt.
- **Self-reflection (PR #62)** at Stage D + n=1 tied score with Stage D
  baseline (57.14%) but banked milestone 3 (Charmander) at step 203,
  which most Stage D runs don't reach. Cross-game test on
  `feat/self-reflection` in flight to determine if it lifts at n>1.

If/when SFT is needed for pokemon: hand-written subgoal trajectories
(house → route1 → lab waypoints) keyed against the now-tracked
`game_states.jsonl` milestone events. DPO/KTO remain plausible only
*after* a Stage D++ multi-episode sweep confirms a clear positive vs
negative split — current single-episode runs are too noisy for
preference labels.

---

## Infrastructure gaps to close before first training run

In priority order — total is ~160 LOC of new code (the eval-baseline gap is closed):

| Item | LOC | Depends on | Blocks |
|---|---|---|---|
| `experiments/training/filter_top_k.py` | ~50 | trajectory JSONL on disk | SFT, DPO, KTO |
| `experiments/training/game_reward_fns.py` | ~100 | `online_evaluator.RewardShaper` | GRPO |
| `serving/gemma_serve.sh` with `--enable-lora` | ~10 | vLLM `--enable-lora` flag | Serving LoRA adapters |
| ~~Post-fix Stage C pokemon baseline~~ | ✅ done via PR #31 | — | unblocked |

Post-PR #31 reality check: with **vmem now established as parity-or-negative
cross-game** and **planner established as the universal lever**, the "pokemon
DPO/KTO" path that the original training plan blocked on a Stage C baseline
is no longer the primary lift candidate. The current scoreboard headline
(Stage D++ 71.43%) suggests step budget + waypoint-aware planning beat
preference learning at this stage. Training infra above is still useful
once Stage D++ plateaus, but it's not in the critical path right now.

Everything else (GRPO trainer, reward infrastructure, trajectory logging, eval harness)
is already written and has been exercised in the `gemma4_rl/` workspace.
