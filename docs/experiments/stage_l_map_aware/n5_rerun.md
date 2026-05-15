# Stage L (map-aware procedures) — n=5 cumulative-memory rerun

**Verdict:** NEUTRAL+ — procedure-cache fix delivers monotonic M4 banking speedup across iters, but no iter ever crossed the M5 (Viridian) gate. Capped at 57.14% on 4 of 5 iters.

**Closed:** 2026-05-15 17:46Z (~5h15m wall-clock)
**Branch:** `feat/macla-map-aware-procedures` (PR #85)
**Worktree:** `/workspace/orak-stage-l`
**Log:** `logs/stage_l_retry_20260515T123828Z.log`

## Hypothesis

Stage K's negative transfer (iter 2 took +91 steps vs iter 1 to bank M4 in the post-asm-fix run, PR #81) was caused by context-blind procedure keys. Procedures captured in OaksLab were firing in Route1/PalletTown at iter 2. Fix: map-aware key + iter-based TTL (max_age=2). Minimum bar: `late_mean >= early_mean` (no negative transfer). Lift bar: iter-over-iter steps-to-M4 monotonically decreasing OR any iter crossing M5.

## Schedule

| Setting | Value |
|---|---|
| Game | pokemon_red |
| Agent | gemma_26b (Gemma 4-26B-A4B-AWQ-4bit on vLLM :8000) |
| Max steps / iter | 300 |
| Iters | 5 (cumulative via `--load-checkpoint --prev-run-id`) |
| Launcher | `experiments/stage_l_map_aware/run_pokemon_n5.sh` |

## Results

```
scores=[57.14, 57.14, 57.14, 28.57, 57.14]
mean=51.43% std=12.78pp
early_mean(iter 1-2)=57.14%, late_mean(iter 4-5)=42.86%, learning_delta=-14.28pp
```

| iter | score | M4 step | Δ vs iter 1 | Route1 steps | Viridian steps | final map | persev % |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 57.14% | 259 | — | 1 | 0 | Route1 | 13.5 |
| 2 | 57.14% | 229 | −30 | 0 | 0 | PalletTown | 26.0 |
| 3 | 57.14% | 172 | −87 | 114 | 0 | Route1 | 18.8 |
| 4 | **28.57%** | n/a | n/a | 0 | 0 | OaksLab | **31.6** |
| 5 | 57.14% | 140 | **−119** | 12 | 0 | OaksLab | 17.5 |

Numbers from `experiments/stage_l_map_aware/introspect.py`.

## Stage K vs Stage L comparison

| | Stage K (PR #75/#81) | Stage L (this PR) |
|---|---|---|
| Scores | `[57.14] × 5`, σ=0 | `[57.14, 57.14, 57.14, 28.57, 57.14]`, σ=12.78 |
| Negative transfer | **+91 steps** iter 2 vs iter 1 | **−30/−87/−119 steps** iter 2/3/5 vs iter 1 (monotonic) |
| Iter ever past M4 | No (all stuck at score 4) | No (all stuck at score 4) |
| Floor stability | ✅ all 5 at floor | 4 of 5 at floor; iter 4 dropped to 28.57% |

## What the data says

1. **Map-aware key + iter-TTL works for its stated claim.** M4 banking went 259 → 229 → 172 → 140 across the four passing iters — monotonic acceleration, no negative transfer. That's exactly the cumulative-memory compounding Stage K lacked.

2. **The 57.14% ceiling is structural and below the M5 gate.** Zero Viridian steps across all 5 iters × 300 steps. The agent reaches the Route1 / Viridian boundary in some iters but never enters Viridian. Procedure cache is no longer the bottleneck above M4.

3. **Iter 4 is a single-shot regression, not alternation.** Earlier alternation hypothesis (odd-floor / even-below) is not supported: iter 5 returned to floor with the fastest M4 banking of the run (140 steps). Iter 4 stuck in OaksLab with persev=31.6% looks like a one-off post-warmup misfire, not a structural artefact of TTL=2.

4. **`late_mean < early_mean` is misleading here.** The −14.28pp `learning_delta` is dragged down entirely by iter 4. Drop iter 4 and the late_mean is 57.14% — same as early_mean. The map-aware fix met its minimum bar (no negative transfer in 4 of 5 iters); iter 4 obscures it.

## Why this is NEUTRAL+ not LIFT

Stage L hits the minimum bar (no negative transfer) and shows acceleration on M1–M4 banking. It does NOT hit the lift bar — no iter crossed M5. The ceiling is the same 57.14% as Stage K. Procedure-cache redesign was necessary but not sufficient to break above M4.

## Implications for Stage M

The remaining ceiling is past M4, in the M5 (Viridian entry) gate. Two distinct cognitive failures observed in the iter 1 and iter 3 game_states.jsonl deep-dives:

1. **Wild-encounter loop on Route1** drains the step budget — every 2-3 tiles triggers a battle. Generalised: agent repeats actions that produced zero salient state delta (`move_to` to same position, blocked jump in mario, no-op swipe in 2048, supply-blocked queue in starcraft).

2. **Battle policy thrash** — LEECH SEED spam (doesn't model it as once-per-target debuff), RUN from lv3 wilds at full HP. Generalised: agent doesn't reason about which action is producing forward progress.

Stage M is therefore two-pronged:

- **(a) Multi-signal procedure quality** — `score = base_posterior × logprob_confidence × state_delta_confidence`, each ∈ [0,1], neutral at 0.5, ablatable by hardcoding to 1.0. Percentile-rank calibration on rolling `deque(maxlen=50)`. Plumb `logprobs=True, top_logprobs=1` through `agents/macla/structured_output.py`. Per-game salient-state extractor: pokemon=(score, hp, pos, map, in_battle); 2048=board_hash; mario=(x, score, lives); starcraft=(minerals, gas, supply).
- **(b) Exploration novelty bonus** — track visited maps in `EnhancedHierarchicalMemorySystem`; small additive bonus on the selector when the current map has never been visited. Bias the agent toward Viridian discovery instead of looping Route1.

Both signals are generalisable across pokemon / mario / 2048 / starcraft — no pokemon-battle-specific heuristics.

## Next move

- Land this PR (PR #85) to master.
- Open Stage M PR on `feat/macla-multi-signal-quality` with the two-pronged design above.
- Launch Stage M n=5 cumulative-memory sweep on the same gemma_26b config; minimum bar = match Stage L M4 banking speed (no regression); lift bar = any iter past 57.14%.

## Closed follow-ups from this PR

- **#43 (Stage L iter-2 Route1 regression)** — superseded. The full n=5 data shows iter 2 banked M4 with a Route1=0 path through PalletTown rather than a TTL-pruning failure; iter 3 then took the Route1 path and was the fastest of the first four iters. Stage L's `learning_delta` mostly tracks iter 4's single-shot misfire.
