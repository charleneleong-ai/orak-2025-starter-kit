# Agentic RL Options — Where Next Beyond the Harness?

Comparing every direction discussed for moving scores beyond what Stage A (harness) and Stage B/C/D (cognitive primitives) achieved. Companion to [`training_plan.md`](training_plan.md) and [`macla_findings.md`](macla_findings.md).

> **Constraint we care about:** one architecture that works across all three games (mario / 2048 / pokemon). Game-specific optimisations are listed but filtered out of the recommendation.

## TL;DR

Three groups of approaches, ordered by engineering effort:

1. **Inference-time (no training)** — `Tree search`, `Voyager skills`, `RLM family`, `Reasoning model swap`. Cheapest to test, rules out hypotheses fast.
2. **RL post-training (uses existing trajectories)** — `RFT`, `DPO`, `GRPO`, `HER`, `Verifier RL`, `Reasoning distillation`. Standard 2026 stack; needs training infra (see [`training_plan.md`](training_plan.md)).
3. **Architectural** — `Reasoning model swap`, `Mixture-of-specialists` (Fastino-style TLMs). Bigger investment, structural lift.

**Single-pick recommendation for "one arch that does all":** **Full RLM** (Zhang/Kraska/Khattab, [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)) on top of the existing Stage A+B+C+D ladder. Inference-only, no training, generalises across games, composes with vector memory + subtask planner, builds toward Fastino MoE later.

---

## Full options table

| # | approach | type | generalises? | eng effort | inference cost (/step) | training cost | expected lift |
|---|---|---|---|---|---|---|---|
| 1 | RFT — filter top trajectories, SFT | RL post-train | ✓ | 1-2 days | 1× | $3-30 / cycle | 10-30% all games |
| 2 | GRPO — R1-style group-relative PO | RL post-train | ✓ | 5-7 days | 1× | $25-65 / cycle | 30-60% top game |
| 3 | Self-improvement loop — recurring rollout→filter→SFT | RL post-train | ✓ | 1 day on top of #1 | 1× | $30-200 / quarter | compounds over loops |
| 4 | Procedure distillation — MACLA procs → SFT data | RL post-train | ✓ | 5-7 days | 1× | medium | uncertain |
| 5 | Self-rewarding DPO — trajectory pairs as preferences | RL post-train | ✓ | 2-3 days | 1× | low (~1hr SFT-class) | 5-15% all games |
| 6 | Verifier RL (PRM/ORM) — small reward model | RL post-train | ✓ | 5-10 days | 1× + 0.1× scoring | medium-high | 10-30% on top of GRPO |
| 7 | Tree search at inference — MCTS / depth-N rollouts | inference | ✗ 2048-fit | 1-2 days | 5-100× | $0 | 2048: +500% (6.88→50+) |
| 8 | Voyager-style skill library | inference + light train | ~ pokemon-fit | 5-7 days | variable | low | pokemon: maybe break ceiling |
| 9 | HER — relabel failed trajectories with achieved goals | RL post-train | ✓ | 3-5 days | 1× | medium | pokemon: meaningful |
| 10 | Curriculum / level gen | RL post-train | ✗ mario-fit | 1-2 weeks | 1× | high | mario only |
| 11 | Reasoning model swap (R1-Distill) | architectural | ✓ | 1-2 days | 1.5-3× | $0 | pokemon: likely break ceiling |
| 12 | Train Gemma into reasoner via GRPO | RL post-train | ✓ | 7-10 days | 1.5× | high | overlaps with #2 |
| 13 | Reasoning distillation from teacher | RL post-train | ✓ | 4-6 days | 1.5× | medium | pokemon: meaningful |
| 14 | RLM-A — Recursive SubtaskPlanner | inference | ✓ | 1 day | 2-3× | $0 | pokemon: meaningful |
| 15 | RLM-B — RecursiveMemoryProvider | inference | ~ pokemon-fit | 2-3 days | 2-4× | $0 | pokemon: large impact |
| 16 | RLM-C — Tree-search recursive (2048) | inference | ✗ 2048-fit | 2 days | 20-100× | $0 | 2048: +500% |
| 17 | RLM-D — Full open-recursion RLM | inference | ✓ | 5-7 days | 2-10× variable | $0 | pokemon: large; others: moderate |
| 18 | RLM-E — Fastino MoE-of-specialists | architectural + train | ✓ | 2-3 weeks | 1× outer + 1× routed | high | structural lift everywhere |

## Filtered: "one arch that does all 3 games"

After dropping game-specific options (#7, #10, #16) and partial-fit ones (#8 Voyager is pokemon-skewed, #15 RLM-B benefits pokemon most):

| approach | composes with Stage A+B+C+D? | training? | inference cost | what it specifically fixes |
|---|---|---|---|---|
| 1 — RFT | ✓ | yes | 1× | weak action policy in general |
| 2 — GRPO | ✓ | yes ($) | 1× | weak credit assignment |
| 3 — Self-improvement loop | ✓ | yes (recurring) | 1× | drift over time |
| 5 — Self-rewarding DPO | ✓ | yes (cheap) | 1× | model can't tell good from bad |
| 6 — Verifier RL | ✓ | yes | 1.1× | dense reward for sparse games |
| 9 — HER | ✓ | yes | 1× | pokemon-class sparse reward |
| 11 — Reasoning model swap | ✓ | no | 1.5-3× | pokemon's "model too small" |
| 13 — Reasoning distillation | ✓ | yes (cheap) | 1.5× | makes Gemma into reasoner cheap |
| **14 — RLM-A (Recursive SubtaskPlanner)** | ✓ (Stage D extension) | no | 2-3× | sub-goal granularity |
| **17 — RLM-D (Full RLM)** | ✓ (new primitive) | no | 2-10× variable | whatever the model decides |
| 18 — Fastino MoE | ✓ (new tier) | yes (heavy) | 1× outer + 1× routed | per-task specialization |

## Recommendation

**If goal is biggest lift fast** → **#11 (reasoning model swap)** then **#14 (recursive subtask)**. Tests the "model too small" hypothesis directly + extends Stage D.

**If goal is most novel single move** → **#17 (Full RLM)**. Open-ended recursive decomposition is the most architecturally interesting thing in the table. Builds toward #18.

**If goal is incremental win on the existing data we already have** → **#5 (self-rewarding DPO)**. Uses `trajectory_samples.jsonl` + `failed_trajectories.jsonl` as preference pairs, no new data collection.

**Single-pick for "one arch direction"**: **#17 (Full RLM)**. Reasons:
- Generalises across games (no game-specific code)
- Inference-only — no training infra needed
- Composes with vmem + planner + harness (extends rather than replaces)
- Maps cleanly to Fastino MoE direction later (#18 = #17 with TLMs as recursion targets)
- Most directly translates the published RLM paper into actionable code

---

## References by approach (with abstracts)

Confidence labels: **✓** confident on the citation, **~** correct paper but might have year/ID slightly off, **?** describing the right line of work without pinning a specific paper.

### Foundational (cross-cutting)

- ✓ **ReAct: Synergizing Reasoning and Acting in Language Models** — Yao et al., 2022 — [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
  > Interleaves reasoning traces with task-specific actions in a single LLM, so reasoning helps plan and exceptions while actions probe the environment for additional info. Outperforms reasoning-only and action-only baselines on QA, fact verification, and decision-making.

- ✓ **Reflexion: Language Agents with Verbal RL** — Shinn et al., 2023 — [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
  > Improves language agents not by updating weights but by maintaining a verbal "self-reflection" memory the agent reads on subsequent attempts. Hits 91% on HumanEval, beating GPT-4-direct.

- ✓ **DPO: Direct Preference Optimization** — Rafailov et al., 2023 — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
  > Re-parameterises RLHF so the optimal policy can be extracted in closed form via a simple classification loss — no separate reward model, no PPO, no sampling during fine-tuning. Matches or beats PPO-RLHF while being far simpler and more stable.

- ✓ **Self-Rewarding Language Models** — Yuan et al. (Meta), 2024 — [arXiv:2401.10020](https://arxiv.org/abs/2401.10020)
  > LLM acts as its own reward model via LLM-as-a-Judge prompting; iterative DPO training improves both instruction-following AND the model's reward-quality, breaking past human-feedback ceilings. Llama 2 70B beats Claude 2, Gemini Pro, and GPT-4-0613 on AlpacaEval 2.0 after 3 iterations.

### #1 — RFT (rejection sampling fine-tune)

- ✓ **STaR: Self-Taught Reasoner** — Zelikman et al., 2022 — [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
  > Bootstrap loop: prompt model with a few rationale examples → generate rationales → keep ones that produce correct answers → fine-tune on those → repeat. Enables successively-more-complex reasoning from a tiny seed.

- ✓ **ReST (Reinforced Self-Training)** — Gulcehre et al. (DeepMind), 2023 — [arXiv:2308.08998](https://arxiv.org/abs/2308.08998)
  > Growing-batch offline RL: model generates samples → use as fixed dataset → fine-tune with offline RL → repeat. More efficient than online RLHF on machine translation.

- ✓ **ReST-EM: Beyond Human Data** — Singh et al., 2023 — [arXiv:2312.06585](https://arxiv.org/abs/2312.06585)
  > EM-flavoured self-training: generate samples, filter with binary feedback (e.g. test passes / theorem holds), fine-tune, iterate. Scales favourably with model size; surpasses human-data SFT on MATH and APPS using PaLM-2.

- ~ **RAFT: Reward rAnked FineTuning** — Dong et al., 2023 — [arXiv:2304.06767](https://arxiv.org/abs/2304.06767)
  > Simpler RLHF alternative: use the reward model to *rank/filter* samples, then SFT on the survivors. Avoids PPO's instabilities; works on both LLMs and diffusion models.

### #2 — GRPO

- ✓ **DeepSeekMath** (introduces GRPO) — DeepSeek, 2024 — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
  > Group Relative Policy Optimization: PPO variant that computes advantages from a *group* of samples around the same prompt instead of needing a critic network — cuts PPO's memory roughly in half. Combined with math-targeted pre-training, the 7B model gets 51.7% on MATH (approaching GPT-4 / Gemini-Ultra).

- ✓ **DeepSeek-R1** — DeepSeek, 2025 — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
  > Pure RL (no human-labelled reasoning trajectories) develops self-reflection, verification, and dynamic strategy adaptation in LLMs — emergent reasoning. Achieves SOTA on math/coding, and the patterns transfer cleanly to smaller distilled models.

### #3 — Self-improvement loop

- ✓ **AlphaZero** — Silver et al., Nature 2017 — superhuman play in Go/chess/shogi from self-play with MCTS-shaped policy improvement; the canonical "rollout → improve → repeat" loop.
- ✓ **Self-Rewarding LMs** — [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) — see Foundational; same paper applies as the LLM-era version of the loop.
- See also **ReST-EM** above for the EM-style iteration pattern.

### #4 — Procedure distillation

Our framing — closest references are skill-library and reward-distillation work:

- ✓ **Voyager** — [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) — see #8.
- ✓ **Eureka** (reward code from LLMs) — Ma et al., NVIDIA, 2023 — [arXiv:2310.12931](https://arxiv.org/abs/2310.12931)
  > LLM writes/refines reward functions as code via evolutionary search. Outperforms human-written rewards on 83% of 29 tasks; first demo of dexterous pen-spinning on a simulated robot hand.
- ? Generic knowledge distillation (Hinton, 2015) — soft-target distillation as the original supervised compression technique.

### #5 — Self-rewarding DPO

- ✓ **DPO** — see Foundational.
- ✓ **KTO: Model Alignment as Prospect Theoretic Optimization** — Ethayarajh et al., 2024 — [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)
  > Aligns LLMs from *binary desirability signals* (no preference pairs needed) via a loss derived from Kahneman-Tversky prospect theory. Matches or beats DPO while requiring less data structure.
- ~ **IPO: A General Theoretical Paradigm to Understand Learning from Human Preferences** — Azar et al., 2023 — [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
  > Generalises RLHF and DPO into a single ΨPO framework that bypasses both the pointwise-reward and reward-generalisation approximations. Identifies pitfalls in DPO and proposes the Identity-PO variant with provable guarantees.
- ✓ **Self-Rewarding LMs** — see Foundational.

### #6 — Verifier RL (PRM/ORM)

- ✓ **Let's Verify Step by Step** — Lightman et al. (OpenAI), 2023 — [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
  > Process supervision (per-step feedback) substantially outperforms outcome supervision on MATH; their PRM-trained model solves 78% of a held-out MATH subset. Releases PRM800K — 800K step-level labels.
- ✓ **Solving math word problems with process- and outcome-based feedback** — Uesato et al. (DeepMind), 2022 — [arXiv:2211.14275](https://arxiv.org/abs/2211.14275)
  > First systematic comparison of outcome-only vs process-step feedback on GSM8K. Outcome supervision matches final-answer accuracy with less labeling, but process supervision is necessary to fix *reasoning* errors (3.4% vs 14% reasoning-error rate).
- ✓ **Math-Shepherd** — Wang et al., 2024 — [arXiv:2312.08935](https://arxiv.org/abs/2312.08935)
  > Trains a step-level reward model *without human annotations* by automatically constructing supervision data. Uses the PRM as both verifier and RL critic on GSM8K/MATH for substantial gains.

### #7 — Tree search at inference

- ✓ **Tree of Thoughts (ToT)** — Yao et al., 2023 — [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
  > Generalises chain-of-thought to a tree where the model explores multiple reasoning paths, self-evaluates branches, and backtracks. 74% on Game of 24 vs 4% for CoT-GPT-4.
- ✓ **Reasoning with Language Model is Planning with World Model (RAP)** — Hao et al., 2023 — [arXiv:2305.14992](https://arxiv.org/abs/2305.14992)
  > LLM is repurposed as both reasoning agent AND world model; MCTS over the LLM-simulated state transitions. Beats CoT and ToT on plan generation, math, and logic.
- ✓ **Graph of Thoughts** — Besta et al., 2023 — [arXiv:2308.09687](https://arxiv.org/abs/2308.09687)
  > Generalises ToT further: thoughts form an arbitrary graph (with merge / refine / aggregate edges), enabling synergies and feedback loops linear/tree topologies can't express. +62% sorting quality vs ToT at -31% cost.

### #8 — Voyager-style skill library

- ✓ **Voyager: An Open-Ended Embodied Agent with Large Language Models** — Wang et al., 2023 — [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)
  > GPT-4-driven Minecraft agent with three pieces: automatic curriculum, an *executable-code* skill library, iterative prompting with environment feedback. Lifelong learning, no human intervention; substantial gains over prior agents.
- ✓ **Eureka** — see #4.
- ✓ **Code as Policies** — Liang et al. (Google), 2022 — [arXiv:2209.07753](https://arxiv.org/abs/2209.07753)
  > Few-shot prompts an LLM to synthesize Python *robot policy code* from natural-language commands. Hierarchical code generation (functions calling functions) is the key — recursive policy synthesis. Demonstrated across multiple real robot platforms.

### #9 — HER (Hindsight Experience Replay)

- ✓ **Hindsight Experience Replay** — Andrychowicz et al. (OpenAI), 2017 — [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)
  > For sparse-reward RL: relabel each failed trajectory with the goal it *did* achieve, treat it as a successful demo. Off-policy compatible; an implicit curriculum. Validated on robot manipulation, sim-to-real.
- ✓ **Reflexion** — see Foundational; the LLM analogue of "relabel and learn from failure."

### #10 — Curriculum / level gen

- ~ **Curriculum Learning** — Bengio et al., ICML 2009 — train on easy examples first, progressively harder. Foundational machine-learning paper.
- ✓ **POET: Open-ended Co-Evolution** — Wang et al. (Uber AI), 2019 — [arXiv:1901.01753](https://arxiv.org/abs/1901.01753)
  > Co-evolves a population of *environments* and *agents*, with stepping-stone solutions transferring between problem instances. Yields behaviours unattainable from direct optimisation alone.
- ? MAESTRO — Garcin et al., 2024 (LLM-side curriculum learning, several papers with similar names)
- ? PCGRL (Procedural Content Generation via RL) literature.

### #11 — Reasoning model swap

- ✓ **DeepSeek-R1** — see #2 (includes R1-Distill-Qwen-1.5B/7B/14B/32B and R1-Distill-Llama-8B/70B).
- ✓ OpenAI o1 system card (Sept 2024) — public link on platform.openai.com.
- ? Qwen-Reasoning variants in the Qwen3 series.

### #12 — Train Gemma into reasoner via GRPO

Same references as #2 (DeepSeekMath, R1).

### #13 — Reasoning distillation

- ✓ **Distilling Step-by-Step** — Hsieh et al. (Google), 2023 — [arXiv:2305.02301](https://arxiv.org/abs/2305.02301)
  > Train smaller task-specific models using the *rationales* generated by a large LLM as additional supervision (multi-task SFT: predict label + rationale). 770M T5 beats 540B PaLM with 80% of the data.
- ✓ **Orca: Progressive Learning from Complex Explanation Traces of GPT-4** — Mukherjee et al. (Microsoft), 2023 — [arXiv:2306.02707](https://arxiv.org/abs/2306.02707)
  > 13B Orca learns from GPT-4's *full reasoning traces* (not just answers) with ChatGPT-guided progressive learning. Hits parity with ChatGPT on Big-Bench Hard.
- ~ R1-Distill-* models reported in [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) itself.

### #14 — RLM-A: Recursive SubtaskPlanner

- ✓ **Recursive Language Models** — Zhang/Kraska/Khattab, 2025 — [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) — see #17.
- ✓ **Plan-and-Solve Prompting** — Wang et al., 2023 — [arXiv:2305.04091](https://arxiv.org/abs/2305.04091)
  > Zero-shot prompting variant: "first devise a plan to divide the task into subtasks, then carry them out". Beats Zero-shot-CoT on multiple reasoning benchmarks; addresses missing-step errors specifically.
- ~ **HuggingGPT** — Shen et al., 2023 — [arXiv:2303.17580](https://arxiv.org/abs/2303.17580)
  > ChatGPT as a controller orchestrating Hugging Face models for multi-modal tasks: task planning → model selection → execution → response summarization. Closest precursor to "central LLM dispatching to specialists."

### #15 — RLM-B: RecursiveMemoryProvider

- ✓ **Recursive Language Models** — see #17.
- ✓ **MemGPT: Towards LLMs as Operating Systems** — Packer et al., 2023 — [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
  > Treats the LLM as an OS with a virtual context manager — paged memory, interrupt-based control flow. Extends usable context far beyond native window for extended conversations and document analysis.

### #16 — RLM-C: Tree-search recursive (2048)

- Same as #7 (ToT) plus the RLM paper for the recursive structure.
- **AlphaZero** (game tree search baseline) for the structural inspiration.

### #17 — RLM-D: Full open-recursion RLM

- ✓ **Recursive Language Models** — Zhang/Kraska/Khattab, 2025 — [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
  > LLM treats long prompts as an external environment: programmatically examines, decomposes, and recursively self-calls on snippets. Handles 100× context-window inputs; RLM-Qwen3-8B beats vanilla Qwen3-8B by 28.3% avg, approaches GPT-5 on long-context tasks at comparable cost.

### #18 — RLM-E: Fastino MoE-of-specialists

Fastino's TLM (Task Language Model) thesis is mostly company-side / press, not arxiv. Closest academic analogues:

- ✓ **Switch Transformer** — Fedus et al. (Google), 2021 — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
  > Simplified MoE routing → trillion-parameter sparse models trainable in bfloat16, with 7× pre-training speedup at constant compute. The infrastructure paper that made MoE practical at scale.
- ✓ **Mixtral 8x7B** — Mistral AI, 2024 — [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
  > 8-expert sparse MoE: each token activates 2 of 8 feed-forward experts (47B params total, 13B active). Matches/beats Llama 2 70B and GPT-3.5 on most benchmarks. The "open-weights MoE works in production" demonstration.
- ✓ **MetaGPT** (multi-agent role specialization) — Hong et al., 2023 — [arXiv:2308.00352](https://arxiv.org/abs/2308.00352)
  > Encodes Standardised Operating Procedures into prompt sequences across role-specialised LLM agents (PM, architect, engineer, QA). Reduces cascading hallucinations vs naive multi-agent chat. Closest agentic analogue to per-task specialists routed by a coordinator.

---

## Tight 8-paper reading list

If you want the smallest set that covers the design space:

1. **ReAct** — [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) — foundational pattern most agentic systems use
2. **Reflexion** — [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) — memory + self-correction
3. **Voyager** — [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) — skill library accumulation
4. **Tree of Thoughts** — [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) — inference-time tree search
5. **DeepSeek-R1** — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — current SOTA reasoning + GRPO recipe
6. **Self-Rewarding LMs** — [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) — self-improvement loop
7. **MemGPT** — [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) — model-as-OS, "context as environment" precursor
8. **Recursive Language Models** — [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) — the recursion paper anchoring this whole discussion

Reading those 8 covers ~80% of the conceptual surface for everything in the comparison table.

## Caveats

A few entries I'm less certain on:

- **HER for LLM agents**: I'm sure on the original 2017 HER paper but the LLM-aware follow-ups are scattered — there isn't a single canonical citation.
- **Procedure distillation**: that's our framing for what MACLA + Voyager-skills do; no single citation.
- **Fastino**: their TLM thesis is mostly company-internal / press; no academic paper to point at directly. Cited the closest MoE / multi-agent analogues instead.
- **Curriculum / level generation**: the LLM-agent-applied versions are scattered; cited the foundational POET paper.
- **MAESTRO**: I noted this in passing but haven't pinned a specific arxiv ID — there are several papers with similar names in 2023-2024.

If any specific cell of the table needs deeper dive, the WebFetch tool can pull current abstracts.
