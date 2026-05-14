# Pokemon: every prior experiment ran without pokered/ asm files

**Status:** finding (2026-05-14)  •  **Scope:** all pokemon_red runs in this repo before 2026-05-14

## The finding

`evaluation_utils/mcp_game_servers/pokemon_red/game/pokered/` is the disassembly checkout that `pyboy_runner.parse_object_sprites()` reads to label on-screen objects with their real sprite names (`SPRITE_OAK`, `SPRITE_POKE_BALL_2`, `SPRITE_BOOKSHELF`, …). The directory is not tracked in git (not a submodule; the parent repo only ignores the `.gbc` ROM, not the asm dir — the dir simply ships empty), so unless a contributor follows the upstream [Orak pokemon setup guide](https://github.com/krafton-ai/Orak/blob/release/docs/setup_pokemon.md) and clones `pret/pokered` into this path, the directory is empty at runtime.

It has been empty in this checkout from day one. When the dir is empty, `parse_object_sprites()` returns `[]` and `get_object_coords()` (`evaluation_utils/mcp_game_servers/pokemon_red/game/pyboy_runner.py:450`) falls back to `f"OBJ_{i}"` placeholders, keyed `OBJ_{i}_{i}` — exactly the `OBJ_7_7`, `OBJ_6_6`, `OBJ_1_1` tokens the Stage J trajectories cycle through in OaksLab.

## Evidence

**Stage J wandb trajectory audit (n=3 × 901 reasoning HTMLs each):**

| Iter | Reasoning steps | Steps that reference `OBJ_n` | Steps that reference any `SPRITE_*` |
|---|---|---|---|
| 1 | 901 | 670 (74%) | 4 |
| 2 | 901 | 689 (76%) | 23 |
| 3 | 901 | 707 (78%) | 22 |

74–78% of reasoning chains were anchored on placeholder `OBJ_n` tokens. The handful of `SPRITE_*` references come from the model's prior-knowledge of Pokemon Red, not from the harness providing them.

**PR #52 (commit `5b111d7`, 2026-05-07) — *the resolver fix that doesn't fix this*.** The case-insensitive `_resolve_asm_path` helper at `pyboy_runner.py:42` and its test at `tests/test_pokemon_asm_resolver.py` were added to handle the casing mismatch between `pret/pokered`'s `RedsHouse2F.asm` and our `map_names.json`'s `RedsHouse2f`. **That fix only helps if the asm files exist on disk.** With `pokered/` empty, the resolver returns `None` for every map and the placeholder fallback fires unconditionally.

## All Pokemon experiments affected

Every row in this table ran with `pokered/` empty. The reasoning chain inside each had no access to sprite names; agents saw `OBJ_n_n` everywhere.

| Date (first row) | Experiment | rows | Score | Source |
|---|---|---|---|---|
| 2026-03-28 → 2026-04-23 | macla (legacy) | 27 | 29.4% → 57.1% | `experiments/macla/results.jsonl` |
| 2026-04-30 | pokemon_check (Stage A baseline) | 3 | 0.0% | `experiments/pokemon_check/gemma/` |
| 2026-05-02 | harness_check_pokemon | 1 | 0.0% | `experiments/harness_check_pokemon/gemma/` |
| 2026-05-02 | pokemon_check_v4 | 2 | 0.0% | `experiments/pokemon_check_v4/gemma/` |
| 2026-05-03 | harness_check_pokemon_v4 | 2 | 14.3% | `experiments/harness_check_pokemon_v4/gemma/` |
| 2026-05-05 | pokemon_check_v13_reflection | 1 | 14.3% | `experiments/pokemon_check_v13_reflection/gemma/` |
| 2026-05-05 | v14_cross_game_planner | 3 | 5.3% (mix) | `experiments/v14_cross_game_planner/gemma/` |
| 2026-05-05 | v14b_pokemon_only | 1 | 14.3% | `experiments/v14b_pokemon_only/gemma/` |
| 2026-05-08 → 2026-05-11 | **PR #31 Stage A–D ablation (Stage D = 57.14% ceiling)** | 10 | 57.1% | `experiments/pr31_ablation_26b/gemma_26b/` |
| 2026-05-11 → 2026-05-12 | cross_game_self_reflect (Stage D+reflect) | 7 | 57.1% | `experiments/cross_game_self_reflect/gemma_26b/` |
| 2026-05-12 | langgraph_validation (Stage E) | 1 | 57.1% | `experiments/langgraph_validation/gemma_26b/` |
| 2026-05-12 | pdc_live (Stage F plan-do-check) | 1 | 28.6% | `experiments/pdc_live/gemma_26b/` |
| 2026-05-13 | **Stage H — Qwen3.5-35B-A3B-Int4 (n=3)** | 1 | 47.6% | `experiments/stage_h_qwen_ceiling/qwen35_a3b_int4/` |
| 2026-05-14 | **Stage J — Qwen3-Thinking 30B-A3B (n=3)** | 1 | 28.6% | `experiments/stage_j_qwen_thinking/qwen3_thinking/` |

Plus the procedure-layer experiments banked separately:
- `experiments/no_procedures/gemma_26b/` — Stage B' (PR #69)
- `experiments/stage_g_qwen_ceiling/*` / `loop_escape/*` — Stage G (PR #70)

Stages B, C and the autoresearch-G runs from PR #70 don't all have their results.jsonl in this checkout but inherit the same constraint.

## What this changes in the cross-stage diagnosis

`docs/experiments/gemma/cross-stage-diagnosis.md` concludes the 4/7 milestone ceiling lives in **LLM reasoning at the milestone boundary**. That conclusion was reached against a reasoning surface where the agent always saw `OBJ_n_n` instead of `SPRITE_OAK`, `SPRITE_POKE_BALL_*`, `SPRITE_BOOKSHELF`, `SPRITE_NPC`, etc. Specifically:

- **Stage A → D** ablation: agents could not distinguish Oak from a bookshelf, a Pokeball from a sign, or one NPC from another, except via prior-knowledge guessing on coordinates.
- **Stage E (verify_action)**: the 91% revision rate is on top of placeholder-anchored actions — the verifier was rewriting `interact_with_object(OBJ_2_2)` to other `OBJ_n_n` calls.
- **Stage F (plan-do-check)**: the validator's over-rejection rate is partially explained by the planner emitting actions against placeholders the validator couldn't ground.
- **Stage G (procedure escape, force-LLM-on-stuck)**: the unconditional LLM fallback fired on the same placeholder surface, so "LLM fallback can't break milestone 4" is conditional on the fallback receiving real sprite names.
- **Stage H (Qwen3.5-35B, 47.6%)** and **Stage J (Qwen3-Thinking, 28.6%)**: the PR #76 verdict ("30B/3B world-knowledge < 35B/3B" / "thinking interferes") was reached against placeholder-anchored input. With `pokered/` populated, the agent's reasoning input becomes a different distribution; both runs need to be re-evaluated.

**The diagnosis doesn't necessarily flip** — the 57.14% Stage D ceiling appearing across 7+ interventions is still a real ceiling. But the *cause* attribution shifts: from "LLM reasoning at the milestone boundary" to **"LLM reasoning given only placeholder sprite tokens at the milestone boundary"**, which is a much weaker claim about the model and a much stronger claim about the harness.

## What's needed next

1. **Populate `pokered/` in both the main checkout and the worktree.** Real `pret/pokered` should produce `data/maps/objects/*.asm` (~248 files). The current partial checkout in the worktree (May 14 21:07) is missing `data/` — needs to be re-cloned cleanly from `https://github.com/pret/pokered.git`.
2. **Verify by parsing a known map.** After clone, `python -c "from evaluation_utils.mcp_game_servers.pokemon_red.game.pyboy_runner import parse_object_sprites; print(parse_object_sprites('evaluation_utils/mcp_game_servers/pokemon_red/game/pokered/data/maps/objects/OaksLab.asm'))"` should print a non-empty list containing real sprite names.
3. **Re-run a Stage D baseline + Stage H ceiling check (n=3 each)** with `pokered/` populated. These two are the cheapest anchors for re-deciding whether the 57.14% ceiling actually holds with real sprite tokens.
4. **Update setup docs** so this gap cannot silently re-occur. Add a runtime check in `pyboy_runner.PyBoyRunner.__init__` that warns loudly (or hard-fails) if `self.asm_dir` is empty.
5. **Tag this finding on the relevant PRs.** PR #76 (Stage J) and the cross-stage diagnosis doc both have verdicts that are conditional on the placeholder reasoning surface. They should reference this note.

## References

- `evaluation_utils/mcp_game_servers/pokemon_red/game/pyboy_runner.py:42` — `_resolve_asm_path` (PR #52)
- `evaluation_utils/mcp_game_servers/pokemon_red/game/pyboy_runner.py:105` — `self.asm_dir = …/pokered/data/maps/objects`
- `evaluation_utils/mcp_game_servers/pokemon_red/game/pyboy_runner.py:450` — `OBJ_{i}` placeholder fallback
- `tests/test_pokemon_asm_resolver.py:8` — Stage A audit reference (`game_logs/pokemon_red/20260506_221856/`) showing the same OBJ_n_n problem
- Stage J trajectory HTMLs: `wandb/run-20260513_191745-pr_stage_j_qwen3_thinking_pokemon_iter1_*/files/media/html/reasoning_*.html`
- Cross-stage diagnosis (subject to reinterpretation): `docs/experiments/gemma/cross-stage-diagnosis.md`
- PR #76 trajectory introspection verdict (also subject to reinterpretation): https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/76
- Upstream setup: https://github.com/krafton-ai/Orak/blob/release/docs/setup_pokemon.md
