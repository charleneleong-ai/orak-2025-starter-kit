"""Game-specific adapter for StarCraft II — used by UnifiedMaclaAgent.

The env emits 5 actions per step ("1: A\\n2: B\\n...") and parses replies
through ``re.findall(r"\\d+: <?([^>\\n]+)>?", text)``; each name must hit
the Protoss action dictionary or the env substitutes EMPTY ACTION. This
adapter formats actions in that exact shape and lists the full Protoss
vocabulary in VALID_ACTIONS / the system prompt so the LLM never has to
guess action names.

Self-reflection is OFF by default — the RTS pacing of one step / ~10s
already absorbs LLM latency; adding a per-step critique call doubles that
without the long-horizon dialog payoff that justifies it on pokemon.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Re-exports read by UnifiedMaclaAgent via self._adapter.<NAME>.
SYSTEM_PROMPT = """\
You are a Protoss commander in a real-time 1v1 StarCraft II match against a
Zerg AI. Each turn you must commit to EXACTLY 5 actions that execute in
sequence; the env charges resources for each in order, so plan the chain
to stay solvent.

### Strategy primer
- Economy first: TRAIN PROBE early, BUILD PYLON the moment supply runs
  low, BUILD ASSIMILATOR once you need gas.
- Production gates research: BUILD GATEWAY → BUILD CYBERNETICSCORE before
  any STALKER / ADEPT line; BUILD STARGATE before air; BUILD ROBOTICSFACILITY
  before OBSERVER / IMMORTAL / COLOSSUS.
- Use CHRONOBOOST NEXUS / CHRONOBOOST CYBERNETICSCORE to accelerate key
  tech and worker counts. CHRONOBOOST charges are wasted if you don't spend
  them.
- MULTI-ATTACK / MULTI-RETREAT are army commands; use SCOUTING PROBE early
  to learn what the Zerg is opening.
- EMPTY ACTION is the legal no-op when waiting for resources or supply —
  prefer it over emitting an unaffordable action.

### Decision output
Emit a JSON object containing:
- `reasoning`: 2-4 sentence strategic justification.
- `current_goal`: short label (e.g. "Expand economy", "Pylon supply",
  "Tech to stalker", "Push with army").
- `actions`: a list of EXACTLY 5 action names taken from VALID_ACTIONS
  below. Order matters; later actions see the resource state left by
  earlier ones.

Every action name MUST appear verbatim in VALID_ACTIONS — case matters,
hyphenation matters, no spaces inside the verb (BUILD CYBERNETICSCORE,
not BUILD CYBERNETICS CORE). Unknown names are replaced with EMPTY ACTION
by the env.

### VALID_ACTIONS (Protoss, 72 entries)
TRAIN: TRAIN PROBE, TRAIN ZEALOT, TRAIN ADEPT, TRAIN STALKER, TRAIN SENTRY,
TRAIN HIGHTEMPLAR, TRAIN DARKTEMPLAR, TRAIN VOIDRAY, TRAIN CARRIER,
TRAIN TEMPEST, TRAIN ORACLE, TRAIN PHOENIX, TRAIN MOTHERSHIP, TRAIN OBSERVER,
TRAIN IMMORTAL, TRAIN WARPPRISM, TRAIN COLOSSUS, TRAIN DISRUPTOR,
MORPH ARCHON.
BUILD: BUILD PYLON, BUILD ASSIMILATOR, BUILD NEXUS, BUILD GATEWAY,
BUILD CYBERNETICSCORE, BUILD FORGE, BUILD TWILIGHTCOUNCIL,
BUILD ROBOTICSFACILITY, BUILD STARGATE, BUILD TEMPLARARCHIVE,
BUILD DARKSHRINE, BUILD ROBOTICSBAY, BUILD FLEETBEACON,
BUILD PHOTONCANNON, BUILD SHIELDBATTERY.
RESEARCH: RESEARCH WARPGATERESEARCH, RESEARCH PROTOSSAIRWEAPONSLEVEL1-3,
RESEARCH PROTOSSAIRARMORSLEVEL1-3, RESEARCH ADEPTPIERCINGATTACK,
RESEARCH BLINKTECH, RESEARCH CHARGE, RESEARCH PROTOSSGROUNDWEAPONSLEVEL1-3,
RESEARCH PROTOSSGROUNDARMORSLEVEL1-3, RESEARCH PROTOSSSHIELDSLEVEL1-3,
RESEARCH EXTENDEDTHERMALLANCE, RESEARCH GRAVITICDRIVE,
RESEARCH OBSERVERGRAVITICBOOSTER, RESEARCH PSISTORMTECH,
RESEARCH VOIDRAYSPEEDUPGRADE, RESEARCH PHOENIXRANGEUPGRADE,
RESEARCH TEMPESTGROUNDATTACKUPGRADE.
OTHER: SCOUTING PROBE, SCOUTING OBSERVER, SCOUTING ZEALOT, SCOUTING PHOENIX,
MULTI-ATTACK, MULTI-RETREAT, CHRONOBOOST NEXUS, CHRONOBOOST CYBERNETICSCORE,
CHRONOBOOST TWILIGHTCOUNCIL, CHRONOBOOST STARGATE, CHRONOBOOST FORGE,
EMPTY ACTION.
"""

USER_PROMPT_TEMPLATE = """\
### Mission
{task_description}

### Last actions issued
{last_action}

### Previous game state (snippet)
{prev_state_str}

### Current game state
{cur_state_str}

Reply with the JSON action object — 5 actions, in execution order.
"""


class StarCraftAction(BaseModel):
    reasoning: str = Field(description="Strategic justification for the 5-action chain.")
    current_goal: str = Field(
        description="Short label for the immediate objective (e.g. 'Expand economy')."
    )
    actions: list[str] = Field(
        description="Exactly 5 action names from VALID_ACTIONS, in execution order.",
        min_length=5,
        max_length=5,
    )


VALID_ACTIONS: list[str] = [
    # TRAIN UNIT
    "TRAIN PROBE",
    "TRAIN ZEALOT",
    "TRAIN ADEPT",
    "TRAIN STALKER",
    "TRAIN SENTRY",
    "TRAIN HIGHTEMPLAR",
    "TRAIN DARKTEMPLAR",
    "TRAIN VOIDRAY",
    "TRAIN CARRIER",
    "TRAIN TEMPEST",
    "TRAIN ORACLE",
    "TRAIN PHOENIX",
    "TRAIN MOTHERSHIP",
    "TRAIN OBSERVER",
    "TRAIN IMMORTAL",
    "TRAIN WARPPRISM",
    "TRAIN COLOSSUS",
    "TRAIN DISRUPTOR",
    "MORPH ARCHON",
    # BUILD STRUCTURE
    "BUILD PYLON",
    "BUILD ASSIMILATOR",
    "BUILD NEXUS",
    "BUILD GATEWAY",
    "BUILD CYBERNETICSCORE",
    "BUILD FORGE",
    "BUILD TWILIGHTCOUNCIL",
    "BUILD ROBOTICSFACILITY",
    "BUILD STARGATE",
    "BUILD TEMPLARARCHIVE",
    "BUILD DARKSHRINE",
    "BUILD ROBOTICSBAY",
    "BUILD FLEETBEACON",
    "BUILD PHOTONCANNON",
    "BUILD SHIELDBATTERY",
    # RESEARCH TECHNIQUE
    "RESEARCH WARPGATERESEARCH",
    "RESEARCH PROTOSSAIRWEAPONSLEVEL1",
    "RESEARCH PROTOSSAIRWEAPONSLEVEL2",
    "RESEARCH PROTOSSAIRWEAPONSLEVEL3",
    "RESEARCH PROTOSSAIRARMORSLEVEL1",
    "RESEARCH PROTOSSAIRARMORSLEVEL2",
    "RESEARCH PROTOSSAIRARMORSLEVEL3",
    "RESEARCH ADEPTPIERCINGATTACK",
    "RESEARCH BLINKTECH",
    "RESEARCH CHARGE",
    "RESEARCH PROTOSSGROUNDWEAPONSLEVEL1",
    "RESEARCH PROTOSSGROUNDWEAPONSLEVEL2",
    "RESEARCH PROTOSSGROUNDWEAPONSLEVEL3",
    "RESEARCH PROTOSSGROUNDARMORSLEVEL1",
    "RESEARCH PROTOSSGROUNDARMORSLEVEL2",
    "RESEARCH PROTOSSGROUNDARMORSLEVEL3",
    "RESEARCH PROTOSSSHIELDSLEVEL1",
    "RESEARCH PROTOSSSHIELDSLEVEL2",
    "RESEARCH PROTOSSSHIELDSLEVEL3",
    "RESEARCH EXTENDEDTHERMALLANCE",
    "RESEARCH GRAVITICDRIVE",
    "RESEARCH OBSERVERGRAVITICBOOSTER",
    "RESEARCH PSISTORMTECH",
    "RESEARCH VOIDRAYSPEEDUPGRADE",
    "RESEARCH PHOENIXRANGEUPGRADE",
    "RESEARCH TEMPESTGROUNDATTACKUPGRADE",
    # OTHER ACTION
    "SCOUTING PROBE",
    "SCOUTING OBSERVER",
    "SCOUTING ZEALOT",
    "SCOUTING PHOENIX",
    "MULTI-ATTACK",
    "MULTI-RETREAT",
    "CHRONOBOOST NEXUS",
    "CHRONOBOOST CYBERNETICSCORE",
    "CHRONOBOOST TWILIGHTCOUNCIL",
    "CHRONOBOOST STARGATE",
    "CHRONOBOOST FORGE",
    "EMPTY ACTION",
]
DEFAULT_ACTION = "EMPTY ACTION"
DEFAULT_GOAL = "Defeat the Zerg opponent by sequencing economy → tech → army → engagement."

# StarCraft has no continuous score signal — victory is a binary at the
# end of the match (env step rewards 0 each step, +50 only at Victory).
# Use game_time as the progress metric so the success detector still has
# something monotonic to compare on between turns.
SCORE_PATTERN = None
PROGRESS_PATTERN = r"[Gg]ame[_ ]?time:?\s*(\d+):(\d+)"
PROGRESS_THRESHOLD = 0.0
# Used by the unified agent's progress-stagnation detector. PROGRESS_PATTERN
# above is `game_time`, a free-running counter — useless as a stagnation
# signal. `Supply used` (army+worker count) rises as the agent builds, stays
# flat when the build queue stalls, and resets per episode.
STAGNATION_PATTERN = r"Supply used:?\s*(\d+)"
LIVES_PATTERN = None
SUCCESS_KEYWORDS = ["victory"]
FATAL_KEYWORDS = ["defeat"]

METRIC_FIELDS = [
    "minerals",
    "vespene",
    "supply_cap",
    "supply_used",
    "supply_left",
    "worker_supply",
    "army_supply",
    "game_time",
]

# RTS observations are already structured dicts rendered via
# ``StarCraftObs.to_text()``. No positional context like pokemon's
# (map_name, x, y) — leave the context extractor empty so it no-ops
# rather than mis-parsing the resource summary.
CONTEXT_EXTRACTION_MODE = "dict_fields"
CONTEXT_FIELDS: dict = {"fields": []}


def extract_action(result: StarCraftAction) -> str:
    """Render the 5-action chain in the env's expected format.

    ``star_craft_env.text2action`` matches ``\\d+: <?([^>\\n]+)>?`` per line,
    so the numbered prefix is mandatory and must not be wrapped in angle
    brackets for the first regex group to capture the bare action name.
    """
    return "\n".join(f"{i + 1}: {action}" for i, action in enumerate(result.actions))


def calculate_metrics(game_info: dict) -> dict:
    metrics: dict = {}
    for field in METRIC_FIELDS:
        if field in game_info:
            metrics[field] = game_info[field]
    if "evaluation_score" in game_info:
        metrics["evaluation_score"] = float(game_info["evaluation_score"])
    return metrics


# Per-game self-reflection recommendation.
# StarCraft is real-time with one step / ~10s already; the long-horizon
# dialog gains that justify reflection on pokemon (PR #64) don't apply
# here. Disable by default; opt in via use_self_reflection in the agent
# YAML if a future experiment wants to test it.
RECOMMENDED_USE_SELF_REFLECTION = False
RECOMMENDED_REFLECTION_EVERY = 30
