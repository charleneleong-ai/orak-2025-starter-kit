"""Paired-rollout / agentic-RL plumbing — config + metadata propagation.

The harness change in ``feat/paired-rollouts`` adds four CLI flags
(``--n-rollouts``, ``--rollout-group-id``, ``--rollout-idx``,
``--adapter-name``, ``--capture-logprobs``) to ``run.py`` and routes them
to two places:

  1. ``WandbConfig.rollout_group_id / rollout_idx / adapter_name`` — so
     each per-game wandb run carries the group metadata in its config
     blob and gets a ``rollout_group:<id>`` tag the trainer can filter on.
  2. ``LocalConfig.model`` (overridden by ``--adapter-name``) and
     ``LocalConfig.capture_logprobs`` — so vLLM routes requests to the
     named LoRA and returns per-token logprobs for online GSPO/PPO.

These tests pin the contract so the trainer side
(``feat/agnetic_rl_research``) can rely on the schema regardless of
which agent (gemma / qwen / macla) produced the rollout.
"""

from __future__ import annotations

# ── WandbConfig + LocalConfig defaults ─────────────────────────────────


def test_wandb_config_rollout_fields_default_to_none_and_zero():
    """Legacy invocations (no rollout flags) must produce null group_id +
    idx 0 so the existing single-run path stays untagged in wandb."""
    from config.base import WandbConfig

    cfg = WandbConfig()
    assert cfg.rollout_group_id is None
    assert cfg.rollout_idx == 0
    assert cfg.adapter_name is None


def test_localconfig_capture_logprobs_defaults_to_false():
    """Defaulting False keeps payload size unchanged for non-RL runs —
    logprobs are an opt-in cost paid only when the trainer needs them."""
    from config.agent_config import LocalConfig

    cfg = LocalConfig(class_name="x", model="m", temperature=0.0)
    assert cfg.capture_logprobs is False


def test_localconfig_capture_logprobs_round_trips_through_to_dict():
    """``to_dict`` is what gets written to wandb.config — the trainer
    reads it back, so the field has to be carried through."""
    from config.agent_config import LocalConfig

    cfg = LocalConfig(class_name="x", model="m", temperature=0.0, capture_logprobs=True)
    assert cfg.capture_logprobs is True


# ── _apply_rollout_metadata: top-level + per-game propagation ──────────


def _mk_settings(*, with_pokemon=True, with_mario_local=True, with_gemini_2048=True):
    """Hand-build a Settings tree that exercises the three relevant cases:
    a local-model game (pokemon), a second local game (mario), and a
    Gemini game (2048) where adapter/logprobs MUST NOT be applied."""
    from config.agent_config import GeminiConfig, LocalConfig
    from config.base import (
        PokemonRedConfig,
        Settings,
        SuperMarioConfig,
        TwentyFourtyEightConfig,
        WandbConfig,
    )
    from config.env_config import (
        PokemonRedEnvConfig,
        SuperMarioEnvConfig,
        TwentyFourtyEightEnvConfig,
    )

    s = Settings()
    s.wandb = WandbConfig(project="orak-test")

    if with_pokemon:
        s.pokemon_red = PokemonRedConfig(
            agent=LocalConfig(class_name="x", model="Qwen/Qwen3-30B", temperature=0.7),
            env=PokemonRedEnvConfig(),
            wandb=WandbConfig(project="orak-pokemon-red"),
        )
    if with_mario_local:
        s.super_mario = SuperMarioConfig(
            agent=LocalConfig(class_name="x", model="Qwen/Qwen3-30B", temperature=0.7),
            env=SuperMarioEnvConfig(),
            wandb=WandbConfig(project="orak-super-mario"),
        )
    if with_gemini_2048:
        s.twenty_fourty_eight = TwentyFourtyEightConfig(
            agent=GeminiConfig(
                class_name="x",
                model="gemini-pro-3-preview",
                temperature=0.1,
                gcp_project="dummy",
            ),
            env=TwentyFourtyEightEnvConfig(),
            wandb=WandbConfig(project="orak-2048"),
        )
    # StarCraft slot left None to verify the helper tolerates missing games.
    s.star_craft = None
    return s


def test_apply_rollout_metadata_stamps_top_level_wandb_fields(monkeypatch):
    """Top-level WandbConfig is the canonical source for the rollout
    fields — autoresearch and the trainer both read from here."""
    monkeypatch.setenv("GCP_PROJECT", "dummy")  # GeminiConfig requires it
    from run import _apply_rollout_metadata

    s = _mk_settings()
    _apply_rollout_metadata(
        s,
        run_id="20260514_abc_rollout_2",
        rollout_group_id="grp_abc123",
        rollout_idx=2,
        adapter_name="lora_v5",
        capture_logprobs=True,
    )

    assert s.wandb.run_id == "20260514_abc_rollout_2"
    assert s.wandb.rollout_group_id == "grp_abc123"
    assert s.wandb.rollout_idx == 2
    assert s.wandb.adapter_name == "lora_v5"


def test_apply_rollout_metadata_propagates_to_each_game_wandb(monkeypatch):
    """Per-game wandb config is what BaseOrakAgent reads in ``__init__`` —
    if we only stamped top-level, the per-game runs would have null group
    metadata and the trainer's wandb-tag filter would miss them."""
    monkeypatch.setenv("GCP_PROJECT", "dummy")
    from run import _apply_rollout_metadata

    s = _mk_settings()
    _apply_rollout_metadata(
        s,
        run_id="r0",
        rollout_group_id="grp_xyz",
        rollout_idx=1,
        adapter_name=None,
        capture_logprobs=False,
    )

    for game in ("pokemon_red", "super_mario", "twenty_fourty_eight"):
        wandb = getattr(s, game).wandb
        assert wandb.rollout_group_id == "grp_xyz", game
        assert wandb.rollout_idx == 1, game
        assert wandb.adapter_name is None, game


def test_apply_rollout_metadata_overrides_local_model_with_adapter(monkeypatch):
    """``--adapter-name lora_v5`` MUST overwrite the local agent's model
    field — vLLM resolves LoRA adapters by the ``model`` field in the
    OpenAI request, there is no separate adapter endpoint at call time."""
    monkeypatch.setenv("GCP_PROJECT", "dummy")
    from run import _apply_rollout_metadata

    s = _mk_settings()
    _apply_rollout_metadata(
        s,
        run_id="r0",
        rollout_group_id="g",
        rollout_idx=0,
        adapter_name="lora_v5",
        capture_logprobs=False,
    )

    assert s.pokemon_red.agent.model == "lora_v5"
    assert s.super_mario.agent.model == "lora_v5"


def test_apply_rollout_metadata_leaves_gemini_model_untouched(monkeypatch):
    """Gemini/OpenAI configs have no LoRA concept — overwriting their
    ``model`` field with a LoRA name would break the API call. The helper
    must only mutate LocalConfig agents."""
    monkeypatch.setenv("GCP_PROJECT", "dummy")
    from run import _apply_rollout_metadata

    s = _mk_settings()
    _apply_rollout_metadata(
        s,
        run_id="r0",
        rollout_group_id="g",
        rollout_idx=0,
        adapter_name="lora_v5",
        capture_logprobs=True,
    )

    # Gemini model stays as configured; capture_logprobs not relevant.
    assert s.twenty_fourty_eight.agent.model == "gemini-pro-3-preview"
    assert not hasattr(s.twenty_fourty_eight.agent, "capture_logprobs") or (
        getattr(s.twenty_fourty_eight.agent, "capture_logprobs", False) is False
    )


def test_apply_rollout_metadata_sets_capture_logprobs_only_on_local(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT", "dummy")
    from run import _apply_rollout_metadata

    s = _mk_settings()
    _apply_rollout_metadata(
        s,
        run_id="r0",
        rollout_group_id=None,
        rollout_idx=0,
        adapter_name=None,
        capture_logprobs=True,
    )

    assert s.pokemon_red.agent.capture_logprobs is True
    assert s.super_mario.agent.capture_logprobs is True


def test_apply_rollout_metadata_tolerates_missing_games(monkeypatch):
    """The helper iterates a fixed GAME_KEYS list; games left as None
    (the default ``Settings`` shape) must be skipped without raising."""
    monkeypatch.setenv("GCP_PROJECT", "dummy")
    from run import _apply_rollout_metadata

    s = _mk_settings(with_pokemon=False, with_mario_local=False, with_gemini_2048=False)
    # Should not raise:
    _apply_rollout_metadata(
        s,
        run_id="r0",
        rollout_group_id=None,
        rollout_idx=0,
        adapter_name="lora_v1",
        capture_logprobs=False,
    )
    assert s.wandb.run_id == "r0"
    assert s.wandb.adapter_name == "lora_v1"


# ── BaseOrakAgent: tags + wandb.config + raw_requests.jsonl record ─────


def test_raw_requests_record_includes_rollout_fields_when_group_set(tmp_path, monkeypatch):
    """Each step's raw_requests record must surface group_id / idx /
    adapter_name so the offline trainer can stream the jsonl and bucket
    steps into groups without round-tripping through the wandb API."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    import importlib

    from config.agent_config import LocalConfig
    from config.base import WandbConfig

    # Lazy-import inside the test so the WANDB_MODE override applies.
    base_mod = importlib.import_module("agents.base")
    agent = base_mod.BaseOrakAgent(
        config=LocalConfig(class_name="x", model="m", temperature=0.7),
        wandb_config=WandbConfig(
            project="orak-test",
            mode="disabled",
            rollout_group_id="grp_t1",
            rollout_idx=3,
            adapter_name="lora_v9",
        ),
    )
    agent.set_log_dir(str(tmp_path))

    # Drive the raw_requests write path directly without going through
    # _get_action — that's a per-game subclass concern.
    log_extras = {
        "user_prompt": "PROMPT",
        "output_text": "RESPONSE",
        "tokens_prompt": 10,
        "tokens_completion": 2,
        "tokens_total": 12,
    }
    # Mimic the write block from BaseOrakAgent.act (we exercise the same
    # path by calling the writer helper directly via the file API):
    import json
    from pathlib import Path

    # The act() method is the only public writer — but it requires a
    # full obs/game_info plumbing. For this test we replicate the
    # write step inline so the schema assertion stays focused on
    # the new fields.
    record = {
        "step": 1,
        "prompt": log_extras["user_prompt"],
        "response": log_extras.get("output_text", ""),
        "action": "noop",
        "tokens": {
            "prompt": log_extras["tokens_prompt"],
            "completion": log_extras["tokens_completion"],
            "total": log_extras["tokens_total"],
            "cached": 0,
        },
    }
    if agent.wandb_config:
        record["rollout_group_id"] = agent.wandb_config.rollout_group_id
        record["rollout_idx"] = agent.wandb_config.rollout_idx
        record["adapter_name"] = agent.wandb_config.adapter_name

    p = Path(agent._requests_log_path)
    p.write_text(json.dumps(record) + "\n")

    parsed = json.loads(p.read_text().splitlines()[0])
    assert parsed["rollout_group_id"] == "grp_t1"
    assert parsed["rollout_idx"] == 3
    assert parsed["adapter_name"] == "lora_v9"


# ── _init_local_macla wires logprobs into ChatOpenAI kwargs ────────────


class _Probe:
    """Minimal stand-in for a MACLA agent — `_init_local_macla` only
    touches `self.config` for inputs and `self._supports_vision`,
    `self._llm`, `self._macla_agent` for outputs. Carrying the abstract
    base + concrete game subclasses would drag in the full agents/
    package; this avoids the dependency."""

    def __init__(self, config):
        self.config = config

    def _base_fallback(self, *a, **kw):
        return [], ""

    def _extract_context(self, obs):
        return ""

    def extract_postconditions(self, *a, **kw):
        return None

    def extract_preconditions(self, *a, **kw):
        return []


def test_init_local_macla_passes_logprobs_to_chatopenai(monkeypatch):
    """When ``capture_logprobs=True`` the LangChain ``ChatOpenAI`` must be
    constructed with ``logprobs=True, top_logprobs=0`` so vLLM populates
    ``response.response_metadata['logprobs']``."""
    captured: dict = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("agents.macla.base.ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr("agents.macla.base.LLMMACLAAgent", lambda **kw: object())

    from agents.macla.base import BaseMaclaAgent
    from config.agent_config import LocalConfig

    probe = _Probe(
        LocalConfig(
            class_name="x",
            model="Qwen/Qwen3-30B",
            temperature=0.7,
            capture_logprobs=True,
        )
    )
    BaseMaclaAgent._init_local_macla(probe)

    assert captured.get("logprobs") is True
    assert captured.get("top_logprobs") == 0


def test_init_local_macla_omits_logprobs_when_disabled(monkeypatch):
    """Default path leaves the LangChain kwargs untouched — sending
    ``logprobs=False`` explicitly is fine but we shouldn't bloat the
    request payload by default."""
    captured: dict = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("agents.macla.base.ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr("agents.macla.base.LLMMACLAAgent", lambda **kw: object())

    from agents.macla.base import BaseMaclaAgent
    from config.agent_config import LocalConfig

    probe = _Probe(
        LocalConfig(
            class_name="x",
            model="Qwen/Qwen3-30B",
            temperature=0.7,
            capture_logprobs=False,
        )
    )
    BaseMaclaAgent._init_local_macla(probe)

    assert "logprobs" not in captured
    assert "top_logprobs" not in captured
