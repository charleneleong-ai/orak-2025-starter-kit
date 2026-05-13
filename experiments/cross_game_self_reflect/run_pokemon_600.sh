#!/usr/bin/env bash
# Pokemon Stage D + self-reflection at 600 steps (PR #64 follow-up #3).
#
# Cross-game retest found pokemon Stage D + reflect tied Stage D baseline
# at 57.14% (4/7) but with qualitatively deeper trajectory (banked
# Charmander naming + trainer-battle progression — milestones Stage D
# baseline typically misses). The 14.29-pp-per-milestone metric is too
# coarse to see the difference at 300 steps.
#
# This run extends to 600 steps to see if the deeper trajectory crosses
# milestone 5 (gym boss = +14.29 pp → 71.43%).
#
# Comparison baselines:
#   Stage D (PR #31):                57.14% (300st)
#   Stage D + reflect (PR #64):      57.14% (300st, deeper trajectory)
#   Stage D++ (PR #31):              71.43% (600st, no reflect)
#   This run (Stage D + reflect 600): target ≥ 71.43%
#
# Adapter recommendations on master after PR #64: pokemon's gemma_26b.yaml
# has no explicit use_self_reflection → adapter recommendation wins →
# reflection ON every 10 steps.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/gemma_26b.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

grep -q "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" "$AGENT_CFG" || {
    echo "FATAL: $AGENT_CFG not AWQ. Current: $(grep '^model:' $AGENT_CFG)"; exit 1; }
# YAML should NOT explicitly set use_self_reflection → adapter wins
grep -q "^use_self_reflection:" "$AGENT_CFG" && {
    echo "WARN: explicit use_self_reflection in $AGENT_CFG bypasses adapter recommendation"; }

restore() {
    echo "[restore] max_steps → 300 (default)"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
}
trap restore EXIT
sed -i 's/^max_steps: .*/max_steps: 600/' "$ENV_CFG"

stage="pokemon_d_reflect_600"
run_id="pr64_${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
game_logs="$GAME_DATA_DIR/game_logs/pokemon_red/$run_id"
started=$(date +%s)

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] $stage (PR #64 follow-up #3)"
echo "  Stage D + self-reflection (adapter default) at 600 steps"
echo "  target: ≥ Stage D++ (71.43%)"
echo "  run_id=$run_id"
echo "================================================================"

if ! uv run python run.py \
    -c gemma_26b \
    --local --games pokemon_red \
    --run-id "$run_id" \
    -d "Pokemon Stage D + self-reflect at 600 steps — does the deeper trajectory bank milestone 5?"; then
    echo "[FAIL] run.py exited non-zero"
    exit 1
fi

actual_dir="$GAME_DATA_DIR/pokemon_red/$run_id"
[ -d "$actual_dir" ] && ln -sfn "$actual_dir" "$game_logs"

elapsed=$(( ( $(date +%s) - started ) / 60 ))
echo "[$stage] runtime=${elapsed}min"
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] $stage DONE"
echo "================================================================"
