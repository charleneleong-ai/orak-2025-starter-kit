#!/usr/bin/env bash
# Auto-retry missing TGAER PR1 n=3 rollouts when system has capacity.
# Polls every 5min, launches missing slots when active count < 16,
# caps at 3 attempts per slot, exits when queue is empty.

set -u
LOG=/workspace/orak-futile-detector/logs/tgaer_auto_retry_$(date -u +%Y%m%dT%H%M%SZ).log
MAX_ACTIVE=16
MAX_ATTEMPTS=3
POLL_INTERVAL=300
LAUNCH_STAGGER=45

# slot format: "side|game|n"
SLOTS=(
  "baseline|star_craft|n3"
  "detector|twenty_fourty_eight|n2"
  "detector|twenty_fourty_eight|n3"
  "detector|star_craft|n1"
)
declare -A ATTEMPTS
for s in "${SLOTS[@]}"; do ATTEMPTS[$s]=0; done

log() { echo "[$(date -u +%H:%M:%S)] $1" | tee -a "$LOG"; }

worktree_for() {
  case "$1" in
    baseline) echo "orak-master-baselines" ;;
    detector) echo "orak-futile-detector" ;;
  esac
}

is_satisfied() {
  local side=$1 game=$2 n=$3
  local wt=$(worktree_for "$side")
  # Completed?
  if ls /workspace/"$wt"/game_logs/"$game"/tgaer_${side}_${game}_${n}_*/evaluation_summary.json 2>/dev/null | head -1 > /dev/null; then
    return 0
  fi
  # Currently running and alive >120s?
  local pids
  pids=$(pgrep -f "tgaer_${side}_${game}_${n}_" 2>/dev/null || true)
  for pid in $pids; do
    local age
    age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    if [ -n "$age" ] && [ "$age" -gt 120 ]; then return 0; fi
  done
  return 1
}

count_active() {
  ps -ef | grep -E "tgaer_(baseline|detector)" | grep -v grep \
    | grep -oP "tgaer_(baseline|detector)_(pokemon_red|super_mario|twenty_fourty_eight|star_craft)_n\d[a-z0-9_]*_\d+T\d+Z" \
    | sort -u | wc -l
}

launch_slot() {
  local side=$1 game=$2 n=$3 attempt=$4
  local wt=$(worktree_for "$side")
  local ts=$(date -u +%Y%m%dT%H%M%SZ)
  cd "/workspace/$wt"
  setsid nohup ./.venv/bin/python run.py -c gemma_26b --local --games "$game" \
    --run-id "tgaer_${side}_${game}_${n}_retry${attempt}_${ts}" \
    -d "TGAER auto-retry ${attempt} for ${side} ${game} ${n}" \
    </dev/null >>logs/tgaer_${side}_${game}_${n}_retry${attempt}_${ts}.log 2>&1 & disown
}

log "auto-retry daemon started (PID=$$, max_active=$MAX_ACTIVE, max_attempts=$MAX_ATTEMPTS, poll=${POLL_INTERVAL}s)"
log "initial queue: ${SLOTS[*]}"

while true; do
  remaining=()
  for slot in "${SLOTS[@]}"; do
    IFS='|' read -r side game n <<< "$slot"
    if is_satisfied "$side" "$game" "$n"; then
      log "✓ $slot satisfied, removing from queue"
      continue
    fi
    if [ "${ATTEMPTS[$slot]}" -ge "$MAX_ATTEMPTS" ]; then
      log "✗ $slot gave up after ${ATTEMPTS[$slot]} attempts"
      continue
    fi
    remaining+=("$slot")
  done

  if [ ${#remaining[@]} -eq 0 ]; then
    log "queue empty, daemon exiting"
    break
  fi

  active=$(count_active)
  log "sweep: ${#remaining[@]} pending, $active active (limit $MAX_ACTIVE)"

  for slot in "${remaining[@]}"; do
    active=$(count_active)
    if [ "$active" -ge "$MAX_ACTIVE" ]; then
      log "  pause: $active active >= limit, will retry in next sweep"
      break
    fi
    IFS='|' read -r side game n <<< "$slot"
    ATTEMPTS[$slot]=$((${ATTEMPTS[$slot]} + 1))
    log "  launch $slot attempt ${ATTEMPTS[$slot]}/${MAX_ATTEMPTS}"
    launch_slot "$side" "$game" "$n" "${ATTEMPTS[$slot]}"
    sleep "$LAUNCH_STAGGER"
  done

  sleep "$POLL_INTERVAL"
done
