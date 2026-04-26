"""
MACLA Experiment Progress Tracker (autoresearch-style)

Logs experiments to experiments/results.jsonl and plots progress per game
using Plotly, showing kept vs discarded changes with a running best line.

Usage:
    # Log a new experiment result
    python experiments/macla_progress.py log \
        --game super_mario \
        --score 1500 \
        --steps 100 \
        --status KEEP \
        --description "lower theta: max_theta=0.30, min_theta=0.10"

    # Plot progress for all games
    python experiments/macla_progress.py plot

    # Plot progress for a specific game
    python experiments/macla_progress.py plot --game super_mario
"""
import json
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path

import typer

import plotly.graph_objects as go
from plotly.subplots import make_subplots

EXPERIMENTS_DIR = Path(__file__).parent


def _tag_dir(tag: str | None) -> Path:
    """Get experiments/<tag>/ directory. Creates if needed."""
    if tag:
        d = EXPERIMENTS_DIR / tag.lower().replace(" ", "_")
    else:
        d = EXPERIMENTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_experiment(game: str, score: float, steps: int, status: str, description: str,
                   wandb_url: str = "", notes: str = "", game_score: float = 0.0,
                   runtime_min: float = 0.0, tags: list[str] | None = None):
    """Append an experiment result to experiments/<tag>/results.jsonl."""
    tag = tags[0] if tags else None
    experiments = load_results(tag=tag)
    game_experiments = [e for e in experiments if e["game"] == game]
    experiment_num = len(game_experiments)

    entry = {
        "experiment": experiment_num,
        "game": game,
        "evaluation_score": score,
        "game_score": game_score,
        "steps": steps,
        "runtime_min": runtime_min,
        "status": status.upper(),
        "description": description,
        "notes": notes,
        "tags": tags or [],
        "wandb_url": wandb_url,
        "timestamp": datetime.now().isoformat(),
    }

    results_file = _tag_dir(tag) / "results.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"Logged to {results_file}: #{experiment_num} {game} score={score} [{status}] {description}")


def load_results(tag: str | None = None) -> list[dict]:
    """Load experiment results from experiments/<tag>/results.jsonl."""
    results_file = _tag_dir(tag) / "results.jsonl"
    if not results_file.exists():
        return []
    results = []
    for line in results_file.read_text().strip().split("\n"):
        if line:
            results.append(json.loads(line))
    return results


GAME_LOG_DIR = Path(__file__).parent.parent / "game_logs"
ALL_GAMES = ["super_mario", "twenty_fourty_eight", "pokemon_red"]


def normalize_eval_score(game: str, eval_score: float, game_score: float) -> float:
    """Normalize evaluation scores to 0-100 scale for cross-game comparison.

    Server returns different scales per game:
    - Mario: already 0-100 (x_pos progress %)
    - 2048: 0-1 fraction (game_score/20000) → multiply by 100
    - Pokemon: raw flag count (0-7) → (flags/7)*100
    """
    if game == "twenty_fourty_eight":
        # Server returns fraction; also compute from game_score as fallback
        if eval_score < 1.0:
            return eval_score * 100
        return min(eval_score, 100.0)
    elif game == "pokemon_red":
        # Server returns raw flag count
        if eval_score <= 7:
            return (eval_score / 7) * 100
        return min(eval_score, 100.0)
    # Mario: already 0-100
    return eval_score


def extract_run_results(run_id: str, games: list[str] | None = None) -> dict[str, dict]:
    """Parse game_logs/<game>/<run_id>/game_states.jsonl for final scores.

    Returns dict[game] -> {evaluation_score, game_score, steps, episodes, max_eval}.
    Scores are normalized to 0-100 scale.
    """
    games = games or ALL_GAMES
    results = {}
    for game in games:
        states_file = GAME_LOG_DIR / game / run_id / "game_states.jsonl"
        if not states_file.exists():
            continue
        lines = states_file.read_text().strip().split("\n")
        if not lines or not lines[0]:
            continue
        entries = [json.loads(l) for l in lines if l]
        last = entries[-1]
        # Count episodes (iteration resets to 1 at episode start)
        episodes = sum(1 for i, e in enumerate(entries) if i > 0 and e["iteration"] <= entries[i - 1]["iteration"])
        # Normalize to 0-100
        max_eval_raw = max(e["evaluation_score"] for e in entries)
        max_game_score = max(e.get("game_score", 0) for e in entries)
        max_eval = normalize_eval_score(game, max_eval_raw, max_game_score)
        results[game] = {
            "evaluation_score": normalize_eval_score(game, last["evaluation_score"], last.get("game_score", 0)),
            "game_score": last.get("game_score", 0),
            "steps": len(entries),
            "episodes": episodes,
            "max_eval": max_eval,
        }
    return results


def _read_agent_models(games: list[str], config_type: str) -> dict[str, str]:
    """Return {game: model_name} from configs/<game>/agent/<config_type>.yaml.
    Used to add a per-label model snippet so reviewers can see at a glance
    which backend each dot was produced by.
    """
    import yaml as _yaml
    cfgs_root = Path(__file__).resolve().parent.parent / "configs"
    out: dict[str, str] = {}
    for g in games:
        p = cfgs_root / g / "agent" / f"{config_type}.yaml"
        if not p.exists():
            continue
        try:
            cfg = _yaml.safe_load(p.read_text()) or {}
        except Exception:
            continue
        m = cfg.get("model")
        if m:
            # Truncate provider prefix for compactness; keep readable shape
            short = m.split("/")[-1] if "/" in m else m
            out[g] = short
    return out


def _read_agent_metadata(games: list[str], config_type: str) -> str:
    """Read configs/<game>/agent/<config_type>.yaml for each game and build a
    HTML-ish multi-line metadata block summarising model, backend, and MACLA params.
    Returns empty string if no config can be read.
    """
    import yaml as _yaml
    cfgs_root = Path(__file__).resolve().parent.parent / "configs"
    blocks = []
    for g in games:
        p = cfgs_root / g / "agent" / f"{config_type}.yaml"
        if not p.exists():
            continue
        try:
            cfg = _yaml.safe_load(p.read_text()) or {}
        except Exception:
            continue
        head_bits = []
        if cfg.get("model"):
            head_bits.append(f"<b>{cfg['model']}</b>")
        if cfg.get("server_type"):
            head_bits.append(cfg["server_type"])
        if cfg.get("temperature") is not None:
            head_bits.append(f"T={cfg['temperature']}")
        if cfg.get("max_tokens"):
            head_bits.append(f"max_tok={cfg['max_tokens']}")
        head = " · ".join(head_bits)
        macla_keys = ["macla_theta_base", "macla_max_theta", "macla_min_theta",
                      "macla_theta_decay", "macla_warmup_steps"]
        macla_bits = [f"{k.replace('macla_', '')}={cfg[k]}" for k in macla_keys if k in cfg]
        macla_line = ", ".join(macla_bits)
        blocks.append(
            f"<b>{g.replace('_', ' ').title()}</b> ({config_type}.yaml)<br>"
            f"&nbsp;&nbsp;{head}<br>"
            f"&nbsp;&nbsp;<span style='color:#666'>{macla_line}</span>"
        )
    return "<br>".join(blocks)


def plot_progress(filter_game: str | None = None, tag: str | None = None,
                  config_type: str | None = None):
    """Plot autoresearch-style progress chart per game using Plotly.

    Args:
        filter_game: Only show this game
        tag: Load from experiments/<tag>/, use as plot title + output dir
        config_type: If set, read configs/<game>/agent/<config_type>.yaml and
            embed model/backend/MACLA params as a metadata annotation.
    """
    results = load_results(tag=tag)

    plot_title = f"{tag} Experiment Progress" if tag else "Experiment Progress"
    if not results:
        print("No results yet. Use 'log' to add experiments.")
        return

    games = sorted(set(r["game"] for r in results))
    if filter_game:
        games = [g for g in games if g == filter_game]

    n_games = len(games)

    # Per-game subtitle with experiment count and kept count
    subtitles = []
    for g in games:
        gr = [r for r in results if r["game"] == g]
        n_exp = len(gr)
        n_kept = sum(1 for r in gr if r["status"] == "KEEP")
        total_runtime = sum(r.get("runtime_min", 0) for r in gr)
        game_title = g.replace("_", " ").title()
        runtime_str = f", {total_runtime:.0f}min total" if total_runtime else ""
        subtitles.append(f"{game_title} — {n_exp} Experiments, {n_kept} Kept Improvements{runtime_str}")

    fig = make_subplots(
        rows=n_games, cols=1,
        subplot_titles=subtitles,
        vertical_spacing=0.12,
    )

    # Track indices of per-experiment label annotations for the toggle widget.
    # Plotly stores subplot titles as annotations first; later calls to
    # fig.add_annotation append after them. We snapshot len() before the loop
    # and after each add_annotation to record exactly which indices are labels.
    label_annotation_indices: list[int] = []
    game_models = _read_agent_models(games, config_type) if config_type else {}

    for i, game in enumerate(games, 1):
        game_results = [r for r in results if r["game"] == game]
        game_results.sort(key=lambda x: x["experiment"])

        # Renumber sequentially (0, 1, 2, ...) to remove gaps from filtered-out experiments
        xs = list(range(len(game_results)))
        ys = [r["evaluation_score"] for r in game_results]
        statuses = [r["status"] for r in game_results]
        descriptions = [r["description"] for r in game_results]
        wandb_urls = [r.get("wandb_url", "") for r in game_results]
        notes_list = [r.get("notes", "") for r in game_results]
        game_scores = [r.get("game_score", 0) for r in game_results]
        steps_list = [r.get("steps", 0) for r in game_results]
        runtimes = [r.get("runtime_min", 0) for r in game_results]

        def _hover(desc, url, notes="", game_score=0, steps=0, runtime=0):
            """Build hover text with game_score, steps, runtime, notes, and W&B link."""
            h = f"<b>{desc}</b>"
            h += f"<br>Game Score: {game_score} | Steps: {steps}"
            if runtime:
                h += f" | Runtime: {runtime:.0f}min"
            if notes:
                h += f"<br><i>{notes}</i>"
            if url:
                h += f"<br><a href='{url}' target='_blank'>W&B Run</a>"
            return h

        def _custom_data(statuses_filter):
            """Return customdata (wandb_urls) for filtered entries."""
            return [u for u, s in zip(wandb_urls, statuses) if s in statuses_filter]

        exp_nums = [r["experiment"] for r in game_results]

        model_for_game = game_models.get(game)

        def _label(exp_num, desc, notes, runtime=0, model=model_for_game):
            """Full readable label: Exp N (runtime) + description + model + outcome."""
            header = f"<b>E{exp_num}</b>"
            if runtime:
                header += f" [{int(runtime)}min]"
            lines = [header]
            lines.append(desc)
            if model:
                lines.append(f"<span style='color:#666;font-size:7px'>model: {model}</span>")
            if notes:
                outcome = notes.split(".")[0]
                lines.append(outcome)
            return "<br>".join(lines)

        # Plot markers (no text) + add annotations for labels
        status_config = {
            "DISCARD": {"color": "#cccccc", "size": 10, "opacity": 0.7, "line_color": "#999", "symbol": "circle", "text_color": "#777"},
            "KEEP": {"color": "#2ecc71", "size": 12, "opacity": 1.0, "line_color": "black", "symbol": "circle", "text_color": "#1a7a3a"},
            "BASELINE": {"color": "#2ecc71", "size": 12, "opacity": 1.0, "line_color": "black", "symbol": "circle", "text_color": "#1a7a3a"},
            "RUNNING": {"color": "#f39c12", "size": 10, "opacity": 1.0, "line_color": "#c27d0e", "symbol": "diamond", "text_color": "#c27d0e"},
            "EARLY_KILL": {"color": "#e74c3c", "size": 10, "opacity": 0.8, "line_color": "#c0392b", "symbol": "x", "text_color": "#c0392b"},
            "CRASH": {"color": "#e74c3c", "size": 10, "opacity": 0.8, "line_color": "#c0392b", "symbol": "x", "text_color": "#c0392b"},
        }
        legend_added = {"disc": False, "kept": False, "run": False, "kill": False}

        for j, (x, y, s, en, d, n, u, gs, st, rt) in enumerate(zip(
            xs, ys, statuses, exp_nums, descriptions, notes_list, wandb_urls, game_scores, steps_list, runtimes
        )):
            cfg = status_config.get(s, status_config["DISCARD"])
            legend_key = "kill" if s in ("EARLY_KILL", "CRASH") else ("disc" if s == "DISCARD" else ("run" if s == "RUNNING" else "kept"))
            legend_name = {"disc": "Discarded", "kept": "Kept", "run": "Running", "kill": "Killed/Crashed"}[legend_key]
            show_legend = (i == 1) and not legend_added[legend_key]
            legend_added[legend_key] = True

            hover = _hover(d, u, n, gs, st, rt)
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers",
                marker=dict(color=cfg["color"], size=cfg["size"], opacity=cfg["opacity"],
                            line=dict(width=1, color=cfg["line_color"]), symbol=cfg["symbol"]),
                name=legend_name, legendgroup=legend_key, showlegend=show_legend,
                customdata=[u],
                hovertext=[hover], hovertemplate="%{hovertext}<br>Score: %{y}<extra>" + legend_name + "</extra>",
            ), row=i, col=1)

            # Add annotation with full label text, alternating positions
            label = _label(en, d, n, rt)
            y_shift = 30 if j % 2 == 0 else -30
            xref = f"x{i}" if i > 1 else "x"
            yref = f"y{i}" if i > 1 else "y"
            label_annotation_indices.append(len(fig.layout.annotations))
            fig.add_annotation(
                x=x, y=y, xref=xref, yref=yref,
                text=label, showarrow=True, arrowhead=2, arrowsize=0.8,
                ax=40, ay=y_shift - (30 if j % 2 == 0 else -30),
                font=dict(size=8, color=cfg["text_color"]),
                align="left", bgcolor="rgba(255,255,255,0.85)",
                bordercolor=cfg["color"], borderwidth=1, borderpad=3,
            )

        # Running best line (step function through kept experiments)
        kept_x = [x for x, s in zip(xs, statuses) if s in ("KEEP", "BASELINE")]
        kept_y = [y for y, s in zip(ys, statuses) if s in ("KEEP", "BASELINE")]
        if kept_x:
            running_best_x = []
            running_best_y = []
            best_so_far = kept_y[0]
            for x, y in zip(kept_x, kept_y):
                best_so_far = max(best_so_far, y)
                running_best_x.append(x)
                running_best_y.append(best_so_far)
            fig.add_trace(go.Scatter(
                x=running_best_x, y=running_best_y, mode="lines",
                line=dict(color="#27ae60", width=2, shape="hv"),
                name="Running best", legendgroup="best", showlegend=(i == 1),
                hoverinfo="skip",
            ), row=i, col=1)

        fig.update_yaxes(title_text="Evaluation Score (higher is better)", rangemode="tozero", row=i, col=1)
        # Only show tick marks for experiments that have data (skip gaps)
        fig.update_xaxes(title_text="Experiment #", tickvals=sorted(set(xs)), dtick=1, row=i, col=1)

    # Per-game eval score formula + W&B link as subplot y-axis title suffix
    EVAL_FORMULAS = {
        "pokemon_red": "eval = (flags/7)×100 — 7 storyline flags",
        "super_mario": "eval = (x_pos−40)/(3161−40)×100 — World 1-1 progress",
        "twenty_fourty_eight": "eval = min(score/20000×100, 100) — progress to 2048 tile",
    }
    WANDB_PROJECTS = {
        "pokemon_red": "https://wandb.ai/chaleong/orak-pokemon-red",
        "super_mario": "https://wandb.ai/chaleong/orak-super-mario",
        "twenty_fourty_eight": "https://wandb.ai/chaleong/orak-2048",
    }
    # Update subplot titles to include formula
    for idx, game in enumerate(games):
        if game in EVAL_FORMULAS:
            # Subplot titles are stored as annotations by plotly
            for ann in fig.layout.annotations:
                game_title = game.replace("_", " ").title()
                if game_title in ann.text:
                    wandb_link = f"<a href='{WANDB_PROJECTS[game]}'>[W&B]</a>"
                    ann.text = f"{ann.text}<br><span style='font-size:11px;color:#666'>{EVAL_FORMULAS[game]} {wandb_link}</span>"

    # Top margin / title — extra room when we add a config metadata box
    metadata = _read_agent_metadata(games, config_type) if config_type else ""
    top_margin = 220 if metadata else 80

    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=20)),
        height=650 * n_games + (140 if metadata else 0),
        margin=dict(t=top_margin),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )

    if metadata:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.0, y=1.0,
            xanchor="left", yanchor="bottom",
            text=metadata, showarrow=False, align="left",
            font=dict(size=11, color="#222"),
            bgcolor="rgba(245,247,250,0.95)",
            bordercolor="#dde",
            borderwidth=1, borderpad=8,
            xshift=0, yshift=40,
        )

    # Save to experiments/<tag>/progress.html with a label-toggle switch
    out_dir = _tag_dir(tag)
    output_path = out_dir / "progress.html"
    try:
        from experiments._chart_widgets import plotly_label_toggle
        post_script = plotly_label_toggle(
            label_indices=label_annotation_indices,
            n_traces=len(fig.data),
            label="labels",
            position="top-right",
            default_on=True,
        )
    except Exception:
        post_script = None
    fig.write_html(str(output_path), post_script=post_script)
    print(f"Saved to {output_path}")

    try:
        png_path = out_dir / "progress.png"
        fig.write_image(str(png_path), width=1800, height=650 * n_games)
        print(f"Saved to {png_path}")
    except Exception:
        pass

    fig.show()


app = typer.Typer(help="MACLA Experiment Progress Tracker (autoresearch-style)")


class Status(str, Enum):
    KEEP = "KEEP"
    DISCARD = "DISCARD"
    BASELINE = "BASELINE"
    RUNNING = "RUNNING"
    CRASH = "CRASH"
    EARLY_KILL = "EARLY_KILL"


@app.command()
def log(
    game: str = typer.Option(..., help="Game name (e.g. super_mario, twenty_fourty_eight, pokemon_red)"),
    score: float = typer.Option(..., help="Evaluation score"),
    status: Status = typer.Option(..., help="Experiment outcome"),
    description: str = typer.Option(..., help="What changed in this experiment"),
    steps: int = typer.Option(0, help="Number of steps completed"),
    game_score: float = typer.Option(0.0, help="In-game score (separate from evaluation score)"),
    runtime_min: float = typer.Option(0.0, help="Runtime in minutes"),
    wandb_url: str = typer.Option("", help="W&B run URL"),
    notes: str = typer.Option("", help="Why kept/discarded, key observations, improvements"),
    tag: str = typer.Option("macla", help="Experiment tag (e.g. macla, unified, baseline). Used for filtering plots."),
):
    """Log an experiment result."""
    log_experiment(game, score, steps, status.value, description, wandb_url, notes, game_score, runtime_min, tags=[tag])


@app.command()
def plot(
    game: str | None = typer.Option(None, help="Filter to a specific game"),
    tag: str | None = typer.Option(None, help="Filter by tag and use as title + filename (e.g. 'macla' → macla.html)"),
):
    """Plot progress chart. Use --tag to filter experiments and generate tag-specific output."""
    plot_progress(game, tag=tag)


if __name__ == "__main__":
    app()
