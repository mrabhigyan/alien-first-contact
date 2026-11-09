"""Turn a logged run into results: a success-rate chart and the emerged dictionary.

    python analyze.py              # latest run
    python analyze.py --run 4      # a specific run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt

from metrics import topsim_test
from world import make_objects

# Reference palette: one series -> slot 1 blue; text in ink tokens.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def load_run(conn: sqlite3.Connection, run_id: int | None) -> tuple[dict, list[tuple]]:
    if run_id is None:
        (run_id,) = conn.execute("SELECT MAX(run_id) FROM runs").fetchone()
        if run_id is None:
            sys.exit("No runs in the database yet.")
    row = conn.execute(
        "SELECT run_id, level, n_candidates, model, alphabet FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()
    if row is None:
        sys.exit(f"Run {run_id} not found.")
    run = dict(zip(("run_id", "level", "n_candidates", "model", "alphabet"), row))
    rounds = conn.execute(
        "SELECT round_no, target, message, success FROM rounds WHERE run_id=? ORDER BY round_no",
        (run_id,),
    ).fetchall()
    if not rounds:
        sys.exit(f"Run {run_id} has no rounds.")
    return run, rounds


def rolling_rate(successes: list[int], window: int) -> list[float]:
    """Success % over the last `window` rounds (fewer at the start)."""
    out = []
    for i in range(len(successes)):
        chunk = successes[max(0, i - window + 1): i + 1]
        out.append(100 * sum(chunk) / len(chunk))
    return out


def plot_success(run: dict, rounds: list[tuple], window: int, path: Path) -> None:
    xs = [r[0] for r in rounds]
    ys = rolling_rate([r[3] for r in rounds], window)
    baseline = 100 / run["n_candidates"]
    final = sum(r[3] for r in rounds[-window:]) * 100 / len(rounds[-window:])

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(baseline, color=INK_SOFT, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(xs[-1], baseline + 2.5, f"random guessing ({baseline:.0f}%)",
            ha="right", va="bottom", fontsize=9, color=INK_SOFT)
    ax.plot(xs, ys, color=SERIES, linewidth=2, solid_capstyle="round")

    ax.set_ylim(0, 105)
    ax.set_xlim(1, xs[-1])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Round", color=INK_SOFT, fontsize=9)
    ax.set_ylabel(f"Success, last {window} rounds", color=INK_SOFT, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    level = {1: "colors", 2: "color + shape", 3: "grid cells"}.get(run["level"], "?")
    fig.suptitle(f"Agents reach {final:.0f}% success, up from {baseline:.0f}% guessing",
                 x=0.07, ha="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_title(f"Run {run['run_id']} · level {run['level']} ({level}) · "
                 f"{run['n_candidates']} candidates · {run['model']}",
                 loc="left", fontsize=9, color=INK_SOFT)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def dictionary(rounds: list[tuple], skip: int) -> dict[str, Counter]:
    """For each target, how often each message was used (after the first `skip` rounds)."""
    usage: dict[str, Counter] = defaultdict(Counter)
    for round_no, target, message, _ in rounds:
        if round_no > skip:
            usage[target][" ".join(json.loads(message))] += 1
    return usage


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Analyze a logged run")
    p.add_argument("--run", type=int, default=None, help="run id (default: latest)")
    p.add_argument("--db", default="runs.db")
    p.add_argument("--window", type=int, default=10, help="rounds per rolling average")
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)

    conn = sqlite3.connect(a.db)
    run, rounds = load_run(conn, a.run)
    chart = Path(a.out) / f"success_run{run['run_id']}.png"
    plot_success(run, rounds, a.window, chart)
    print(f"Chart saved: {chart}")

    skip = len(rounds) // 2
    print(f"\nDictionary (rounds {skip + 1}-{len(rounds)}): target -> message (times used)")
    for target, counts in sorted(dictionary(rounds, skip).items()):
        top, n = counts.most_common(1)[0]
        share = 100 * n / sum(counts.values())
        print(f"  {target:<18} {top:<8} {n:>3}x  ({share:.0f}% consistent)")

    by_name = {t.name: t for t in make_objects(run["level"])}
    lexicon = {by_name[target]: counts.most_common(1)[0][0].split()
               for target, counts in dictionary(rounds, skip).items()}
    result = topsim_test(lexicon)
    print("\nCompositionality (topographic similarity, 0 = none, 1 = perfect):")
    if result is None:
        print("  not defined here (needs meanings with 2+ attributes, e.g. level 2 or 3)")
    else:
        print(f"  topsim = {result.score:.2f}   shuffled messages = {result.shuffled_mean:.2f}"
              f"   p = {result.p_value:.3f}   ({result.n_meanings} meanings)")


if __name__ == "__main__":
    main()
