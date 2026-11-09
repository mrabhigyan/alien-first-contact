"""Turn a logged run into results: a success chart, the emerged dictionary, structure scores,
and experiment-specific reports (held-out test, newcomer, two tribes).

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
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt

from main import open_db
from metrics import topsim_test
from world import make_objects

# Reference palette: one series -> slot 1 blue; text in ink tokens.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"

LEVEL_LABEL = {1: "colors", 2: "color + shape", 3: "grid cells"}


class Row(NamedTuple):
    round_no: int
    target: str
    message: str        # JSON list, as stored
    success: int
    phase: str
    speaker_id: str
    listener_id: str
    picked: str | None


RUN_FIELDS = ("run_id", "level", "n_candidates", "model", "alphabet", "experiment", "feedback",
              "held_out", "notes_every", "memory_window", "input_tokens", "output_tokens")


def load_run(conn: sqlite3.Connection, run_id: int | None) -> tuple[dict, list[Row]]:
    if run_id is None:
        (run_id,) = conn.execute("SELECT MAX(run_id) FROM runs").fetchone()
        if run_id is None:
            sys.exit("No runs in the database yet.")
    row = conn.execute(f"SELECT {', '.join(RUN_FIELDS)} FROM runs WHERE run_id=?",
                       (run_id,)).fetchone()
    if row is None:
        sys.exit(f"Run {run_id} not found.")
    run = dict(zip(RUN_FIELDS, row, strict=True))
    run["held_out"] = json.loads(run["held_out"] or "[]")
    rounds = [Row(*r) for r in conn.execute(
        "SELECT round_no, target, message, success, phase, speaker_id, listener_id, picked"
        " FROM rounds WHERE run_id=? ORDER BY round_no", (run_id,))]
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


def phases_of(rounds: list[Row]) -> list[tuple[str, list[Row]]]:
    out: dict[str, list[Row]] = {}
    for r in rounds:
        out.setdefault(r.phase, []).append(r)
    return list(out.items())


def rate(rows) -> float:
    return 100 * sum(r[3] for r in rows) / len(rows) if rows else 0.0


# ---------------------------------------------------------------- chart


def plot_success(run: dict, rounds: list[Row], window: int, path: Path) -> None:
    baseline = 100 / run["n_candidates"]
    phases = phases_of(rounds)
    final_rows = phases[-1][1][-window:]
    final = rate(final_rows)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(baseline, color=INK_SOFT, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(rounds[-1].round_no, baseline + 2.5, f"random guessing ({baseline:.0f}%)",
            ha="right", va="bottom", fontsize=9, color=INK_SOFT)

    # Rolling average restarts in each phase, so a phase change shows as a fresh start.
    for i, (name, rows) in enumerate(phases):
        xs = [r.round_no for r in rows]
        ax.plot(xs, rolling_rate([r.success for r in rows], window), color=SERIES,
                linewidth=2, solid_capstyle="round")
        if len(phases) > 1:
            if i > 0:
                ax.axvline(xs[0] - 0.5, color=INK_SOFT, linewidth=0.8, linestyle=":")
            ax.text(xs[0] + 1, 102, name, fontsize=8.5, color=INK_SOFT, va="bottom")

    ax.set_ylim(0, 108)
    ax.set_xlim(1, rounds[-1].round_no)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Round", color=INK_SOFT, fontsize=9)
    ax.set_ylabel(f"Success, last {window} rounds", color=INK_SOFT, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    level = LEVEL_LABEL.get(run["level"], "?")
    extra = "" if run["experiment"] == "standard" else f" · {run['experiment']}"
    extra += " · full feedback" if run["feedback"] == "full" else ""
    ending = f"ends at {final:.0f}% success" if len(phases) > 1 else f"reach {final:.0f}% success"
    fig.suptitle(f"Agents {ending}, vs {baseline:.0f}% guessing",
                 x=0.07, ha="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_title(f"Run {run['run_id']} · level {run['level']} ({level}){extra} · "
                 f"{run['model']}", loc="left", fontsize=9, color=INK_SOFT, pad=14)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- language


def dictionary(rounds, skip: int) -> dict[str, Counter]:
    """For each target, how often each message was used (rounds numbered above `skip`).
    Accepts any rows whose first three fields are round_no, target, message (JSON)."""
    usage: dict[str, Counter] = defaultdict(Counter)
    for r in rounds:
        if r[0] > skip:
            usage[r[1]][" ".join(json.loads(r[2]))] += 1
    return usage


def lexicon_of(rows: list[Row], level: int) -> dict:
    """Dominant message per target in the second half of these rows -> {Thing: [symbols]}."""
    if not rows:
        return {}
    skip = rows[len(rows) // 2].round_no - 1
    by_name = {t.name: t for t in make_objects(level)}
    return {by_name[t]: c.most_common(1)[0][0].split()
            for t, c in dictionary(rows, skip).items()}


def print_dictionary(rows: list[Row], title: str) -> None:
    skip = rows[len(rows) // 2].round_no - 1
    print(f"\n{title} (rounds {skip + 1}-{rows[-1].round_no}): target -> message (times used)")
    for target, counts in sorted(dictionary(rows, skip).items()):
        top, n = counts.most_common(1)[0]
        share = 100 * n / sum(counts.values())
        print(f"  {target:<18} {top:<8} {n:>3}x  ({share:.0f}% consistent)")


def print_topsim(lexicon: dict, label: str = "") -> None:
    result = topsim_test(lexicon)
    print(f"\nCompositionality{label} (topographic similarity, 0 = none, 1 = perfect):")
    if result is None:
        print("  not defined here (needs meanings with 2+ attributes, e.g. level 2 or 3)")
    else:
        print(f"  topsim = {result.score:.2f}   shuffled messages = {abs(result.shuffled_mean):.2f}"
              f"   p = {result.p_value:.3f}   ({result.n_meanings} meanings)")


# ---------------------------------------------------------------- experiment reports


def report_holdout(run: dict, rounds: list[Row]) -> None:
    test = [r for r in rounds if r.phase == "test"]
    seen: set[str] = set()
    first = []
    for r in test:
        if r.target not in seen:
            seen.add(r.target)
            first.append(r)
    print(f"\nHeld-out test: {len(run['held_out'])} objects never seen in training")
    print(f"  first time each is shown (zero-shot): {sum(r.success for r in first)}/{len(first)}"
          f" correct ({rate(first):.0f}%)")
    print(f"  all test rounds: {rate(test):.0f}% over {len(test)} rounds")
    train = [r for r in rounds if r.phase == "train"]
    lex = {t.name: m for t, m in lexicon_of(train, run["level"]).items()}
    for r in first:
        print(f"  {r.target:<18} sent {' '.join(json.loads(r.message)):<8} "
              f"{'OK' if r.success else 'x (picked ' + str(r.picked) + ')'}")
    if run["level"] == 2:
        print("  For reference, training messages sharing a color or shape with each:")
        for name in run["held_out"]:
            color, shape = name.split()
            same = [f"{n}={' '.join(m)}" for n, m in sorted(lex.items())
                    if n.split()[0] == color or n.split()[1] == shape][:4]
            print(f"    {name:<16} {', '.join(same)}")


def report_newcomer(rounds: list[Row], k: int = 20) -> None:
    train = [r for r in rounds if r.phase == "train"]
    new = [r for r in rounds if r.phase == "newcomer"]
    print("\nNewcomer: does a fresh listener learn faster from a speaker who already has a language?")
    print(f"  original pair, first {k} rounds: {rate(train[:k]):.0f}%")
    print(f"  newcomer,      first {k} rounds: {rate(new[:k]):.0f}%")
    print(f"  original pair, last {k} rounds of training: {rate(train[-k:]):.0f}%")
    print(f"  newcomer,      last {k} rounds:  {rate(new[-k:]):.0f}%")


def report_tribes(run: dict, rounds: list[Row], k: int = 20) -> None:
    train = [r for r in rounds if r.phase == "train"]
    cross = [r for r in rounds if r.phase == "cross"]
    lex = {}
    print("\nTwo tribes: each pair builds its own dialect, then they meet")
    for tribe in ("A", "B"):
        rows = [r for r in train if r.speaker_id == tribe]
        lex[tribe] = {t.name: m for t, m in lexicon_of(rows, run["level"]).items()}
        print(f"  tribe {tribe} training success: {rate(rows):.0f}%   last {k}: {rate(rows[-k:]):.0f}%")
    shared = [n for n in lex["A"] if lex["B"].get(n) == lex["A"][n]]
    print(f"  same message for the same object in both dialects: "
          f"{len(shared)}/{len(set(lex['A']) & set(lex['B']))}")
    if cross:
        print(f"  cross-tribe success, first {k} rounds: {rate(cross[:k]):.0f}%")
        print(f"  cross-tribe success, last {k} rounds:  {rate(cross[-k:]):.0f}%")
        for pair in (("A", "B"), ("B", "A")):
            rows = [r for r in cross if (r.speaker_id, r.listener_id) == pair]
            print(f"  {pair[0]} speaks to {pair[1]}: {rate(rows):.0f}% over {len(rows)} rounds")
        after = {}
        for tribe in ("A", "B"):
            rows = [r for r in cross if r.speaker_id == tribe]
            after[tribe] = {t.name: m for t, m in lexicon_of(rows, run["level"]).items()}
        both = set(after["A"]) & set(after["B"])
        same = [n for n in both if after["A"][n] == after["B"][n]]
        print(f"  after meeting, same message in both dialects: {len(same)}/{len(both)}")


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Analyze a logged run")
    p.add_argument("--run", type=int, default=None, help="run id (default: latest)")
    p.add_argument("--db", default="runs.db")
    p.add_argument("--window", type=int, default=10, help="rounds per rolling average")
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)

    conn = open_db(a.db)
    run, rounds = load_run(conn, a.run)
    chart = Path(a.out) / f"success_run{run['run_id']}.png"
    plot_success(run, rounds, a.window, chart)
    print(f"Chart saved: {chart}")
    print(f"Run {run['run_id']}: level {run['level']}, experiment {run['experiment']}, "
          f"feedback {run['feedback']}")
    for name, rows in phases_of(rounds):
        print(f"  phase {name:<9} {rate(rows):5.1f}% over {len(rows)} rounds")

    train = [r for r in rounds if r.phase == "train"]
    speakers = sorted({r.speaker_id for r in train})
    for s in speakers:
        rows = [r for r in train if r.speaker_id == s]
        label = f" of speaker {s}" if len(speakers) > 1 else ""
        print_dictionary(rows, f"Dictionary{label}")
        print_topsim(lexicon_of(rows, run["level"]), label)

    if run["experiment"] == "holdout":
        report_holdout(run, rounds)
    elif run["experiment"] == "newcomer":
        report_newcomer(rounds)
    elif run["experiment"] == "tribes":
        report_tribes(run, rounds)


if __name__ == "__main__":
    main()
