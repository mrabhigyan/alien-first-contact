"""Neural baseline: the same referential game, played by small neural networks trained
from scratch, with no prior knowledge of language, colors, or shapes.

    python neural_baseline.py --level 2 --seeds 5

Each agent is a tiny network. The speaker reads an object and emits 3 symbols with a GRU;
the listener reads the symbols and scores the 4 candidates. Both are trained end to end
with the straight-through Gumbel-softmax estimator (Havrylov & Titov, 2017), using only
the game's success signal. Results are measured exactly like the LLM runs: games needed
to learn, topographic similarity (metrics.py), and accuracy on held-out objects.

Difference from the LLM game: messages always have exactly 3 symbols (LLMs may send 1-3).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from main import choose_held_out
from metrics import topsim_test
from world import DEFAULT_ALPHABET_SIZE, DEFAULT_MAX_LEN, Thing, make_alphabet, make_objects

RESULTS = Path(__file__).parent / "results"


# ---------------------------------------------------------------- data


def encode(objects: list[Thing]) -> torch.Tensor:
    """One-hot per attribute, concatenated (e.g. 6 colors + 5 shapes = 11 inputs)."""
    n_attr = len(objects[0].attributes)
    values = [sorted({o.attributes[i] for o in objects}, key=str) for i in range(n_attr)]
    rows = []
    for o in objects:
        vec: list[float] = []
        for i, vals in enumerate(values):
            one_hot = [0.0] * len(vals)
            one_hot[vals.index(o.attributes[i])] = 1.0
            vec += one_hot
        rows.append(vec)
    return torch.tensor(rows)


def sample_games(target_pool: torch.Tensor, cand_pool: torch.Tensor, batch: int, k: int,
                 gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Indices into cand_pool: targets (B), candidates (B x k, distinct, containing the
    target), and the target's position among the candidates (B)."""
    targets = target_pool[torch.randint(len(target_pool), (batch,), generator=gen)]
    scores = torch.rand(batch, len(cand_pool), generator=gen)
    # Never pick the target as a distractor. Columns are positions in cand_pool, which
    # differ from object indices once objects are held out, so look positions up.
    is_target = cand_pool[None, :] == targets[:, None]
    scores[is_target] = -1.0
    distractors = cand_pool[scores.argsort(dim=1, descending=True)[:, : k - 1]]
    cands = torch.cat([targets[:, None], distractors], dim=1)
    order = torch.rand(batch, k, generator=gen).argsort(dim=1)
    cands = cands.gather(1, order)
    labels = (order == 0).float().argmax(dim=1)          # where the target landed
    return targets, cands, labels


# ---------------------------------------------------------------- agents


class SpeakerNet(nn.Module):
    def __init__(self, n_in: int, vocab: int, length: int, hidden: int = 128) -> None:
        super().__init__()
        self.vocab, self.length = vocab, length
        self.encoder = nn.Linear(n_in, hidden)
        self.cell = nn.GRUCell(vocab, hidden)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Message as one-hot symbols, B x length x vocab (sampled in training, greedy in eval)."""
        h = torch.tanh(self.encoder(x))
        prev = torch.zeros(x.size(0), self.vocab, device=x.device)
        symbols = []
        for _ in range(self.length):
            h = self.cell(prev, h)
            logits = self.out(h)
            if self.training:
                sym = F.gumbel_softmax(logits, tau=tau, hard=True)
            else:
                sym = F.one_hot(logits.argmax(-1), self.vocab).float()
            symbols.append(sym)
            prev = sym
        return torch.stack(symbols, dim=1)


class ListenerNet(nn.Module):
    def __init__(self, n_in: int, vocab: int, hidden: int = 128) -> None:
        super().__init__()
        self.embed = nn.Linear(vocab, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.objects = nn.Linear(n_in, hidden)

    def forward(self, message: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Scores B x k: how well each candidate matches the message."""
        _, h = self.gru(self.embed(message))
        return torch.einsum("bh,bkh->bk", h[-1], self.objects(candidates))


# ---------------------------------------------------------------- experiment


@dataclass
class SeedResult:
    seed: int
    games_to_90: int | None          # training games until eval accuracy first reached 90%
    final_accuracy: float
    topsim: float | None
    topsim_shuffled: float | None
    held_out: list[str]
    held_out_accuracy: float | None  # zero-shot accuracy on never-seen objects
    curve: list[tuple[int, float]] = field(default_factory=list)  # (games played, accuracy)
    lexicon: dict[str, str] = field(default_factory=dict)


@torch.no_grad()
def accuracy(speaker, listener, feats, targets, cands_pool, k, gen, device, n=2000) -> float:
    speaker.eval(), listener.eval()
    t, c, y = sample_games(targets, cands_pool, n, k, gen)
    msg = speaker(feats[t].to(device))
    pred = listener(msg, feats[c].to(device)).argmax(dim=1).cpu()
    speaker.train(), listener.train()
    return float((pred == y).float().mean())


def run_seed(level: int, seed: int, steps: int, batch: int, k: int, vocab: int, length: int,
             holdout: int, eval_every: int, device: str) -> SeedResult:
    random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    objects = make_objects(level)
    feats = encode(objects)
    held = choose_held_out(objects, holdout, random.Random(seed)) if holdout else []
    train_idx = torch.tensor([i for i, o in enumerate(objects) if o not in held])
    held_idx = torch.tensor([i for i, o in enumerate(objects) if o in held])
    all_idx = torch.arange(len(objects))

    speaker = SpeakerNet(feats.size(1), vocab, length).to(device)
    listener = ListenerNet(feats.size(1), vocab).to(device)
    optim = torch.optim.Adam([*speaker.parameters(), *listener.parameters()], lr=1e-3)

    curve: list[tuple[int, float]] = []
    games_to_90 = None
    # Measure often early (log-spaced) and then every `eval_every` steps, so the
    # learning curve is faithful on a log-scaled axis.
    log_steps = {int(1.25 ** i) for i in range(60)}
    for step in range(1, steps + 1):
        t, c, y = sample_games(train_idx, train_idx, batch, k, gen)
        msg = speaker(feats[t].to(device))
        loss = F.cross_entropy(listener(msg, feats[c].to(device)), y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % eval_every == 0 or step in log_steps:
            acc = accuracy(speaker, listener, feats, train_idx, train_idx, k, gen, device)
            curve.append((step * batch, acc))
            if games_to_90 is None and acc >= 0.9:
                games_to_90 = step * batch

    final = accuracy(speaker, listener, feats, train_idx, train_idx, k, gen, device, n=5000)
    held_acc = (accuracy(speaker, listener, feats, held_idx, all_idx, k, gen, device, n=5000)
                if held else None)

    # The emerged lexicon, written with a random alien alphabet like the LLM runs.
    alphabet = make_alphabet(random.Random(seed), max(vocab, DEFAULT_ALPHABET_SIZE))
    speaker.eval()
    with torch.no_grad():
        ids = speaker(feats.to(device)).argmax(dim=-1).cpu().tolist()
    lexicon = {o: [alphabet[i] for i in row] for o, row in zip(objects, ids) if o not in held}
    ts = topsim_test(lexicon, shuffles=300)
    return SeedResult(
        seed=seed, games_to_90=games_to_90, final_accuracy=final,
        topsim=ts.score if ts else None, topsim_shuffled=ts.shuffled_mean if ts else None,
        held_out=[o.name for o in held], held_out_accuracy=held_acc, curve=curve,
        lexicon={o.name: " ".join(m) for o, m in lexicon.items()},
    )


def summarize(results: list[SeedResult]) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    return {
        "seeds": len(results),
        "reached_90": sum(r.games_to_90 is not None for r in results),
        "games_to_90_mean": mean([r.games_to_90 for r in results]),
        "final_accuracy_mean": mean([r.final_accuracy for r in results]),
        "topsim_mean": mean([r.topsim for r in results]),
        "topsim_min": min((r.topsim for r in results if r.topsim is not None), default=None),
        "topsim_max": max((r.topsim for r in results if r.topsim is not None), default=None),
        "held_out_accuracy_mean": mean([r.held_out_accuracy for r in results]),
    }


def plot_comparison(level: int, db: Path = Path(__file__).parent / "runs.db") -> Path:
    """Learning curves on one log-scaled axis: LLM runs (rounds played) vs neural seeds
    (training games played). Same game, same level, same 25% baseline."""
    import sqlite3

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from analyze import rolling_rate

    data = json.loads((RESULTS / f"neural_baseline_level{level}.json").read_text(encoding="utf-8"))
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    llm_runs = [rid for (rid,) in conn.execute(
        "SELECT run_id FROM runs WHERE level=? AND mock=0 AND experiment='standard'"
        " AND feedback='outcome' AND finished_at IS NOT NULL", (level,))]
    llm_curves = []
    for rid in llm_runs:
        succ = [s for (s,) in conn.execute(
            "SELECT success FROM rounds WHERE run_id=? ORDER BY round_no", (rid,))]
        if len(succ) >= 50:
            llm_curves.append(rolling_rate(succ, 10))

    surface, ink, soft, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    llm_color, nn_color = "#2a78d6", "#eb6834"  # palette slots 1 and 2
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.axhline(25, color=soft, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(1.4e5, 27, "random guessing (25%)", fontsize=8.5, color=soft, ha="right")

    for curve in llm_curves:  # from round 10, once the 10-round average is full
        ax.plot(range(10, len(curve) + 1), curve[9:], color=llm_color, linewidth=1.4, alpha=0.55)
    for seed in data["seeds"]:
        xs, ys = zip(*seed["curve"])
        ax.plot(xs, [100 * y for y in ys], color=nn_color, linewidth=1.4, alpha=0.55)

    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.set_xlim(8, max(max(x for x, _ in s["curve"]) for s in data["seeds"]))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Games played (log scale)", color=soft, fontsize=9)
    ax.set_ylabel("Success", color=soft, fontsize=9)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(colors=soft, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)

    s = data["summary"]
    llm_label = f"LLM agents ({len(llm_curves)} runs)"
    nn_label = f"Neural agents ({s['seeds']} seeds)"
    ax.text(14, 36, llm_label, color=llm_color, fontsize=9.5, fontweight="bold")
    ax.text(2500, 36, nn_label, color=nn_color, fontsize=9.5, fontweight="bold")
    g = s["games_to_90_mean"]
    nn_games = f"{float(f'{g:.2g}'):,.0f}" if g else "many"  # 2 significant figures
    fig.suptitle(f"LLM agents agree in ~20 games; neural agents need ~{nn_games}",
                 x=0.07, ha="left", fontsize=13, fontweight="bold", color=ink)
    ax.set_title(f"Level {level}: same game, same 8 symbols, same 4 candidates · "
                 f"neural agents start with no prior knowledge", loc="left", fontsize=9,
                 color=soft)
    ax.legend(handles=[plt.Line2D([], [], color=llm_color, lw=2, label=llm_label),
                       plt.Line2D([], [], color=nn_color, lw=2, label=nn_label)],
              loc="lower right", frameon=False, fontsize=8.5, labelcolor=soft)
    fig.tight_layout()
    out = RESULTS / f"llm_vs_neural_level{level}.png"
    fig.savefig(out, facecolor=surface)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Neural-network baseline for the referential game")
    p.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=3000, help="training steps per seed")
    p.add_argument("--batch", type=int, default=64, help="games per training step")
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--vocab", type=int, default=DEFAULT_ALPHABET_SIZE)
    p.add_argument("--length", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--holdout", type=int, default=5, help="objects never seen in training (0 = none)")
    p.add_argument("--eval-every", type=int, default=25)
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    holdout = 0 if a.level == 1 else a.holdout
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Neural baseline · level {a.level} · {a.seeds} seeds · {a.steps} steps x {a.batch} "
          f"games · device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    results = []
    for seed in range(a.seeds):
        t0 = time.time()
        r = run_seed(a.level, seed, a.steps, a.batch, a.candidates, a.vocab, a.length,
                     holdout, a.eval_every, device)
        results.append(r)
        held = f"{100 * r.held_out_accuracy:.0f}%" if r.held_out_accuracy is not None else "-"
        ts = f"{r.topsim:.2f}" if r.topsim is not None else "-"
        print(f"  seed {seed}: accuracy {100 * r.final_accuracy:.1f}% · games to 90%: "
              f"{r.games_to_90 or 'never'} · topsim {ts} · held-out {held} · {time.time() - t0:.0f}s")

    summary = summarize(results)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"neural_baseline_level{a.level}.json"
    out.write_text(json.dumps({"config": vars(a), "device": device, "summary": summary,
                               "seeds": [asdict(r) for r in results]}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nSummary: {json.dumps(summary, indent=1)}\nSaved {out}")
    print(f"Chart saved: {plot_comparison(a.level)}")


if __name__ == "__main__":
    main()
