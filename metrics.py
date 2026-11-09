"""Language-structure metrics for an emerged dictionary.

Topographic similarity (topsim): Spearman correlation, over all pairs of meanings,
between how different two meanings are and how different their messages are.
~0 = no structure; 1 = similar meanings always get similar messages (compositional).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from scipy.stats import spearmanr

from world import Cell, Thing


def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two symbol sequences."""
    prev = list(range(len(b) + 1))
    for i, sa in enumerate(a, start=1):
        cur = [i]
        for j, sb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (sa != sb)))
        prev = cur
    return prev[-1]


def meaning_distance(x: Thing, y: Thing) -> int:
    """Cells: grid (Manhattan) distance. Objects: number of attributes that differ."""
    if isinstance(x, Cell) and isinstance(y, Cell):
        return abs(x.dx - y.dx) + abs(x.dy - y.dy)
    return sum(a != b for a, b in zip(x.attributes, y.attributes, strict=True))


def topsim(lexicon: dict[Thing, list[str]]) -> float | None:
    """Topographic similarity of a meaning -> message mapping, or None if undefined
    (fewer than 3 meanings, or all meaning distances equal, as at Level 1)."""
    items = list(lexicon.items())
    if len(items) < 3:
        return None
    md, sd = [], []
    for (m1, s1), (m2, s2) in itertools.combinations(items, 2):
        md.append(meaning_distance(m1, m2))
        sd.append(edit_distance(s1, s2))
    if len(set(md)) < 2 or len(set(sd)) < 2:
        return None
    return float(spearmanr(md, sd).statistic)


@dataclass(frozen=True)
class TopsimResult:
    score: float
    shuffled_mean: float
    p_value: float     # share of shuffles scoring >= the real score
    n_meanings: int


def topsim_test(lexicon: dict[Thing, list[str]], shuffles: int = 1000,
                seed: int = 0) -> TopsimResult | None:
    """Topsim plus a permutation test: shuffle which message goes with which meaning."""
    score = topsim(lexicon)
    if score is None:
        return None
    rng = random.Random(seed)
    meanings, messages = list(lexicon), list(lexicon.values())
    null = []
    for _ in range(shuffles):
        rng.shuffle(messages)
        s = topsim(dict(zip(meanings, messages, strict=True)))
        null.append(s if s is not None else 0.0)
    p = (1 + sum(s >= score for s in null)) / (1 + shuffles)
    return TopsimResult(score, sum(null) / len(null), p, len(lexicon))
