"""The game world: objects, the alien alphabet, and the message validator.

The validator is the single source of truth for what counts as a legal message.
Any message it rejects is scored as a failed round.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow", "purple", "orange")
SHAPES: tuple[str, ...] = ("circle", "square", "triangle", "star", "diamond")

# Abstract glyphs with no obvious link to any color, shape, letter, or number.
# Deliberately excluded: Latin/Greek/Cyrillic letters, runes (they map to Latin),
# digits, arrows, and glyphs that look like our shapes (●, ■, ▲, ★, ◆, ...).
GLYPH_POOL: tuple[str, ...] = (
    "⟁", "⧖", "⨳", "⍟", "⏃", "⏚", "⌖", "⍾", "⎈", "⏧",
    "⨂", "⩚", "⩛", "⪥", "⫚", "⫷", "⬲", "⟟", "⟒", "⏁",
    "⏂", "⍜", "⌇", "⍦", "⍭", "⎍", "⎎", "⏆", "⏇", "⏈",
    "⏉", "⌰", "⍡", "⍢", "⍤", "⍩", "⌸", "⌹", "⍯", "⍨",
)

DEFAULT_ALPHABET_SIZE = 8
DEFAULT_MAX_LEN = 3


@dataclass(frozen=True)
class Obj:
    """An object the speaker can describe. Level 1 objects have no shape."""

    color: str
    shape: str | None = None

    @property
    def name(self) -> str:
        return f"{self.color} {self.shape}" if self.shape else self.color


def make_objects(level: int) -> list[Obj]:
    """All objects for a level. Level 1: colors only. Level 2: color x shape."""
    if level == 1:
        return [Obj(c) for c in COLORS]
    if level == 2:
        return [Obj(c, s) for c in COLORS for s in SHAPES]
    raise ValueError(f"Unsupported level: {level} (expected 1 or 2)")


def make_alphabet(rng: random.Random, size: int = DEFAULT_ALPHABET_SIZE) -> list[str]:
    """A fresh random alphabet for one run. Nothing carries over between runs."""
    if not 1 <= size <= len(GLYPH_POOL):
        raise ValueError(f"Alphabet size must be 1..{len(GLYPH_POOL)}, got {size}")
    return rng.sample(GLYPH_POOL, size)


@dataclass(frozen=True)
class Validation:
    ok: bool
    reason: str = ""


def validate_message(
    message: list[str], alphabet: list[str], max_len: int = DEFAULT_MAX_LEN
) -> Validation:
    """Check a message is 1..max_len symbols, each from this run's alphabet."""
    if not isinstance(message, list) or not message:
        return Validation(False, "message must be a non-empty list of symbols")
    if len(message) > max_len:
        return Validation(False, f"message has {len(message)} symbols (max {max_len})")
    allowed = set(alphabet)
    for symbol in message:
        if not isinstance(symbol, str) or symbol not in allowed:
            return Validation(False, f"symbol {symbol!r} is not in the alphabet")
    return Validation(True)


def sample_round(
    objects: list[Obj], n_candidates: int, rng: random.Random
) -> tuple[Obj, list[Obj]]:
    """Pick a target and a shuffled list of distinct candidates that contains it."""
    if not 2 <= n_candidates <= len(objects):
        raise ValueError(f"n_candidates must be 2..{len(objects)}, got {n_candidates}")
    candidates = rng.sample(objects, n_candidates)
    target = rng.choice(candidates)
    return target, candidates
