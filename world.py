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

    @property
    def attributes(self) -> tuple:
        return (self.color, self.shape) if self.shape else (self.color,)


GRID_RADIUS = 3  # 7x7 grid; both agents stand in the centre


@dataclass(frozen=True)
class Cell:
    """Level 3: a grid cell, as an offset from the centre. dx > 0 = right, dy > 0 = up."""

    dx: int
    dy: int

    @property
    def name(self) -> str:
        parts = []
        if self.dx:
            parts.append(f"{abs(self.dx)} {'right' if self.dx > 0 else 'left'}")
        if self.dy:
            parts.append(f"{abs(self.dy)} {'up' if self.dy > 0 else 'down'}")
        return ", ".join(parts)

    @property
    def attributes(self) -> tuple:
        return (self.dx, self.dy)


Thing = Obj | Cell  # anything a round can be about


def make_objects(level: int) -> list[Thing]:
    """All targets for a level. 1: colors. 2: color x shape. 3: grid cells around the centre."""
    if level == 1:
        return [Obj(c) for c in COLORS]
    if level == 2:
        return [Obj(c, s) for c in COLORS for s in SHAPES]
    if level == 3:
        r = GRID_RADIUS
        return [Cell(dx, dy) for dy in range(r, -r - 1, -1) for dx in range(-r, r + 1)
                if (dx, dy) != (0, 0)]
    raise ValueError(f"Unsupported level: {level} (expected 1, 2 or 3)")


def _grid(marks: dict[tuple[int, int], str]) -> str:
    """Draw the grid as text. S = where both agents stand; other marks as given."""
    r = GRID_RADIUS
    rows = []
    for dy in range(r, -r - 1, -1):
        rows.append(" ".join(marks.get((dx, dy), "S" if (dx, dy) == (0, 0) else ".")
                             for dx in range(-r, r + 1)))
    return "\n".join(rows)


def describe_target(target: Thing) -> str:
    """How the speaker sees the target. Cells are shown as a picture, never in words."""
    if isinstance(target, Cell):
        return ("the cell marked T on this grid (S is where you and your partner stand):\n"
                + _grid({(target.dx, target.dy): "T"}))
    return target.name


def describe_candidates(candidates: list[Thing]) -> str:
    """How the listener sees the options, numbered from 1."""
    if candidates and isinstance(candidates[0], Cell):
        marks = {(c.dx, c.dy): str(i) for i, c in enumerate(candidates, start=1)}  # type: ignore[union-attr]
        return (f"Cells 1 to {len(candidates)} on this grid "
                f"(S is where you and your partner stand):\n" + _grid(marks))
    return "\n".join(f"{i}. {obj.name}" for i, obj in enumerate(candidates, start=1))


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
    objects: list[Thing], n_candidates: int, rng: random.Random
) -> tuple[Thing, list[Thing]]:
    """Pick a target and a shuffled list of distinct candidates that contains it."""
    if not 2 <= n_candidates <= len(objects):
        raise ValueError(f"n_candidates must be 2..{len(objects)}, got {n_candidates}")
    candidates = rng.sample(objects, n_candidates)
    target = rng.choice(candidates)
    return target, candidates
