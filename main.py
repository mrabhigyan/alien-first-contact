"""Run the game: speakers and listeners play many rounds; every round is logged to SQLite.

    python main.py --mock --rounds 20                       # free test, no API calls
    python main.py --rounds 100 --level 1                   # standard run with Gemini
    python main.py --level 2 --experiment holdout           # test on never-seen objects
    python main.py --level 2 --experiment newcomer          # a fresh listener joins later
    python main.py --level 2 --experiment tribes            # two pairs, then they meet
    python main.py --level 2 --feedback full                # speaker learns what was picked

A run is a list of phases. Each phase says which (speaker, listener) pairs play,
which objects can be targets, and which can appear as candidates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from dotenv import load_dotenv

from agents import Agent, Brain, GeminiBrain, Listener, MockBrain, Speaker
from costs import estimate_cost, format_cost
from world import (
    DEFAULT_ALPHABET_SIZE,
    DEFAULT_MAX_LEN,
    Thing,
    make_alphabet,
    make_objects,
    sample_round,
    validate_message,
)

DEFAULT_MODEL = "gemini-3.8-flash"
REPORT_EVERY = 10
EXPERIMENTS = ("standard", "holdout", "newcomer", "tribes")
PHASE2_DEFAULT = {"standard": 0, "holdout": 30, "newcomer": 100, "tribes": 100}


@dataclass(frozen=True)
class Config:
    rounds: int
    level: int
    candidates: int
    alphabet_size: int
    max_len: int
    memory: int
    seed: int
    mock: bool
    db: str
    model: str
    notes_every: int = 0         # 0 = never; N = each agent rewrites its own notes every N rounds
    feedback: str = "outcome"    # 'outcome': speaker learns success only; 'full': also what was picked
    experiment: str = "standard"
    holdout: int = 5             # held-out objects in the holdout experiment
    phase2_rounds: int = 0       # rounds in the second phase (test / newcomer / cross-tribe)


@dataclass(frozen=True)
class Phase:
    name: str
    rounds: int
    pairs: list[tuple[str, str]]  # (speaker id, listener id), taking turns round by round
    targets: list[Thing]
    pool: list[Thing]             # candidates come from here; contains every target


@dataclass(frozen=True)
class RoundResult:
    round_no: int
    target: Thing
    candidates: list[Thing]
    message: list[str]
    valid: bool
    error: str
    choice: int | None
    picked: Thing | None
    success: bool
    speaker_reasoning: str
    listener_reasoning: str
    phase: str = "train"
    speaker_id: str = "A"
    listener_id: str = "A"


@dataclass
class Population:
    """Every agent in a run. Agents never share memory; only messages pass between them."""

    speakers: dict[str, Speaker] = field(default_factory=dict)
    listeners: dict[str, Listener] = field(default_factory=dict)

    def all(self) -> list[tuple[str, Agent]]:
        return ([(f"speaker {k}", a) for k, a in self.speakers.items()]
                + [(f"listener {k}", a) for k, a in self.listeners.items()])


# ---------------------------------------------------------------- database

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    level         INTEGER NOT NULL,
    n_candidates  INTEGER NOT NULL,
    alphabet      TEXT NOT NULL,      -- JSON list of symbols
    max_len       INTEGER NOT NULL,
    memory_window INTEGER NOT NULL,
    seed          INTEGER NOT NULL,
    model         TEXT NOT NULL,
    mock          INTEGER NOT NULL,
    rounds_played INTEGER NOT NULL DEFAULT 0,
    successes     INTEGER NOT NULL DEFAULT 0,
    api_calls     INTEGER NOT NULL DEFAULT 0,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS rounds (
    run_id             INTEGER NOT NULL REFERENCES runs(run_id),
    round_no           INTEGER NOT NULL,
    target             TEXT NOT NULL,
    candidates         TEXT NOT NULL, -- JSON list of names, in the order shown
    message            TEXT NOT NULL, -- JSON list of symbols as sent
    valid              INTEGER NOT NULL,
    error              TEXT NOT NULL,
    choice             INTEGER,       -- 1-based, as the listener answered
    picked             TEXT,
    success            INTEGER NOT NULL,
    speaker_reasoning  TEXT NOT NULL,
    listener_reasoning TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    PRIMARY KEY (run_id, round_no)
);
CREATE TABLE IF NOT EXISTS notes (
    run_id     INTEGER NOT NULL REFERENCES runs(run_id),
    round_no   INTEGER NOT NULL,  -- notes written after this round
    agent      TEXT NOT NULL,     -- e.g. 'speaker A', 'listener N'
    notes      TEXT NOT NULL,
    PRIMARY KEY (run_id, round_no, agent)
);
"""

# Columns added after the first version of the schema: (table, column, definition).
MIGRATIONS = [
    ("runs", "notes_every", "INTEGER NOT NULL DEFAULT 0"),
    ("runs", "feedback", "TEXT NOT NULL DEFAULT 'outcome'"),
    ("runs", "experiment", "TEXT NOT NULL DEFAULT 'standard'"),
    ("runs", "held_out", "TEXT NOT NULL DEFAULT '[]'"),  # JSON names of held-out objects
    ("rounds", "phase", "TEXT NOT NULL DEFAULT 'train'"),
    ("rounds", "speaker_id", "TEXT NOT NULL DEFAULT 'A'"),
    ("rounds", "listener_id", "TEXT NOT NULL DEFAULT 'A'"),
]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def open_db(path: str) -> sqlite3.Connection:
    # timeout: wait up to 60s if another run is writing (parallel runs share the file)
    conn = sqlite3.connect(path, timeout=60)
    try:
        # WAL: readers (dashboard, analysis) never block writers (running experiments).
        # Can only be switched on while no other connection is open; it then stays on.
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    conn.executescript(SCHEMA)
    for table, column, definition in MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    conn.commit()
    return conn


def start_run(conn: sqlite3.Connection, cfg: Config, alphabet: list[str],
              held_out: list[Thing]) -> int:
    cur = conn.execute(
        "INSERT INTO runs (started_at, level, n_candidates, alphabet, max_len, memory_window,"
        " seed, model, mock, notes_every, feedback, experiment, held_out)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            now(), cfg.level, cfg.candidates, json.dumps(alphabet, ensure_ascii=False),
            cfg.max_len, cfg.memory, cfg.seed, "mock" if cfg.mock else cfg.model, int(cfg.mock),
            cfg.notes_every, cfg.feedback, cfg.experiment,
            json.dumps([t.name for t in held_out]),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def with_retry(action, attempts: int = 8) -> None:
    """Run a database write; if the file is locked by another process, wait and retry."""
    for attempt in range(attempts):
        try:
            action()
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def log_round(conn: sqlite3.Connection, run_id: int, r: RoundResult) -> None:
    with_retry(lambda: _insert_round(conn, run_id, r))


def _insert_round(conn: sqlite3.Connection, run_id: int, r: RoundResult) -> None:
    conn.execute(
        "INSERT INTO rounds (run_id, round_no, target, candidates, message, valid, error, choice,"
        " picked, success, speaker_reasoning, listener_reasoning, created_at, phase, speaker_id,"
        " listener_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id, r.round_no, r.target.name,
            json.dumps([c.name for c in r.candidates]),
            json.dumps(r.message, ensure_ascii=False),
            int(r.valid), r.error, r.choice, r.picked.name if r.picked else None,
            int(r.success), r.speaker_reasoning, r.listener_reasoning, now(),
            r.phase, r.speaker_id, r.listener_id,
        ),
    )
    conn.commit()  # commit every round, so a crash never loses earlier rounds


def finish_run(conn: sqlite3.Connection, run_id: int, results: list[RoundResult], brain: Brain) -> None:
    def write() -> None:
        conn.execute(
            "UPDATE runs SET finished_at=?, rounds_played=?, successes=?, api_calls=?,"
            " input_tokens=?, output_tokens=? WHERE run_id=?",
            (
                now(), len(results), sum(r.success for r in results), brain.usage.calls,
                brain.usage.input_tokens, brain.usage.output_tokens, run_id,
            ),
        )
        conn.commit()

    with_retry(write)


# ---------------------------------------------------------------- experiment design


def choose_held_out(objects: list[Thing], k: int, rng: random.Random) -> list[Thing]:
    """Pick k objects to hide during training. Every attribute value of a held-out object
    must still appear in training (otherwise it could not be described), and no two
    held-out objects share an attribute value."""
    order = objects[:]
    rng.shuffle(order)
    chosen: list[Thing] = []
    for obj in order:
        if len(chosen) == k:
            break
        if any(set(obj.attributes) & set(c.attributes) for c in chosen):
            continue
        training = [o for o in objects if o not in chosen and o != obj]
        seen = {v for o in training for v in o.attributes}
        if all(v in seen for v in obj.attributes):
            chosen.append(obj)
    if len(chosen) < k:
        raise ValueError(f"Could only hold out {len(chosen)} of {k} objects at this level")
    return chosen


def design(cfg: Config, objects: list[Thing], rng: random.Random
           ) -> tuple[list[Phase], list[Thing], list[tuple[str, str]]]:
    """Phases, held-out objects, and which (role, id) agents to create."""
    p2 = cfg.phase2_rounds
    if cfg.experiment == "standard":
        return [Phase("train", cfg.rounds, [("A", "A")], objects, objects)], [], [
            ("speaker", "A"), ("listener", "A")]
    if cfg.experiment == "holdout":
        if cfg.level == 1:
            raise ValueError("The holdout experiment needs level 2 or 3 (objects with parts)")
        held = choose_held_out(objects, cfg.holdout, rng)
        train = [o for o in objects if o not in held]
        return [Phase("train", cfg.rounds, [("A", "A")], train, train),
                Phase("test", p2, [("A", "A")], held, objects)], held, [
            ("speaker", "A"), ("listener", "A")]
    if cfg.experiment == "newcomer":
        return [Phase("train", cfg.rounds, [("A", "A")], objects, objects),
                Phase("newcomer", p2, [("A", "N")], objects, objects)], [], [
            ("speaker", "A"), ("listener", "A"), ("listener", "N")]
    if cfg.experiment == "tribes":
        return [Phase("train", cfg.rounds, [("A", "A"), ("B", "B")], objects, objects),
                Phase("cross", p2, [("A", "B"), ("B", "A")], objects, objects)], [], [
            ("speaker", "A"), ("listener", "A"), ("speaker", "B"), ("listener", "B")]
    raise ValueError(f"Unknown experiment: {cfg.experiment}")


def build_population(cfg: Config, brain: Brain, alphabet: list[str],
                     roles: list[tuple[str, str]]) -> Population:
    # One brain object serves every agent: API calls are stateless, so nothing leaks.
    # Each agent's memory lives only in its own notebook and notes.
    pop = Population()
    for role, key in roles:
        if role == "speaker":
            pop.speakers[key] = Speaker(brain=brain, alphabet=alphabet, max_len=cfg.max_len,
                                        memory_window=cfg.memory)
        else:
            pop.listeners[key] = Listener(brain=brain, alphabet=alphabet, memory_window=cfg.memory)
    return pop


def pick(phase: Phase, cfg: Config, rng: random.Random) -> tuple[Thing, list[Thing]]:
    """Target + candidates for one round of this phase."""
    if phase.targets == phase.pool:
        return sample_round(phase.pool, cfg.candidates, rng)  # same stream as earlier versions
    target = rng.choice(phase.targets)
    others = rng.sample([o for o in phase.pool if o != target], cfg.candidates - 1)
    candidates = others + [target]
    rng.shuffle(candidates)
    return target, candidates


# ---------------------------------------------------------------- game


def play_round(
    round_no: int,
    speaker: Speaker,
    listener: Listener,
    target: Thing,
    candidates: list[Thing],
    cfg: Config,
) -> RoundResult:
    said = speaker.speak(round_no, target)
    message = [s.strip() for s in said.message]
    check = validate_message(message, speaker.alphabet, cfg.max_len)

    if not check.ok:
        # The validator is the source of truth: an invalid message fails the round,
        # and nothing reaches the listener.
        speaker.remember(round_no, target, message, f"INVALID message ({check.reason}): FAILED")
        return RoundResult(
            round_no, target, candidates, message, False, check.reason,
            None, None, False, said.reasoning, "",
        )

    heard = listener.listen(round_no, message, candidates)
    in_range = 1 <= heard.choice <= len(candidates)
    picked = candidates[heard.choice - 1] if in_range else None
    success = picked == target
    error = "" if in_range else f"listener choice {heard.choice} out of range"

    outcome = "SUCCESS" if success else "FAILED"
    if cfg.feedback == "full" and not success:
        outcome += f" (partner picked {picked.name if picked else 'an invalid option'})"
    speaker.remember(round_no, target, message, outcome)
    listener.remember(round_no, message, candidates, picked, target)
    return RoundResult(
        round_no, target, candidates, message, True, error,
        heard.choice, picked, success, said.reasoning, heard.reasoning,
    )


def update_notes(conn: sqlite3.Connection, run_id: int, round_no: int, pop: Population) -> None:
    """Each agent rewrites its own private notes; every version is logged for inspection."""
    written = []
    for name, agent in pop.all():
        if agent.notes_through < len(agent.notebook):
            agent.update_notes(round_no)
            written.append((run_id, round_no, name, agent.notes))

    def write() -> None:
        conn.executemany("INSERT INTO notes VALUES (?, ?, ?, ?)", written)
        conn.commit()

    with_retry(write)
    print(f"     [notes updated after round {round_no}]")


def build_brain(cfg: Config, alphabet: list[str]) -> Brain:
    if cfg.mock:
        return MockBrain(random.Random(cfg.seed + 1), alphabet, cfg.candidates, cfg.max_len)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your-key-here":
        sys.exit("No GEMINI_API_KEY found. Put your key in .env (see .env.example).")
    return GeminiBrain(api_key=api_key, model=cfg.model)


def run(cfg: Config, conn: sqlite3.Connection, brain: Brain | None = None) -> list[RoundResult]:
    """Play a whole run. Pass `brain` to override (used by tests)."""
    rng = random.Random(cfg.seed)
    alphabet = make_alphabet(rng, cfg.alphabet_size)
    objects = make_objects(cfg.level)
    phases, held_out, roles = design(cfg, objects, rng)
    brain = brain or build_brain(cfg, alphabet)
    pop = build_population(cfg, brain, alphabet, roles)

    run_id = start_run(conn, cfg, alphabet, held_out)
    print_header(cfg, run_id, alphabet, len(objects), held_out)
    total = sum(p.rounds for p in phases)

    results: list[RoundResult] = []
    round_no = 0
    try:
        for phase in phases:
            print(f"\n--- phase: {phase.name} ({phase.rounds} rounds) ---")
            turns = itertools.cycle(phase.pairs)
            for _ in range(phase.rounds):
                round_no += 1
                s_id, l_id = next(turns)
                target, candidates = pick(phase, cfg, rng)
                result = play_round(round_no, pop.speakers[s_id], pop.listeners[l_id],
                                    target, candidates, cfg)
                result = RoundResult(**{**result.__dict__, "phase": phase.name,
                                        "speaker_id": s_id, "listener_id": l_id})
                results.append(result)
                log_round(conn, run_id, result)
                print_round(result, show_pair=cfg.experiment != "standard")
                if round_no % REPORT_EVERY == 0:
                    print_block(results, cfg.candidates)
                if cfg.notes_every and round_no % cfg.notes_every == 0 and round_no < total:
                    update_notes(conn, run_id, round_no, pop)
    except KeyboardInterrupt:
        print("\nStopped by user. Saving what we have...")
    finally:
        finish_run(conn, run_id, results, brain)
        print_summary(results, cfg.candidates, brain, run_id)
    return results


# ---------------------------------------------------------------- printing


def print_header(cfg: Config, run_id: int, alphabet: list[str], n_objects: int,
                 held_out: list[Thing]) -> None:
    mode = "MOCK (random, free)" if cfg.mock else f"Gemini ({cfg.model})"
    print(f"Run {run_id} | {mode} | level {cfg.level} ({n_objects} objects) | seed {cfg.seed}")
    print(f"Experiment: {cfg.experiment} | feedback: {cfg.feedback} | notes every "
          f"{cfg.notes_every or '-'} | memory {cfg.memory}")
    if held_out:
        print(f"Held out until the test phase: {', '.join(t.name for t in held_out)}")
    print(f"Alphabet: {' '.join(alphabet)} | {cfg.candidates} candidates | "
          f"baseline {100 / cfg.candidates:.0f}%")


def print_round(r: RoundResult, show_pair: bool = False) -> None:
    sent = " ".join(r.message) if r.message else "-"
    if not r.valid:
        outcome = f"INVALID ({r.error})"
    else:
        outcome = f"picked {r.picked.name if r.picked else '?'} " + ("OK" if r.success else "x")
    pair = f" [{r.speaker_id}>{r.listener_id}]" if show_pair else ""
    print(f"{r.round_no:4d}{pair} | {r.target.name:<16} | {sent:<7} | {outcome}")


def rate(results: list[RoundResult]) -> float:
    return 100 * sum(r.success for r in results) / len(results) if results else 0.0


def print_block(results: list[RoundResult], n_candidates: int) -> None:
    block = results[-REPORT_EVERY:]
    print(f"     -- rounds {block[0].round_no}-{block[-1].round_no}: {rate(block):.0f}% "
          f"(overall {rate(results):.0f}%, baseline {100 / n_candidates:.0f}%)")


def print_summary(results: list[RoundResult], n_candidates: int, brain: Brain, run_id: int) -> None:
    if not results:
        print("No rounds played.")
        return
    invalid = sum(not r.valid for r in results)
    last = results[-20:]
    u = brain.usage
    print(f"\n=== Run {run_id} done: {len(results)} rounds ===")
    print(f"Success overall : {rate(results):.1f}%   (baseline {100 / n_candidates:.0f}%)")
    for phase in dict.fromkeys(r.phase for r in results):
        rs = [r for r in results if r.phase == phase]
        if len(rs) < len(results):
            print(f"  phase {phase:<9}: {rate(rs):.1f}% over {len(rs)} rounds")
    print(f"Success last {len(last):<3}: {rate(last):.1f}%")
    print(f"Invalid messages: {invalid}")
    print(f"API calls       : {u.calls}   tokens in/out: {u.input_tokens:,} / {u.output_tokens:,}")
    model = getattr(brain, "model", "mock")
    print(f"Estimated cost  : {format_cost(estimate_cost(model, u.input_tokens, u.output_tokens))}")


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(description="Alien First Contact: emergent language game")
    p.add_argument("--rounds", type=int, default=20, help="rounds in the first (training) phase")
    p.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--alphabet-size", type=int, default=DEFAULT_ALPHABET_SIZE)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--memory", type=int, default=20, help="recent notebook rounds shown to each agent")
    p.add_argument("--notes-every", type=int, default=20,
                   help="each agent rewrites its own private notes every N rounds (0 = off)")
    p.add_argument("--feedback", choices=["outcome", "full"], default="outcome",
                   help="full = on failure the speaker also learns what the listener picked")
    p.add_argument("--experiment", choices=EXPERIMENTS, default="standard")
    p.add_argument("--holdout", type=int, default=5, help="objects held out (holdout experiment)")
    p.add_argument("--phase2-rounds", type=int, default=None,
                   help="rounds in the second phase (default depends on the experiment)")
    p.add_argument("--seed", type=int, default=None, help="random if omitted")
    p.add_argument("--mock", action="store_true", help="random answers, no API calls")
    p.add_argument("--db", default="runs.db")
    p.add_argument("--model", default=None, help="overrides GEMINI_MODEL from .env")
    a = p.parse_args(argv)
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(2**31)
    model = a.model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    phase2 = a.phase2_rounds if a.phase2_rounds is not None else PHASE2_DEFAULT[a.experiment]
    return Config(a.rounds, a.level, a.candidates, a.alphabet_size, a.max_len, a.memory,
                  seed, a.mock, a.db, model, a.notes_every, a.feedback, a.experiment,
                  a.holdout, phase2)


def main(argv: list[str] | None = None) -> None:
    # Windows consoles may not print alien symbols without this.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()  # reads GEMINI_API_KEY / GEMINI_MODEL from .env into the environment
    cfg = parse_args(argv)
    conn = open_db(cfg.db)
    try:
        run(cfg, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
