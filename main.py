"""Run the game: speaker and listener play many rounds; every round is logged to SQLite.

    python main.py --mock --rounds 20          # free test, no API calls
    python main.py --rounds 100 --level 1      # real run with Gemini
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from dotenv import load_dotenv

from costs import estimate_cost, format_cost
from agents import Brain, GeminiBrain, Listener, MockBrain, Speaker
from world import (
    DEFAULT_ALPHABET_SIZE,
    DEFAULT_MAX_LEN,
    Obj,
    make_alphabet,
    make_objects,
    sample_round,
    validate_message,
)

DEFAULT_MODEL = "gemini-3.8-flash"
REPORT_EVERY = 10


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
    notes_every: int = 0  # 0 = agents never write notes; N = each agent updates notes every N rounds


@dataclass(frozen=True)
class RoundResult:
    round_no: int
    target: Obj
    candidates: list[Obj]
    message: list[str]
    valid: bool
    error: str
    choice: int | None
    picked: Obj | None
    success: bool
    speaker_reasoning: str
    listener_reasoning: str


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
    agent      TEXT NOT NULL,     -- 'speaker' or 'listener'
    notes      TEXT NOT NULL,
    PRIMARY KEY (run_id, round_no, agent)
);
"""

# Columns added after the first version of the schema: (table, column, definition).
MIGRATIONS = [("runs", "notes_every", "INTEGER NOT NULL DEFAULT 0")]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_db(path: str) -> sqlite3.Connection:
    # timeout: wait up to 30s if another run is writing (parallel runs share the file)
    conn = sqlite3.connect(path, timeout=30)
    conn.executescript(SCHEMA)
    for table, column, definition in MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    conn.commit()
    return conn


def start_run(conn: sqlite3.Connection, cfg: Config, alphabet: list[str]) -> int:
    cur = conn.execute(
        "INSERT INTO runs (started_at, level, n_candidates, alphabet, max_len, memory_window,"
        " seed, model, mock, notes_every) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            now(), cfg.level, cfg.candidates, json.dumps(alphabet, ensure_ascii=False),
            cfg.max_len, cfg.memory, cfg.seed, "mock" if cfg.mock else cfg.model, int(cfg.mock),
            cfg.notes_every,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def log_round(conn: sqlite3.Connection, run_id: int, r: RoundResult) -> None:
    conn.execute(
        "INSERT INTO rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id, r.round_no, r.target.name,
            json.dumps([c.name for c in r.candidates]),
            json.dumps(r.message, ensure_ascii=False),
            int(r.valid), r.error, r.choice, r.picked.name if r.picked else None,
            int(r.success), r.speaker_reasoning, r.listener_reasoning, now(),
        ),
    )
    conn.commit()  # commit every round, so a crash never loses earlier rounds


def finish_run(conn: sqlite3.Connection, run_id: int, results: list[RoundResult], brain: Brain) -> None:
    conn.execute(
        "UPDATE runs SET finished_at=?, rounds_played=?, successes=?, api_calls=?,"
        " input_tokens=?, output_tokens=? WHERE run_id=?",
        (
            now(), len(results), sum(r.success for r in results), brain.usage.calls,
            brain.usage.input_tokens, brain.usage.output_tokens, run_id,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------- game


def play_round(
    round_no: int,
    speaker: Speaker,
    listener: Listener,
    objects: list[Obj],
    cfg: Config,
    rng: random.Random,
) -> RoundResult:
    target, candidates = sample_round(objects, cfg.candidates, rng)

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

    speaker.remember(round_no, target, message, "SUCCESS" if success else "FAILED")
    listener.remember(round_no, message, candidates, picked, target)
    return RoundResult(
        round_no, target, candidates, message, True, error,
        heard.choice, picked, success, said.reasoning, heard.reasoning,
    )


def update_notes(
    conn: sqlite3.Connection, run_id: int, round_no: int, speaker: Speaker, listener: Listener
) -> None:
    """Each agent rewrites its own private notes; both versions are logged for inspection."""
    for name, agent in (("speaker", speaker), ("listener", listener)):
        agent.update_notes()
        conn.execute("INSERT INTO notes VALUES (?, ?, ?, ?)", (run_id, round_no, name, agent.notes))
    conn.commit()
    print(f"     [notes updated by both agents after round {round_no}]")


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
    brain = brain or build_brain(cfg, alphabet)

    # One brain object serves both agents: API calls are stateless, so nothing leaks.
    # Each agent's memory lives only in its own notebook.
    speaker = Speaker(brain=brain, alphabet=alphabet, max_len=cfg.max_len, memory_window=cfg.memory)
    listener = Listener(brain=brain, alphabet=alphabet, memory_window=cfg.memory)

    run_id = start_run(conn, cfg, alphabet)
    print_header(cfg, run_id, alphabet, len(objects))

    results: list[RoundResult] = []
    try:
        for round_no in range(1, cfg.rounds + 1):
            result = play_round(round_no, speaker, listener, objects, cfg, rng)
            results.append(result)
            log_round(conn, run_id, result)
            print_round(result)
            if round_no % REPORT_EVERY == 0:
                print_block(results, cfg.candidates)
            if cfg.notes_every and round_no % cfg.notes_every == 0 and round_no < cfg.rounds:
                update_notes(conn, run_id, round_no, speaker, listener)
    except KeyboardInterrupt:
        print("\nStopped by user. Saving what we have...")
    finally:
        finish_run(conn, run_id, results, brain)
        print_summary(results, cfg.candidates, brain, run_id)
    return results


# ---------------------------------------------------------------- printing


def print_header(cfg: Config, run_id: int, alphabet: list[str], n_objects: int) -> None:
    mode = "MOCK (random, free)" if cfg.mock else f"Gemini ({cfg.model})"
    print(f"Run {run_id} | {mode} | level {cfg.level} ({n_objects} objects) | seed {cfg.seed}")
    print(f"Alphabet: {' '.join(alphabet)} | {cfg.candidates} candidates | "
          f"baseline {100 / cfg.candidates:.0f}%\n")


def print_round(r: RoundResult) -> None:
    sent = " ".join(r.message) if r.message else "-"
    if not r.valid:
        outcome = f"INVALID ({r.error})"
    else:
        outcome = f"picked {r.picked.name if r.picked else '?'} " + ("OK" if r.success else "x")
    print(f"{r.round_no:4d} | {r.target.name:<16} | {sent:<7} | {outcome}")


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
    print(f"Success last {len(last):<3}: {rate(last):.1f}%")
    print(f"Invalid messages: {invalid}")
    print(f"API calls       : {u.calls}   tokens in/out: {u.input_tokens:,} / {u.output_tokens:,}")
    model = getattr(brain, "model", "mock")
    print(f"Estimated cost  : {format_cost(estimate_cost(model, u.input_tokens, u.output_tokens))}")


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(description="Alien First Contact: emergent language game")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--level", type=int, default=1, choices=[1, 2])
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--alphabet-size", type=int, default=DEFAULT_ALPHABET_SIZE)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--memory", type=int, default=20, help="recent notebook rounds shown to each agent")
    p.add_argument("--notes-every", type=int, default=20,
                   help="each agent rewrites its own private notes every N rounds (0 = off)")
    p.add_argument("--seed", type=int, default=None, help="random if omitted")
    p.add_argument("--mock", action="store_true", help="random answers, no API calls")
    p.add_argument("--db", default="runs.db")
    p.add_argument("--model", default=None, help="overrides GEMINI_MODEL from .env")
    a = p.parse_args(argv)
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(2**31)
    model = a.model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    return Config(a.rounds, a.level, a.candidates, a.alphabet_size, a.max_len, a.memory,
                  seed, a.mock, a.db, model, a.notes_every)


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
