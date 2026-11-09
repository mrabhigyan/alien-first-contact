import io
import json
import unittest
from contextlib import redirect_stdout

from agents import ListenerOut, NotesOut, SpeakerOut, Usage
from main import Config, open_db, run


def config(**overrides) -> Config:
    base = dict(rounds=30, level=1, candidates=4, alphabet_size=8, max_len=3,
                memory=40, seed=123, mock=True, db=":memory:", model="mock")
    base.update(overrides)
    return Config(**base)


def quiet_run(cfg: Config, brain=None):
    conn = open_db(":memory:")
    with redirect_stdout(io.StringIO()):
        results = run(cfg, conn, brain)
    return results, conn


class CheatingBrain:
    """Speaker writes English. The listener must never be asked."""

    def __init__(self) -> None:
        self.usage = Usage()
        self.listener_calls = 0

    def ask(self, system, prompt, schema):
        self.usage.calls += 1
        if schema is SpeakerOut:
            return SpeakerOut(reasoning="cheat", message=["red"])
        self.listener_calls += 1
        return ListenerOut(reasoning="", choice=1)


class TestMockRun(unittest.TestCase):
    def test_all_rounds_logged(self) -> None:
        results, conn = quiet_run(config())
        self.assertEqual(len(results), 30)
        (n,) = conn.execute("SELECT COUNT(*) FROM rounds").fetchone()
        self.assertEqual(n, 30)
        row = conn.execute("SELECT rounds_played, api_calls FROM runs").fetchone()
        self.assertEqual(row, (30, 60))  # 2 calls per round

    def test_mock_messages_are_valid(self) -> None:
        results, _ = quiet_run(config())
        self.assertTrue(all(r.valid for r in results))

    def test_same_seed_same_run(self) -> None:
        a, _ = quiet_run(config(seed=7))
        b, _ = quiet_run(config(seed=7))
        self.assertEqual([(r.target, r.message, r.choice) for r in a],
                         [(r.target, r.message, r.choice) for r in b])

    def test_level_3(self) -> None:
        results, _ = quiet_run(config(level=3, rounds=10))
        self.assertTrue(all(hasattr(r.target, "dx") for r in results))

    def test_level_2(self) -> None:
        results, _ = quiet_run(config(level=2, rounds=10))
        self.assertTrue(all(r.target.shape for r in results))


class TestRulesEnforced(unittest.TestCase):
    def test_invalid_message_fails_and_never_reaches_listener(self) -> None:
        brain = CheatingBrain()
        results, conn = quiet_run(config(rounds=5, mock=False), brain)
        self.assertTrue(all(not r.valid and not r.success for r in results))
        self.assertEqual(brain.listener_calls, 0)
        msgs = [json.loads(m) for (m,) in conn.execute("SELECT message FROM rounds")]
        self.assertEqual(msgs, [["red"]] * 5)


class RecordingBrain:
    """Valid answers; remembers every prompt so tests can check what each agent saw."""

    def __init__(self, alphabet_symbol_source) -> None:
        self.usage = Usage()
        self.prompts: list[tuple[str, str]] = []  # (schema name, prompt)
        self.notes_written = 0
        self.alphabet_symbol_source = alphabet_symbol_source

    def ask(self, system, prompt, schema):
        self.usage.calls += 1
        self.prompts.append((schema.__name__, prompt))
        if schema is SpeakerOut:
            symbol = system.split("\n\n")[1].split()[0]  # first symbol of this run's alphabet
            return SpeakerOut(reasoning="", message=[symbol])
        if schema is ListenerOut:
            return ListenerOut(reasoning="", choice=1)
        self.notes_written += 1
        # A unique marker per write, so leaks are detectable.
        marker = "SPEAKER-SECRET" if "I sent" in prompt else "LISTENER-SECRET"
        return NotesOut(reasoning="", notes=f"{marker}-{self.notes_written}")


class TestNotes(unittest.TestCase):
    def test_notes_written_logged_and_private(self) -> None:
        brain = RecordingBrain(None)
        results, conn = quiet_run(config(rounds=25, mock=False, notes_every=10), brain)
        self.assertEqual(len(results), 25)
        # Rounds 10 and 20 -> 2 updates x 2 agents.
        rows = conn.execute("SELECT round_no, agent FROM notes ORDER BY round_no, agent").fetchall()
        self.assertEqual(rows, [(10, "listener"), (10, "speaker"), (20, "listener"), (20, "speaker")])
        # Each agent sees its own notes, never the other's.
        speaker_prompts = [p for s, p in brain.prompts if s == "SpeakerOut"]
        listener_prompts = [p for s, p in brain.prompts if s == "ListenerOut"]
        self.assertTrue(any("SPEAKER-SECRET" in p for p in speaker_prompts))
        self.assertTrue(any("LISTENER-SECRET" in p for p in listener_prompts))
        self.assertFalse(any("LISTENER-SECRET" in p for p in speaker_prompts))
        self.assertFalse(any("SPEAKER-SECRET" in p for p in listener_prompts))

    def test_notes_off(self) -> None:
        _, conn = quiet_run(config(rounds=25, notes_every=0))
        (n,) = conn.execute("SELECT COUNT(*) FROM notes").fetchone()
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
