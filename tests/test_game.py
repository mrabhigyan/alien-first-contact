import io
import json
import unittest
from contextlib import redirect_stdout

from agents import ListenerOut, SpeakerOut, Usage
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


if __name__ == "__main__":
    unittest.main()
