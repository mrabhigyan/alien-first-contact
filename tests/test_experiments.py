import io
import json
import random
import unittest
from contextlib import redirect_stdout

from agents import ListenerOut, NotesOut, SpeakerOut, Usage
from main import Config, choose_held_out, open_db, run
from world import make_objects


def config(**overrides) -> Config:
    base = dict(rounds=40, level=2, candidates=4, alphabet_size=8, max_len=3, memory=20,
                seed=11, mock=True, db=":memory:", model="mock", notes_every=0)
    base.update(overrides)
    return Config(**base)


def quiet_run(cfg: Config, brain=None):
    conn = open_db(":memory:")
    with redirect_stdout(io.StringIO()):
        results = run(cfg, conn, brain)
    return results, conn


class AlwaysWrongBrain:
    """Valid message, but the listener always picks option 1 (often wrong)."""

    def __init__(self) -> None:
        self.usage = Usage()
        self.speaker_prompts: list[str] = []

    def ask(self, system, prompt, schema):
        self.usage.calls += 1
        if schema is SpeakerOut:
            self.speaker_prompts.append(prompt)
            return SpeakerOut(reasoning="", message=[system.split("\n\n")[1].split()[0]])
        if schema is ListenerOut:
            return ListenerOut(reasoning="", choice=1)
        return NotesOut(reasoning="", notes="n")


class TestHoldout(unittest.TestCase):
    def test_held_out_objects_are_describable_and_distinct(self) -> None:
        objects = make_objects(2)
        held = choose_held_out(objects, 5, random.Random(3))
        self.assertEqual(len(held), 5)
        self.assertEqual(len({o.color for o in held}), 5)   # no shared colors
        self.assertEqual(len({o.shape for o in held}), 5)   # no shared shapes
        training = [o for o in objects if o not in held]
        for o in held:
            self.assertTrue(any(t.color == o.color for t in training))
            self.assertTrue(any(t.shape == o.shape for t in training))

    def test_never_seen_in_training_then_tested(self) -> None:
        results, conn = quiet_run(config(experiment="holdout", phase2_rounds=15))
        held = set(json.loads(conn.execute("SELECT held_out FROM runs").fetchone()[0]))
        self.assertEqual(len(held), 5)
        train = [r for r in results if r.phase == "train"]
        test = [r for r in results if r.phase == "test"]
        self.assertEqual((len(train), len(test)), (40, 15))
        for r in train:  # not as target, not even as a candidate
            self.assertFalse({c.name for c in r.candidates} & held)
        self.assertTrue(all(r.target.name in held for r in test))

    def test_level1_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quiet_run(config(level=1, experiment="holdout", phase2_rounds=5))


class TestNewcomer(unittest.TestCase):
    def test_fresh_listener_joins_in_phase_two(self) -> None:
        results, _ = quiet_run(config(experiment="newcomer", rounds=20, phase2_rounds=10))
        self.assertTrue(all(r.listener_id == "A" for r in results[:20]))
        self.assertTrue(all((r.speaker_id, r.listener_id) == ("A", "N") for r in results[20:]))


class TestTribes(unittest.TestCase):
    def test_pairs_train_apart_then_cross(self) -> None:
        results, _ = quiet_run(config(experiment="tribes", rounds=20, phase2_rounds=10))
        train, cross = results[:20], results[20:]
        self.assertEqual({(r.speaker_id, r.listener_id) for r in train}, {("A", "A"), ("B", "B")})
        self.assertEqual({(r.speaker_id, r.listener_id) for r in cross}, {("A", "B"), ("B", "A")})
        self.assertTrue(all(r.phase == "cross" for r in cross))

    def test_each_agent_only_remembers_its_own_rounds(self) -> None:
        brain = AlwaysWrongBrain()
        quiet_run(config(experiment="tribes", rounds=10, phase2_rounds=0, mock=False), brain)
        # Speaker A plays rounds 1,3,5,...; its last prompt must not mention round 2 entries.
        last_a = brain.speaker_prompts[-2]
        self.assertIn("Round 7:", last_a)
        self.assertNotIn("Round 8:", last_a)
        self.assertNotIn("Round 6:", last_a)


class TestFeedback(unittest.TestCase):
    def _speaker_saw_pick(self, feedback: str) -> bool:
        brain = AlwaysWrongBrain()
        quiet_run(config(rounds=15, mock=False, feedback=feedback), brain)
        return any("partner picked" in p for p in brain.speaker_prompts)

    def test_full_feedback_tells_speaker_what_was_picked(self) -> None:
        self.assertTrue(self._speaker_saw_pick("full"))

    def test_outcome_feedback_does_not(self) -> None:
        self.assertFalse(self._speaker_saw_pick("outcome"))


if __name__ == "__main__":
    unittest.main()
