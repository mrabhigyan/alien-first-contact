import random
import unittest

from world import (
    GLYPH_POOL,
    make_alphabet,
    make_objects,
    sample_round,
    validate_message,
)


class TestObjects(unittest.TestCase):
    def test_level_sizes(self) -> None:
        self.assertEqual(len(make_objects(1)), 6)
        self.assertEqual(len(make_objects(2)), 30)

    def test_names(self) -> None:
        self.assertEqual(make_objects(1)[0].name, "red")
        self.assertEqual(make_objects(2)[0].name, "red circle")

    def test_bad_level(self) -> None:
        with self.assertRaises(ValueError):
            make_objects(9)


class TestAlphabet(unittest.TestCase):
    def test_pool_has_no_duplicates(self) -> None:
        self.assertEqual(len(GLYPH_POOL), len(set(GLYPH_POOL)))

    def test_pool_has_no_ascii(self) -> None:
        self.assertTrue(all(not g.isascii() and len(g) == 1 for g in GLYPH_POOL))

    def test_alphabet_is_random_per_run(self) -> None:
        a = make_alphabet(random.Random(1))
        b = make_alphabet(random.Random(2))
        self.assertEqual(len(a), 8)
        self.assertEqual(len(set(a)), 8)
        self.assertNotEqual(a, b)


class TestValidator(unittest.TestCase):
    alphabet = ["⟁", "⧖", "⨳"]

    def test_valid(self) -> None:
        self.assertTrue(validate_message(["⟁"], self.alphabet).ok)
        self.assertTrue(validate_message(["⟁", "⧖", "⟁"], self.alphabet).ok)

    def test_empty(self) -> None:
        self.assertFalse(validate_message([], self.alphabet).ok)

    def test_too_long(self) -> None:
        self.assertFalse(validate_message(["⟁"] * 4, self.alphabet).ok)

    def test_foreign_symbol(self) -> None:
        self.assertFalse(validate_message(["red"], self.alphabet).ok)
        self.assertFalse(validate_message(["a"], self.alphabet).ok)

    def test_not_a_list(self) -> None:
        self.assertFalse(validate_message("⟁⧖", self.alphabet).ok)  # type: ignore[arg-type]


class TestSampleRound(unittest.TestCase):
    def test_target_in_distinct_candidates(self) -> None:
        rng = random.Random(0)
        objects = make_objects(2)
        for _ in range(200):
            target, cands = sample_round(objects, 4, rng)
            self.assertEqual(len(cands), 4)
            self.assertEqual(len(set(cands)), 4)
            self.assertIn(target, cands)


if __name__ == "__main__":
    unittest.main()
