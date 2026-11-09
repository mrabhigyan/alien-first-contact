import unittest

from metrics import edit_distance, meaning_distance, topsim, topsim_test
from world import COLORS, SHAPES, Cell, Obj, make_objects


class TestMetrics(unittest.TestCase):
    def test_edit_distance(self) -> None:
        self.assertEqual(edit_distance(["a", "b"], ["a", "b"]), 0)
        self.assertEqual(edit_distance(["a", "b"], ["a", "c"]), 1)
        self.assertEqual(edit_distance(["a"], ["a", "b", "b"]), 2)
        self.assertEqual(edit_distance([], ["x"]), 1)

    def test_meaning_distance(self) -> None:
        self.assertEqual(meaning_distance(Obj("red", "star"), Obj("red", "circle")), 1)
        self.assertEqual(meaning_distance(Obj("red", "star"), Obj("blue", "circle")), 2)
        self.assertEqual(meaning_distance(Cell(1, 2), Cell(-1, 0)), 4)

    def test_perfectly_compositional_language_scores_high(self) -> None:
        colors = dict(zip(COLORS, "abcdef", strict=True))
        shapes = dict(zip(SHAPES, "vwxyz", strict=True))
        lexicon = {o: [colors[o.color], shapes[o.shape]] for o in make_objects(2)}
        result = topsim_test(lexicon, shuffles=200)
        self.assertGreater(result.score, 0.95)
        self.assertLess(result.shuffled_mean, 0.2)
        self.assertLess(result.p_value, 0.01)

    def test_level1_undefined(self) -> None:
        lexicon = {o: [s] for o, s in zip(make_objects(1), "abcdef", strict=True)}
        self.assertIsNone(topsim(lexicon))


if __name__ == "__main__":
    unittest.main()
