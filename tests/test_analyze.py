import json
import unittest

from analyze import dictionary, rolling_rate


class TestAnalyze(unittest.TestCase):
    def test_rolling_rate(self) -> None:
        self.assertEqual(rolling_rate([0, 1, 1, 1], window=2), [0.0, 50.0, 100.0, 100.0])

    def test_dictionary_skips_early_rounds(self) -> None:
        rounds = [
            (1, "red", json.dumps(["⍩"]), 0),
            (2, "red", json.dumps(["⍢"]), 1),
            (3, "red", json.dumps(["⍢"]), 1),
        ]
        usage = dictionary(rounds, skip=1)
        self.assertEqual(usage["red"].most_common(1), [("⍢", 2)])


if __name__ == "__main__":
    unittest.main()
