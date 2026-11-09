import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed (optional: neural baseline)")
class TestNeuralBaseline(unittest.TestCase):
    def test_games_are_valid(self) -> None:
        import torch

        from neural_baseline import sample_games
        gen = torch.Generator().manual_seed(0)
        pool = torch.arange(30)
        t, c, y = sample_games(pool, pool, 256, 4, gen)
        self.assertEqual(c.shape, (256, 4))
        self.assertTrue(bool((c.gather(1, y[:, None]).squeeze(1) == t).all()))  # label points at target
        self.assertTrue(all(len(set(row)) == 4 for row in c.tolist()))           # distinct candidates

    def test_held_out_never_in_training_games(self) -> None:
        import torch

        from neural_baseline import sample_games
        gen = torch.Generator().manual_seed(1)
        train = torch.tensor([i for i in range(30) if i not in (3, 9)])
        _, c, _ = sample_games(train, train, 512, 4, gen)
        self.assertFalse(bool(((c == 3) | (c == 9)).any()))

    def test_learns_above_chance_quickly(self) -> None:
        from neural_baseline import run_seed
        r = run_seed(level=1, seed=0, steps=300, batch=64, k=4, vocab=8, length=3,
                     holdout=0, eval_every=100, device="cpu")
        self.assertGreater(r.final_accuracy, 0.6)  # chance is 0.25


if __name__ == "__main__":
    unittest.main()
