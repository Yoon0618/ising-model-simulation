import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np

import monte_carlo_core


ROOT = pathlib.Path(__file__).resolve().parent


def load_frontend():
    spec = importlib.util.spec_from_file_location(
        "monte_carlo_frontend",
        ROOT / "monte-carlo.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetropolisOnlyTests(unittest.TestCase):
    def test_run_simulation_no_longer_requires_method_parameter(self):
        param = {
            "dim": 2,
            "N": 4,
            "plus_ratio": 0.5,
            "T": 1.0,
            "beta": 1.0,
            "step": 5,
            "sleep": 0,
            "plot": False,
            "shot_number": 0,
        }

        result = monte_carlo_core.run_simulation(
            param,
            rng_initial_state=np.random.default_rng(seed=2026),
        )

        self.assertEqual(5, len(result["steps"]))
        self.assertEqual((4, 4), result["final_state"].shape)

    def test_save_results_uses_single_metropolis_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = {
                "steps": np.arange(2),
                "energy_history": np.array([0, -4]),
                "final_state": np.ones((2, 2), dtype=np.int64),
            }

            monte_carlo_core.save_results(
                result,
                save_dir=tmp_dir,
                shot_tracker_path=pathlib.Path(tmp_dir) / "shot_tracker.txt",
            )

            saved = list(pathlib.Path(tmp_dir).glob("*.npz"))
            self.assertEqual(1, len(saved))

            with np.load(saved[0]) as data:
                self.assertEqual(
                    {"steps", "energy_history", "final_state"},
                    set(data.files),
                )

    def test_source_no_longer_contains_mc_specific_names(self):
        frontend = load_frontend()

        source_files = [
            ROOT / "monte-carlo.py",
            ROOT / "monte_carlo_core.py",
        ]
        source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)

        old_method_label = "".join((chr(77), chr(67)))
        comparison_label = "Monte" + "-" + "Carlo"

        forbidden = [
            old_method_label,
            comparison_label,
            "results_" + old_method_label,
            "results_" + "".join(("M", "P")),
            "param_" + old_method_label,
            "param_" + "".join(("M", "P")),
            "comparison" + "_plot_results",
            "apply" + "_acceptance_rule",
        ]

        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)

        self.assertTrue(callable(frontend.run))


if __name__ == "__main__":
    unittest.main()
