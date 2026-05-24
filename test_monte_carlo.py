import importlib.util
import tempfile
import unittest
from pathlib import Path

import matplotlib
from matplotlib.colors import LogNorm
import numpy as np
from rich.console import Console
import io


def load_monte_carlo_module():
    module_path = Path(__file__).with_name("monte-carlo.py")
    spec = importlib.util.spec_from_file_location("monte_carlo", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlotAndSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mc = load_monte_carlo_module()

    def test_uses_non_gui_backend_for_saved_plots(self):
        self.assertEqual(matplotlib.get_backend().lower(), "agg")

    def test_run_captures_report_snapshots(self):
        param = {
            "method": "MC",
            "dim": 2,
            "N": 4,
            "plus_ratio": 0.5,
            "T": 1.0,
            "beta": 1.0,
            "step": 8,
            "sleep": 0,
            "plot": False,
            "shot_number": 0,
        }

        result = self.mc.run(
            param,
            rng_initial_state=np.random.default_rng(2026),
            console=Console(file=io.StringIO(), force_terminal=False),
        )

        snapshots = result["snapshots"]
        self.assertEqual([item["percent"] for item in snapshots], [0, 33, 66, 99])
        self.assertEqual([item["step"] for item in snapshots], [0, 3, 6, 8])
        self.assertTrue(all(item["state"].shape == (4, 4) for item in snapshots))

    def test_snapshot_panel_plot_saves_png(self):
        snapshots = [
            {
                "percent": percent,
                "step": step,
                "energy": -step,
                "state": np.ones((4, 4), dtype=int),
            }
            for percent, step in [(0, 0), (33, 3), (66, 6), (99, 8)]
        ]
        param = {"method": "MP", "N": 4, "step": 8, "shot_number": 0}

        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.mc.snapshot_panel_plot(
                snapshots,
                param=param,
                save_dir=temp_dir,
                show=False,
            )

            self.assertTrue(Path(path).exists())
            self.assertEqual(Path(path).suffix, ".png")

    def test_dashboard_state_text_is_windows_console_safe(self):
        model = self.mc.State(
            N=4,
            dim=2,
            method="MC",
            rng_initial_state=np.random.default_rng(2026),
        )

        model.make_state_plot_text().encode("cp949")

    def test_make_beta_sweep_values_descends_geometrically(self):
        beta_values = self.mc.make_beta_sweep_values(count=50, start=1.0, stop=0.0001)

        self.assertEqual(len(beta_values), 50)
        self.assertAlmostEqual(beta_values[0], 1.0)
        self.assertAlmostEqual(beta_values[-1], 0.0001)
        self.assertTrue(np.all(np.diff(beta_values) < 0))
        np.testing.assert_allclose(
            beta_values[1:] / beta_values[:-1],
            np.full(49, beta_values[1] / beta_values[0]),
        )

    def test_save_beta_sweep_results_writes_one_npz(self):
        steps = np.array([0, 1, 2])
        beta_values = np.array([1.0, 0.1])
        energy_histories = np.array([[-4, -6, -8], [-4, -4, -6]])
        param = {"method": "MP", "dim": 2, "N": 4, "step": 3}

        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.mc.save_beta_sweep_results(
                steps=steps,
                beta_values=beta_values,
                energy_histories=energy_histories,
                param=param,
                save_dir=temp_dir,
                shot_number=7,
            )

            self.assertTrue(Path(path).exists())
            with np.load(path) as data:
                np.testing.assert_array_equal(data["steps"], steps)
                np.testing.assert_array_equal(data["beta_values"], beta_values)
                np.testing.assert_array_equal(data["energy_histories_MP"], energy_histories)
                self.assertEqual(data["N"].item(), 4)
                self.assertEqual(data["dim"].item(), 2)

    def test_beta_sweep_plot_saves_one_graph(self):
        results = {
            "steps": np.array([0, 1, 2]),
            "beta_values": np.array([1.0, 0.1]),
            "energy_histories": np.array([[-4, -6, -8], [-4, -4, -6]]),
            "theoretical_lowest_energy": -32,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.mc.beta_sweep_comparison_plot_results(
                results,
                show=False,
                save_dir=temp_dir,
                shot_number=7,
            )

            self.assertTrue(Path(path).exists())
            self.assertEqual(Path(path).suffix, ".png")

    def test_beta_color_mapper_uses_log_scale(self):
        beta_values = np.array([1.0, 0.1, 0.01, 0.0001])

        cmap, norm, mappable = self.mc.make_beta_color_mapper(beta_values)

        self.assertIsInstance(norm, LogNorm)
        self.assertAlmostEqual(norm.vmin, 0.0001)
        self.assertAlmostEqual(norm.vmax, 1.0)
        self.assertIs(mappable.norm, norm)
        self.assertIs(mappable.cmap, cmap)

    def test_beta_color_mapper_uses_smooth_thermal_palette_by_default(self):
        beta_values = np.array([1.0, 0.1, 0.01, 0.0001])

        cmap, norm, _ = self.mc.make_beta_color_mapper(beta_values)
        hot_color = cmap(norm(np.min(beta_values)))
        cold_color = cmap(norm(np.max(beta_values)))

        self.assertGreater(hot_color[0], hot_color[2])
        self.assertLess(hot_color[1], 0.4)
        self.assertGreater(cold_color[2], cold_color[0])


if __name__ == "__main__":
    unittest.main()
