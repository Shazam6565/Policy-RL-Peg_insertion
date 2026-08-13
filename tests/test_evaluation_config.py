import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).parents[1]
SCRIPTS_DIR = REPO_ROOT / "force_peg_rl" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluation_config import apply_suite_to_env_cfg, load_named_yaml, validate_suite  # noqa: E402
from evaluation_results import grade_summary, summarize_rows  # noqa: E402

SUITES_PATH = REPO_ROOT / "force_peg_rl" / "configs" / "evaluation_suites.yaml"
RUBRIC_PATH = REPO_ROOT / "force_peg_rl" / "configs" / "evaluation_rubric.yaml"
SUITE_NAMES = ("nominal", "pose_shift", "low_friction", "high_friction", "mass_shift", "combined_ood")


class EvaluationConfigTests(unittest.TestCase):
    def test_all_suites_and_rubrics_validate(self):
        for name in SUITE_NAMES:
            suite = load_named_yaml(SUITES_PATH, name, kind="evaluation suite")
            self.assertEqual(validate_suite(name, suite)["episodes"], suite["episodes"])
            rubric = load_named_yaml(RUBRIC_PATH, name, kind="evaluation rubric")
            summary = {
                "episodes": suite["episodes"],
                "success_rate_pct": 100.0,
                "peak_force_p95_n": 0.0,
            }
            self.assertTrue(grade_summary(summary, rubric)["passed"])

    def test_apply_suite_sets_pose_mass_and_friction(self):
        suite = validate_suite("combined_ood", load_named_yaml(SUITES_PATH, "combined_ood", kind="evaluation suite"))
        cfg = SimpleNamespace(
            task=SimpleNamespace(
                hand_init_pos_noise=[0.02, 0.02, 0.01],
                hand_init_orn_noise=[0.0, 0.0, 0.5],
                held_asset_pos_noise=[0.003, 0.0, 0.003],
            ),
            events=SimpleNamespace(
                object_scale_mass=SimpleNamespace(
                    params={"mass_distribution_params": (-0.005, 0.005), "operation": "add"}
                )
            ),
            evaluation_peg_friction_range=None,
            evaluation_socket_friction_range=None,
        )

        apply_suite_to_env_cfg(cfg, suite)

        self.assertEqual(cfg.task.hand_init_pos_noise, [0.035, 0.035, 0.01])
        self.assertAlmostEqual(cfg.task.hand_init_orn_noise[2], math.radians(70.0))
        self.assertEqual(cfg.task.held_asset_pos_noise, [0.0, 0.0, 0.003])
        self.assertEqual(cfg.events.object_scale_mass.params["mass_distribution_params"], (0.5, 1.7))
        self.assertEqual(cfg.events.object_scale_mass.params["operation"], "scale")
        self.assertEqual(cfg.evaluation_peg_friction_range, (0.2, 1.2))
        self.assertEqual(cfg.evaluation_socket_friction_range, (0.2, 1.2))

    def test_rejects_asymmetric_pose_range(self):
        suite = load_named_yaml(SUITES_PATH, "nominal", kind="evaluation suite")
        suite["conditions"]["initial_xy_offset_range_m"] = [-0.01, 0.02]
        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_suite("nominal", suite)

    def test_summary_and_rubric(self):
        rows = [
            {
                "success": True,
                "episode_steps": 10,
                "episode_return": 2.0,
                "max_contact_force": 10.0,
                "mean_contact_force": 4.0,
                "termination_reason": "success",
            },
            {
                "success": False,
                "episode_steps": 20,
                "episode_return": 0.0,
                "max_contact_force": 20.0,
                "mean_contact_force": 6.0,
                "termination_reason": "timeout",
            },
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["success_rate_pct"], 50.0)
        self.assertEqual(summary["median_completion_steps"], 15.0)
        self.assertAlmostEqual(summary["peak_force_p95_n"], 19.5)
        grade = grade_summary(summary, {"success_rate_pct": {"min": 50.0}, "peak_force_p95_n": {"max": 20.0}})
        self.assertTrue(grade["passed"])


if __name__ == "__main__":
    unittest.main()
