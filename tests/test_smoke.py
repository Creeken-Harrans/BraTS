from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RUN_PY = PROJECT_ROOT / "run.py"
DOCTOR_PY = PROJECT_ROOT / "doctor.py"


class SmokeTests(unittest.TestCase):
    maxDiff = None

    def run_command(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(RUN_PY), *args],
            cwd=WORKSPACE_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_doctor_wrapper(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(DOCTOR_PY)],
            cwd=WORKSPACE_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def assert_command_ok(self, *args: str) -> str:
        result = self.run_command(*args)
        if result.returncode != 0:
            self.fail(
                f"Command failed: python BraTS/run.py {' '.join(args)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return result.stdout

    def test_help_without_args(self) -> None:
        stdout = self.assert_command_ok()
        self.assertIn("Standalone BraTS2020 project entrypoint", stdout)
        self.assertIn("prepare-dataset", stdout)
        self.assertIn("plan-preprocess", stdout)

    def test_doctor(self) -> None:
        stdout = self.assert_command_ok("doctor")
        self.assertIn("BraTS project doctor", stdout)
        self.assertIn("dataset: Dataset220_BraTS2020", stdout)

    def test_doctor_wrapper_script(self) -> None:
        result = self.run_doctor_wrapper()
        if result.returncode != 0:
            self.fail(
                f"doctor.py failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        self.assertIn("BraTS project doctor", result.stdout)

    def test_prepare_dataset_is_idempotent(self) -> None:
        stdout = self.assert_command_ok("prepare-dataset")
        self.assertTrue(
            "[INFO] Reusing existing dataset directory:" in stdout
            or "[OK] Converted dataset written to:" in stdout
        )
        self.assertTrue(
            (
                PROJECT_ROOT / "01_data_preparation" / "metadata" / "raw_dataset.json"
            ).is_file()
        )

    def test_help_for_heavier_commands(self) -> None:
        for command in (
            "plan-preprocess",
            "train",
            "predict",
            "evaluate",
            "find-best-config",
            "report-evaluation",
        ):
            with self.subTest(command=command):
                stdout = self.assert_command_ok(command, "--help")
                self.assertIn("usage:", stdout)

    def test_prepare_dataset_help_mentions_force(self) -> None:
        stdout = self.assert_command_ok("prepare-dataset", "--help")
        self.assertIn("--force", stdout)

    def test_plan_preprocess_help_mentions_reuse_controls(self) -> None:
        stdout = self.assert_command_ok("plan-preprocess", "--help")
        self.assertIn("--recompute-fingerprint", stdout)
        self.assertIn("--recompute-plans", stdout)
        self.assertIn("--force-preprocess", stdout)

    def test_predict_help_mentions_training_sampling_controls(self) -> None:
        stdout = self.assert_command_ok("predict", "--help")
        self.assertIn("--sample-training-cases", stdout)
        self.assertIn("--sample-seed", stdout)
        self.assertIn("--disable-auto-sample-training", stdout)


if __name__ == "__main__":
    unittest.main()
