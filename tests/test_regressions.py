from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "03_training_and_results" / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats_project.cli import cmd_evaluate, _resolve_existing_training_checkpoint


def load_report_module():
    module_path = PROJECT_ROOT / "04_inference_and_evaluation" / "generate_evaluation_report.py"
    spec = importlib.util.spec_from_file_location("generate_evaluation_report", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load report module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyReaderWriter:
    def read_seg(self, _: str):
        raise AssertionError("read_seg should not be called when prediction coverage validation fails early")


class RegressionTests(unittest.TestCase):
    def test_resolve_existing_training_checkpoint_prefers_final_for_resume(self) -> None:
        try:
            file_path_utilities = importlib.import_module("brats_project.utilities.file_path_utilities")
        except ModuleNotFoundError as exc:
            self.skipTest(f"checkpoint path utilities unavailable: {exc}")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for checkpoint_name in ("checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth"):
                (output_dir / checkpoint_name).write_text("stub", encoding="utf-8")

            with patch.object(file_path_utilities, "get_output_folder", return_value=str(output_dir)):
                checkpoint = _resolve_existing_training_checkpoint(
                    "Dataset220_BraTS2020",
                    "SegTrainer",
                    "ProjectPlans",
                    "3d_fullres",
                    0,
                )

        self.assertEqual(checkpoint, output_dir / "checkpoint_final.pth")

    def test_compile_helpers_detect_and_unwrap_orig_mod(self) -> None:
        try:
            from brats_project.utilities.helpers import is_compiled_module, unwrap_compiled_module
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch-dependent helpers unavailable: {exc}")

        class Wrapped:
            def __init__(self) -> None:
                self._orig_mod = object()

        wrapped = Wrapped()
        plain = object()

        self.assertTrue(is_compiled_module(wrapped))
        self.assertIs(unwrap_compiled_module(wrapped), wrapped._orig_mod)
        self.assertFalse(is_compiled_module(plain))
        self.assertIs(unwrap_compiled_module(plain), plain)

    def test_run_training_disables_resume_state_when_continue_requested_but_no_checkpoint_exists(self) -> None:
        try:
            import brats_project.run.run_training as run_training_module
        except ModuleNotFoundError as exc:
            self.skipTest(f"training dependencies missing: {exc}")

        fake_trainer = type(
            "Trainer",
            (),
            {
                "run_training": lambda self: None,
                "perform_actual_validation": lambda self, _: None,
            },
        )()

        with patch.object(
            run_training_module,
            "resolve_training_resume_checkpoint",
            return_value=None,
        ), patch.object(
            run_training_module,
            "get_trainer_from_args",
            return_value=fake_trainer,
        ) as get_trainer_mock, patch.object(
            run_training_module,
            "maybe_load_checkpoint",
        ), patch.object(
            run_training_module,
            "torch",
        ) as torch_mock:
            torch_mock.cuda.is_available.return_value = False
            run_training_module.run_training(
                dataset_name_or_id="Dataset220_BraTS2020",
                configuration="3d_fullres",
                fold=0,
                continue_training=True,
                only_run_validation=False,
                device=type("D", (), {"type": "cpu"})(),
            )

        self.assertFalse(get_trainer_mock.call_args.args[5])

    def test_compute_metrics_on_folder_rejects_empty_prediction_dir(self) -> None:
        try:
            from brats_project.evaluation.evaluate_predictions import compute_metrics_on_folder
        except ModuleNotFoundError as exc:
            self.skipTest(f"evaluation dependencies missing: {exc}")

        with tempfile.TemporaryDirectory() as gt_dir, tempfile.TemporaryDirectory() as pred_dir:
            gt_path = Path(gt_dir) / "case_000.nii.gz"
            gt_path.write_text("gt", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "No prediction files"):
                compute_metrics_on_folder(
                    gt_dir,
                    pred_dir,
                    None,
                    DummyReaderWriter(),
                    ".nii.gz",
                    [1],
                )

    def test_cmd_evaluate_requires_explicit_chill_for_subset_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pred_dir = temp_root / "pred"
            gt_dir = temp_root / "gt"
            pred_dir.mkdir()
            gt_dir.mkdir()
            (pred_dir / "case_000.nii.gz").write_text("pred", encoding="utf-8")
            (gt_dir / "case_000.nii.gz").write_text("gt", encoding="utf-8")
            (gt_dir / "case_001.nii.gz").write_text("gt", encoding="utf-8")

            dataset_json = temp_root / "dataset.json"
            dataset_json.write_text(json.dumps({"file_ending": ".nii.gz"}), encoding="utf-8")
            plans_json = temp_root / "plans.json"
            plans_json.write_text("{}", encoding="utf-8")

            args = Namespace(
                pred_dir=str(pred_dir),
                gt_dir=str(gt_dir),
                dataset_json=None,
                plans_json=None,
                trainer="SegTrainer",
                configuration="3d_fullres",
                plans="ProjectPlans",
                output_file=None,
                num_processes=1,
                chill=False,
            )
            fake_eval_module = types.ModuleType("brats_project.evaluation.evaluate_predictions")
            fake_eval_module.compute_metrics_on_folder2 = unittest.mock.Mock()

            with patch("brats_project.cli._resolve_predict_defaults_from_inference_info", return_value=("SegTrainer", "3d_fullres", "ProjectPlans", ["0"])), \
                 patch("brats_project.cli._resolve_evaluation_metadata", return_value=(dataset_json, plans_json)), \
                 patch.dict(sys.modules, {"brats_project.evaluation.evaluate_predictions": fake_eval_module}):
                with self.assertRaisesRegex(RuntimeError, "rerun with --chill"):
                    cmd_evaluate(args)
                fake_eval_module.compute_metrics_on_folder2.assert_not_called()

    def test_determine_comparable_folds_uses_shared_intersection(self) -> None:
        try:
            from brats_project.evaluation.find_best_configuration import determine_comparable_folds
        except ModuleNotFoundError as exc:
            self.skipTest(f"best-config dependencies missing: {exc}")

        models = [
            {"trainer": "SegTrainer", "plans": "ProjectPlans", "configuration": "3d_fullres"},
            {"trainer": "SegTrainer", "plans": "ProjectPlans", "configuration": "2d"},
        ]
        def fake_output_folder(_: str, trainer: str, plans: str, configuration: str, fold=None) -> str:
            return f"/tmp/Dataset220/{trainer}__{plans}__{configuration}"

        with patch(
            "brats_project.evaluation.find_best_configuration.get_output_folder",
            side_effect=fake_output_folder,
        ), patch(
            "brats_project.evaluation.find_best_configuration.get_available_validation_folds",
            side_effect=[(0, 1, 2), (1, 2, 3)],
        ):
            comparable_models, comparable_folds = determine_comparable_folds("Dataset220_BraTS2020", models, (0, 1, 2, 3))

        self.assertEqual(comparable_models, models)
        self.assertEqual(comparable_folds, (1, 2))

    def test_manual_initialization_allows_compile_for_uncompiled_ddp_module(self) -> None:
        try:
            import brats_project.inference.predict_from_raw_data as predict_module
        except ModuleNotFoundError as exc:
            self.skipTest(f"inference dependencies missing: {exc}")

        predictor = predict_module.nnUNetPredictor(device=type("D", (), {"type": "cuda"})())
        predictor.network = None

        class PlainModule:
            pass

        class FakeDDP:
            def __init__(self, module):
                self.module = module

        fake_network = FakeDDP(PlainModule())

        with patch.object(predict_module, "DistributedDataParallel", FakeDDP), \
             patch.object(predict_module, "torch") as torch_mock, \
             patch.dict("os.environ", {"PROJECT_COMPILE": "true"}, clear=False):
            torch_mock.compile.side_effect = lambda module: ("compiled", module)
            predictor.manual_initialization(
                network=fake_network,
                plans_manager=type("Plans", (), {"get_label_manager": lambda self, _: "lm"})(),
                configuration_manager=object(),
                parameters=None,
                dataset_json={},
                trainer_name="SegTrainer",
                inference_allowed_mirroring_axes=None,
            )

        self.assertEqual(predictor.network, ("compiled", fake_network))

    def test_parse_case_metrics_keeps_undefined_precision_and_uses_nanmean_for_case_rank(self) -> None:
        try:
            report_module = load_report_module()
        except ModuleNotFoundError as exc:
            self.skipTest(f"report dependencies missing: {exc}")

        summary = {
            "metric_per_case": [
                {
                    "prediction_file": str(PROJECT_ROOT / "pred_case.nii.gz"),
                    "reference_file": str(PROJECT_ROOT / "ref_case.nii.gz"),
                    "metrics": {
                        "(1, 2, 3)": {"Dice": 0.8, "TP": 8, "FP": 2, "FN": 2},
                        "(2, 3)": {"Dice": 0.6, "TP": 6, "FP": 4, "FN": 4},
                        "(3,)": {"Dice": float("nan"), "TP": 0, "FP": 0, "FN": 0},
                    },
                }
            ]
        }

        case_metrics = report_module.parse_case_metrics(summary)
        self.assertEqual(len(case_metrics), 1)
        self.assertAlmostEqual(case_metrics[0]["dice_mean"], 0.7)
        self.assertTrue(report_module.math.isnan(case_metrics[0]["precision_et"]))
        self.assertTrue(report_module.math.isnan(case_metrics[0]["recall_et"]))


if __name__ == "__main__":
    unittest.main()
