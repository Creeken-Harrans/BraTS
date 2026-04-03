from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from brats_project.project_layout import (
    configure_environment,
    get_project_root,
    get_workspace_root,
    load_project_config,
    resolve_workspace_path,
    workspace_relative_string,
)


def _project_file(relative_path: str) -> Path:
    return get_project_root() / relative_path


def _default_evaluation_output_file() -> Path:
    output_dir = _project_file("04_inference_and_evaluation/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "summary.json"


def _load_external_module(relative_path: str, module_name: str) -> Any:
    module_path = _project_file(relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_dataset_defaults() -> dict[str, Any]:
    config = load_project_config()
    return config["dataset"]


def _call_entrypoint(entrypoint: Any, argv: list[str]) -> None:
    old_argv = sys.argv[:]
    sys.argv = ["brats", *argv]
    try:
        entrypoint()
    finally:
        sys.argv = old_argv


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _prepare_preprocess_log(dataset_name: str, plans_name: str) -> Path:
    logs_dir = _project_file("02_preprocess/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"preprocess_{dataset_name}_{plans_name}_{timestamp}.log"
    log_path.write_text("", encoding="utf-8")
    latest_log = logs_dir / "latest.log"
    if latest_log.exists() or latest_log.is_symlink():
        latest_log.unlink()
    latest_log.symlink_to(log_path.name)
    return log_path


_PREDICTION_INPUT_AUXILIARY_FILES = {"README.md", "sample_selection.json"}


def _directory_has_nifti_files(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.nii.gz"))


def _directory_has_only_auxiliary_files(directory: Path) -> bool:
    if not directory.is_dir():
        return True
    return all(
        path.name in _PREDICTION_INPUT_AUXILIARY_FILES for path in directory.iterdir()
    )


def _gather_training_cases(dataset_name: str) -> dict[str, list[Path]]:
    env = configure_environment()
    images_tr_dir = Path(env["PROJECT_RAW"]) / dataset_name / "imagesTr"
    labels_tr_dir = Path(env["PROJECT_RAW"]) / dataset_name / "labelsTr"
    case_files: dict[str, list[Path]] = {}
    for image_file in sorted(images_tr_dir.glob("*.nii.gz")):
        if len(image_file.name) <= 12 or image_file.name[-12] != "_":
            continue
        case_id = image_file.name[:-12]
        case_files.setdefault(case_id, []).append(image_file)

    valid_cases: dict[str, list[Path]] = {}
    for case_id, modality_files in case_files.items():
        if len(modality_files) != 4:
            continue
        if not (labels_tr_dir / f"{case_id}.nii.gz").is_file():
            continue
        valid_cases[case_id] = sorted(modality_files)
    return valid_cases


def _prepare_sampled_prediction_input(
    dataset_name: str,
    input_dir: Path,
    sample_count: int,
    sample_seed: int,
) -> list[str]:
    case_files = _gather_training_cases(dataset_name)
    available_cases = sorted(case_files)
    if len(available_cases) == 0:
        raise RuntimeError(
            "Unable to sample validation inputs from the training set because no complete imagesTr/labelsTr "
            f"cases were found for {dataset_name}."
        )
    if sample_count > len(available_cases):
        raise RuntimeError(
            f"Requested {sample_count} sampled validation cases, but only {len(available_cases)} complete "
            f"training cases are available in {dataset_name}."
        )
    if _directory_has_nifti_files(input_dir) or not _directory_has_only_auxiliary_files(
        input_dir
    ):
        raise RuntimeError(
            f"Cannot auto-populate prediction input because the target directory is not empty: {input_dir}\n"
            "Only auxiliary files such as README.md may already exist there. Remove the existing input payload first."
        )

    selected_case_ids = sorted(
        random.Random(sample_seed).sample(list(case_files), sample_count)
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    for case_id in selected_case_ids:
        for source in case_files[case_id]:
            shutil.copy2(source, input_dir / source.name)

    sample_metadata = {
        "source": "PROJECT_RAW/imagesTr",
        "dataset_name": dataset_name,
        "sample_seed": sample_seed,
        "sample_count": sample_count,
        "selected_case_ids": selected_case_ids,
    }
    (input_dir / "sample_selection.json").write_text(
        json.dumps(sample_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[INFO] Input directory was empty. Sampled {sample_count} training cases into "
        f"{workspace_relative_string(input_dir)} with seed {sample_seed}."
    )
    return selected_case_ids


def _resolve_existing_training_checkpoint(
    dataset_name: str,
    trainer: str,
    plans: str,
    configuration: str,
    fold: str | int,
) -> Path | None:
    configure_environment()
    from brats_project.run.run_training import resolve_training_resume_checkpoint

    checkpoint = resolve_training_resume_checkpoint(
        dataset_name, configuration, fold, trainer, plans
    )
    return Path(checkpoint) if checkpoint is not None else None


def _sync_preprocess_metadata(dataset_name: str, plans_name: str) -> None:
    env = configure_environment()
    raw_dataset_dir = Path(env["PROJECT_RAW"]) / dataset_name
    preprocessed_dir = Path(env["PROJECT_PREPROCESSED"]) / dataset_name
    metadata_dir = _project_file("02_preprocess/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(raw_dataset_dir / "dataset.json", metadata_dir / "dataset.json")
    _copy_if_exists(
        raw_dataset_dir / "dataset.json",
        _project_file("01_data_preparation/metadata/raw_dataset.json"),
    )
    _copy_if_exists(
        preprocessed_dir / "dataset_fingerprint.json",
        metadata_dir / "dataset_fingerprint.json",
    )
    _copy_if_exists(
        preprocessed_dir / "splits_final.json", metadata_dir / "splits_final.json"
    )
    _copy_if_exists(
        preprocessed_dir / f"{plans_name}.json", metadata_dir / f"{plans_name}.json"
    )
    _copy_if_exists(
        preprocessed_dir / "nnUNetPlans.json", metadata_dir / "nnUNetPlans.json"
    )


def _sync_training_snapshot(
    dataset_name: str, trainer: str, plans: str, configuration: str, fold: int
) -> None:
    configure_environment()
    from brats_project.utilities.file_path_utilities import get_output_folder

    source_fold_dir = Path(
        get_output_folder(dataset_name, trainer, plans, configuration, fold=fold)
    )
    target_fold_dir = _project_file(f"03_training_and_results/results/fold_{fold}")
    if not source_fold_dir.is_dir():
        return

    target_fold_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("debug.json", "progress.png", "training_log_*.txt", "summary.json"):
        for source in source_fold_dir.glob(pattern):
            _copy_if_exists(source, target_fold_dir / source.name)


def _build_device(device_name: str) -> Any:
    import torch

    if device_name == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
        return torch.device("cpu")
    if device_name == "mps":
        return torch.device("mps")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch.device("cuda")


def _resolve_evaluation_metadata(
    pred_dir: Path,
    dataset_json: str | None,
    plans_json: str | None,
    trainer: str,
    plans: str,
    configuration: str,
) -> tuple[Path, Path]:
    configure_environment()
    dataset_json_path = (
        resolve_workspace_path(dataset_json)
        if dataset_json
        else pred_dir / "dataset.json"
    )
    plans_json_path = (
        resolve_workspace_path(plans_json) if plans_json else pred_dir / "plans.json"
    )

    if dataset_json_path.is_file() and plans_json_path.is_file():
        return dataset_json_path, plans_json_path

    dataset_name = _get_dataset_defaults()["name"]
    from brats_project.utilities.file_path_utilities import get_output_folder

    model_output_dir = Path(
        get_output_folder(dataset_name, trainer, plans, configuration)
    )
    fallback_dataset_json = model_output_dir / "dataset.json"
    fallback_plans_json = model_output_dir / "plans.json"

    if not dataset_json_path.is_file():
        dataset_json_path = fallback_dataset_json
    if not plans_json_path.is_file():
        plans_json_path = fallback_plans_json

    missing = [
        str(path) for path in (dataset_json_path, plans_json_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Unable to locate evaluation metadata files. Checked prediction folder first and then the "
            f"trained model folder. Missing: {missing}"
        )
    return dataset_json_path, plans_json_path


def _load_inference_information(dataset_name: str) -> dict[str, Any] | None:
    configure_environment()
    inference_info = (
        Path(os.environ["PROJECT_RESULTS"])
        / dataset_name
        / "inference_information.json"
    )
    if not inference_info.is_file():
        return None
    return json.loads(inference_info.read_text(encoding="utf-8"))


def _resolve_predict_defaults_from_inference_info(
    dataset_name: str,
    args: argparse.Namespace,
    dataset_defaults: dict[str, Any],
) -> tuple[str, str, str, list[str]]:
    trainer = args.trainer
    configuration = args.configuration
    plans = args.plans
    folds = [str(fold) for fold in args.folds] if args.folds else []

    user_overrode_model = any(
        [
            trainer != dataset_defaults["trainer"],
            configuration != dataset_defaults["default_configuration"],
            plans != dataset_defaults["plans"],
            folds != [str(fold) for fold in dataset_defaults["default_folds"]],
        ]
    )
    if user_overrode_model:
        return trainer, configuration, plans, folds

    inference_info = _load_inference_information(dataset_name)
    if inference_info is None:
        return trainer, configuration, plans, folds

    selected_models = inference_info.get("best_model_or_ensemble", {}).get(
        "selected_model_or_models", []
    )
    if len(selected_models) != 1:
        raise RuntimeError(
            "The current inference_information.json points to an ensemble. "
            "Automatic `run.py predict` fallback only supports a single best model. "
            "Please pass --trainer/--configuration/--plans explicitly, or run each ensemble member manually."
        )

    selected = selected_models[0]
    selected_folds = [str(fold) for fold in inference_info.get("folds", [])]
    if selected_folds:
        folds = selected_folds
    trainer = selected["trainer"]
    configuration = selected["configuration"]
    plans = selected["plans_identifier"]
    print(
        "[INFO] Using best model from inference_information.json: "
        f"trainer={trainer}, plans={plans}, configuration={configuration}, folds={folds}"
    )
    return trainer, configuration, plans, folds


def _preprocessed_configuration_looks_complete(
    preprocessed_dir: Path,
    configuration_manager: Any,
    expected_case_count: int,
) -> bool:
    output_dir = preprocessed_dir / configuration_manager.data_identifier
    if not output_dir.is_dir():
        return False

    data_b2nd_files = [
        p for p in output_dir.glob("*.b2nd") if not p.name.endswith("_seg.b2nd")
    ]
    seg_b2nd_files = list(output_dir.glob("*_seg.b2nd"))
    npz_files = list(output_dir.glob("*.npz"))
    properties_files = list(output_dir.glob("*.pkl"))

    has_complete_b2nd = (
        len(data_b2nd_files) == expected_case_count
        and len(seg_b2nd_files) == expected_case_count
        and len(properties_files) == expected_case_count
    )
    has_complete_npz = (
        len(npz_files) == expected_case_count
        and len(properties_files) == expected_case_count
    )
    return has_complete_b2nd or has_complete_npz


def cmd_doctor(_: argparse.Namespace) -> int:
    env = configure_environment()
    config = load_project_config()
    dataset = config["dataset"]
    paths = config["paths"]
    runtime = config["runtime"]
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    print("BraTS project doctor")
    print(f"path_base: {workspace_relative_string(get_project_root())}")
    print(f"project_root: {workspace_relative_string(get_project_root())}")
    print(f"dataset: {dataset['name']} (id={dataset['id']})")
    print(f"archive_root: {paths['archive_root']}")
    print(f"project_raw_root: {workspace_relative_string(env['PROJECT_RAW'])}")
    print(
        f"project_preprocessed_root: {workspace_relative_string(env['PROJECT_PREPROCESSED'])}"
    )
    print(f"project_results_root: {workspace_relative_string(env['PROJECT_RESULTS'])}")
    print(f"torch_expected: {runtime['torch']}")
    print(f"cuda_expected: {runtime['cuda']}")
    if torch is None:
        print("torch_installed: missing")
        print("cuda_available: unknown")
    else:
        print(f"torch_installed: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            print(f"cuda_version: {cuda_version}")
            print(f"current_device: {torch.cuda.get_device_name(0)}")

    archive_root = resolve_workspace_path(paths["archive_root"])
    print(f"archive_exists: {archive_root.is_dir()}")
    print(
        f"dataset_raw_exists: {(Path(env['PROJECT_RAW']) / dataset['name']).is_dir()}"
    )
    print(
        f"dataset_preprocessed_exists: {(Path(env['PROJECT_PREPROCESSED']) / dataset['name']).is_dir()}"
    )
    return 0


def cmd_visualize_first_case(args: argparse.Namespace) -> int:
    module = _load_external_module(
        "00_first_case_visualization/visualize_first_case.py",
        "visualize_first_case",
    )
    module.main(
        data_root=resolve_workspace_path(args.data_root),
        output_dir=resolve_workspace_path(args.output_dir),
    )
    return 0


def cmd_prepare_dataset(args: argparse.Namespace) -> int:
    module = _load_external_module(
        "01_data_preparation/scripts/prepare_brats2020_for_project.py",
        "prepare_brats2020_for_project",
    )
    configure_environment()
    module.main(
        src_root=resolve_workspace_path(args.src_root),
        project_raw=resolve_workspace_path(args.project_raw),
        force=args.force,
    )
    dataset_name = _get_dataset_defaults()["name"]
    _sync_preprocess_metadata(dataset_name, args.plans)
    return 0


def cmd_plan_preprocess(args: argparse.Namespace) -> int:
    env = configure_environment()
    from brats_project.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints,
        plan_experiments,
        preprocess,
    )
    from brats_project.experiment_planning.verify_dataset_integrity import (
        verify_dataset_integrity,
    )
    from brats_project.utilities.dataset_name_id_conversion import (
        maybe_convert_to_dataset_name,
    )
    from brats_project.utilities.plans_handling.plans_handler import PlansManager

    dataset = _get_dataset_defaults()
    dataset_id = args.dataset_id or dataset["id"]
    dataset_name = maybe_convert_to_dataset_name(dataset_id)
    preprocessed_dir = Path(env["PROJECT_PREPROCESSED"]) / dataset_name
    raw_dataset_dir = Path(env["PROJECT_RAW"]) / dataset_name
    fingerprint_file = preprocessed_dir / "dataset_fingerprint.json"
    plans_file = preprocessed_dir / f"{args.plans}.json"
    raw_dataset_json = json.loads(
        (raw_dataset_dir / "dataset.json").read_text(encoding="utf-8")
    )
    expected_case_count = int(raw_dataset_json["numTraining"])
    preprocess_log_path = _prepare_preprocess_log(dataset_name, args.plans)
    os.environ["PROJECT_PREPROCESS_LOG"] = str(preprocess_log_path)
    print(
        f"[INFO] Preprocess detail log: {workspace_relative_string(preprocess_log_path)}"
    )

    try:
        if args.verify_dataset:
            verify_dataset_integrity(str(raw_dataset_dir), args.num_processes)

        should_extract_fingerprint = (
            args.clean or args.recompute_fingerprint or not fingerprint_file.is_file()
        )
        if should_extract_fingerprint:
            extract_fingerprints(
                [dataset_id],
                num_processes=args.num_processes,
                check_dataset_integrity=False,
                clean=True,
                verbose=not args.quiet,
                show_progress_bar=not args.disable_progress_bar,
            )
        else:
            print(f"[INFO] Reusing existing dataset fingerprint: {fingerprint_file}")

        should_plan = args.clean or args.recompute_plans or not plans_file.is_file()
        if should_plan:
            plan_experiments(
                [dataset_id],
                gpu_memory_target_in_gb=args.gpu_memory_target,
                overwrite_plans_name=args.plans,
            )
        else:
            print(f"[INFO] Reusing existing plans file: {plans_file}")

        configs_to_preprocess = list(args.configurations)
        if not args.force_preprocess:
            plans_manager = PlansManager(str(plans_file))
            pending_configurations: list[str] = []
            for configuration in args.configurations:
                if configuration not in plans_manager.available_configurations:
                    pending_configurations.append(configuration)
                    continue
                configuration_manager = plans_manager.get_configuration(configuration)
                output_dir = preprocessed_dir / configuration_manager.data_identifier
                if _preprocessed_configuration_looks_complete(
                    preprocessed_dir,
                    configuration_manager,
                    expected_case_count,
                ):
                    print(
                        f"[INFO] Reusing existing preprocessed data for {configuration}: {output_dir}"
                    )
                    continue
                if output_dir.is_dir():
                    print(
                        f"[INFO] Existing preprocessed data for {configuration} is incomplete. "
                        f"Regenerating: {output_dir}"
                    )
                pending_configurations.append(configuration)
            configs_to_preprocess = pending_configurations

        if configs_to_preprocess:
            preprocess(
                [dataset_id],
                plans_identifier=args.plans,
                configurations=configs_to_preprocess,
                num_processes=args.preprocess_processes,
                verbose=not args.quiet,
                show_progress_bar=not args.disable_progress_bar,
            )
        else:
            print(
                "[INFO] All requested preprocessing outputs already exist. Skipping preprocess."
            )

        _sync_preprocess_metadata(dataset_name, args.plans)
        return 0
    finally:
        os.environ.pop("PROJECT_PREPROCESS_LOG", None)


def cmd_train(args: argparse.Namespace) -> int:
    configure_environment()
    from brats_project.run.run_training import run_training
    from brats_project.utilities.plans_handling.plans_handler import PlansManager

    dataset = _get_dataset_defaults()
    preprocessed_dir = Path(os.environ["PROJECT_PREPROCESSED"]) / dataset["name"]
    dataset_json = json.loads(
        (preprocessed_dir / "dataset.json").read_text(encoding="utf-8")
    )
    expected_case_count = int(dataset_json["numTraining"])
    plans_manager = PlansManager(str(preprocessed_dir / f"{args.plans}.json"))
    configuration_manager = plans_manager.get_configuration(args.configuration)
    if not _preprocessed_configuration_looks_complete(
        preprocessed_dir,
        configuration_manager,
        expected_case_count,
    ):
        raise RuntimeError(
            "Preprocessed training data is missing or incomplete for the requested configuration.\n"
            f"Expected complete directory: {preprocessed_dir / configuration_manager.data_identifier}\n"
            "Repair it with:\n"
            f"python BraTS/run.py plan-preprocess --plans {args.plans} "
            f"--configurations {args.configuration} --force-preprocess"
        )
    continue_training = args.continue_training
    if (
        not args.restart_training
        and not args.validation_only
        and args.pretrained_weights is None
    ):
        existing_checkpoint = _resolve_existing_training_checkpoint(
            dataset["name"], args.trainer, args.plans, args.configuration, args.fold
        )
        if existing_checkpoint is not None:
            continue_training = True
            print(
                f"[INFO] Found existing checkpoint, resuming training from: {existing_checkpoint}"
            )
        else:
            print(
                "[INFO] No existing checkpoint found. Starting a new training run from scratch."
            )
    run_training(
        dataset_name_or_id=dataset["name"],
        configuration=args.configuration,
        fold=args.fold,
        trainer_class_name=args.trainer,
        plans_identifier=args.plans,
        pretrained_weights=args.pretrained_weights,
        num_gpus=args.num_gpus,
        export_validation_probabilities=args.npz,
        continue_training=continue_training,
        only_run_validation=args.validation_only,
        disable_checkpointing=args.disable_checkpointing,
        val_with_best=args.val_best,
        device=_build_device(args.device),
    )
    if args.fold != "all":
        _sync_training_snapshot(
            dataset["name"],
            args.trainer,
            args.plans,
            args.configuration,
            int(args.fold),
        )
    return 0


def cmd_train_all(args: argparse.Namespace) -> int:
    dataset = _get_dataset_defaults()
    for fold in dataset["default_folds"]:
        fold_args = argparse.Namespace(**vars(args))
        fold_args.fold = fold
        cmd_train(fold_args)
    return 0


def cmd_find_best_config(args: argparse.Namespace) -> int:
    configure_environment()
    from brats_project.evaluation.find_best_configuration import (
        discover_trained_models_with_validation,
        dumb_trainer_config_plans_to_trained_models_dict,
        find_best_configuration,
    )

    dataset = _get_dataset_defaults()
    model_dict = dumb_trainer_config_plans_to_trained_models_dict(
        args.trainers,
        args.configurations,
        args.plans_identifiers,
    )
    discovered_models = discover_trained_models_with_validation(dataset["name"])
    requested_model_keys = {
        (m["trainer"], m["plans"], m["configuration"]) for m in model_dict
    }
    discovered_model_keys = {
        (m["trainer"], m["plans"], m["configuration"]) for m in discovered_models
    }
    if discovered_models and requested_model_keys.isdisjoint(discovered_model_keys):
        print(
            "[INFO] Requested default model combination does not have validation outputs yet. "
            "Falling back to discovered trained models with validation results."
        )
        model_dict = discovered_models
    find_best_configuration(
        dataset["name"],
        allowed_trained_models=model_dict,
        allow_ensembling=not args.disable_ensembling,
        num_processes=args.num_processes,
        overwrite=not args.no_overwrite,
        folds=tuple(args.folds),
    )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    configure_environment()
    from brats_project.inference.predict_from_raw_data import predict_entry_point

    dataset = _get_dataset_defaults()
    input_dir = resolve_workspace_path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_training_cases is not None and args.sample_training_cases < 1:
        raise RuntimeError("--sample-training-cases must be at least 1.")
    default_input_dir = resolve_workspace_path(
        load_project_config()["paths"]["prediction_input_root"]
    )
    should_auto_sample_training = (
        args.sample_training_cases is None
        and not args.disable_auto_sample_training
        and input_dir == default_input_dir
        and not _directory_has_nifti_files(input_dir)
    )
    if should_auto_sample_training:
        _prepare_sampled_prediction_input(
            dataset_name=dataset["name"],
            input_dir=input_dir,
            sample_count=args.auto_sample_training_cases,
            sample_seed=args.sample_seed,
        )
    elif args.sample_training_cases is not None:
        _prepare_sampled_prediction_input(
            dataset_name=dataset["name"],
            input_dir=input_dir,
            sample_count=args.sample_training_cases,
            sample_seed=args.sample_seed,
        )
    elif not _directory_has_nifti_files(input_dir):
        raise RuntimeError(
            f"Prediction input folder is empty: {input_dir}\n"
            "Put inference cases into that directory, or let the CLI auto-sample temporary validation cases from "
            "the training set by using the default input directory or by passing --sample-training-cases."
        )

    trainer, configuration, plans, folds = (
        _resolve_predict_defaults_from_inference_info(dataset["name"], args, dataset)
    )
    argv = [
        "-d",
        dataset["name"],
        "-i",
        str(input_dir),
        "-o",
        str(resolve_workspace_path(args.output_dir)),
        "-c",
        configuration,
        "-tr",
        trainer,
        "-p",
        plans,
        "-device",
        args.device,
        "-step_size",
        str(args.step_size),
        "-npp",
        str(args.npp),
        "-nps",
        str(args.nps),
    ]
    if folds:
        argv.extend(["-f", *folds])
    if args.disable_tta:
        argv.append("--disable_tta")
    if args.verbose:
        argv.append("--verbose")
    if args.save_probabilities:
        argv.append("--save_probabilities")
    if args.continue_prediction:
        argv.append("--continue_prediction")
    if args.disable_progress_bar:
        argv.append("--disable_progress_bar")
    if args.not_on_device:
        argv.append("--not_on_device")
    if args.prev_stage_predictions is not None:
        argv.extend(
            [
                "-prev_stage_predictions",
                str(resolve_workspace_path(args.prev_stage_predictions)),
            ]
        )

    _call_entrypoint(predict_entry_point, argv)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    configure_environment()
    from brats_project.evaluation.evaluate_predictions import compute_metrics_on_folder2

    pred_dir = resolve_workspace_path(args.pred_dir)
    dataset = _get_dataset_defaults()
    trainer, configuration, plans, _ = _resolve_predict_defaults_from_inference_info(
        dataset["name"],
        argparse.Namespace(
            trainer=args.trainer,
            configuration=args.configuration,
            plans=args.plans,
            folds=[str(fold) for fold in dataset["default_folds"]],
        ),
        dataset,
    )
    dataset_json_path, plans_json_path = _resolve_evaluation_metadata(
        pred_dir=pred_dir,
        dataset_json=args.dataset_json,
        plans_json=args.plans_json,
        trainer=trainer,
        plans=plans,
        configuration=configuration,
    )
    dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    file_ending = dataset_json["file_ending"]
    predicted_files = sorted(p.name for p in pred_dir.glob(f"*{file_ending}"))
    if len(predicted_files) == 0:
        raise RuntimeError(
            f"Prediction folder is empty: {pred_dir}\n"
            "Run `python BraTS/run.py predict` first, or pass --pred-dir to a folder that already contains "
            "predicted segmentations."
        )
    gt_dir = resolve_workspace_path(args.gt_dir)
    gt_files = sorted(p.name for p in gt_dir.glob(f"*{file_ending}"))
    gt_file_count = len(gt_files)
    chill = args.chill
    if not chill and predicted_files != gt_files:
        missing_predictions = sorted(set(gt_files) - set(predicted_files))
        extra_predictions = sorted(set(predicted_files) - set(gt_files))
        details = []
        if missing_predictions:
            details.append(f"missing predictions for {len(missing_predictions)} cases")
        if extra_predictions:
            details.append(
                f"{len(extra_predictions)} predictions have no matching ground truth"
            )
        detail_text = (
            "; ".join(details)
            if details
            else "prediction filenames do not match the ground-truth set"
        )
        raise RuntimeError(
            "Prediction coverage does not match the ground-truth directory.\n"
            f"predictions: {len(predicted_files)} files, ground truth: {gt_file_count} files\n"
            f"details: {detail_text}\n"
            "If you intentionally want to evaluate only a subset, rerun with --chill."
        )
    if chill and predicted_files != gt_files:
        print(
            f"[INFO] Evaluating a prediction subset with --chill: "
            f"{len(predicted_files)} prediction files vs {gt_file_count} ground-truth files."
        )
    compute_metrics_on_folder2(
        str(gt_dir),
        str(pred_dir),
        str(dataset_json_path),
        str(plans_json_path),
        output_file=(
            str(resolve_workspace_path(args.output_file))
            if args.output_file
            else str(_default_evaluation_output_file())
        ),
        num_processes=args.num_processes,
        chill=chill,
    )
    return 0


def cmd_report_evaluation(args: argparse.Namespace) -> int:
    configure_environment()
    dataset = _get_dataset_defaults()
    env = configure_environment()
    try:
        module = _load_external_module(
            "04_inference_and_evaluation/generate_evaluation_report.py",
            "generate_evaluation_report",
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "report-evaluation requires the visualization dependencies used by the report generator "
            f"(missing module: {exc.name}). Run it inside the pytorch environment where matplotlib/nibabel are installed."
        ) from exc
    summary_file = (
        resolve_workspace_path(args.summary_file)
        if args.summary_file
        else _default_evaluation_output_file()
    )
    sample_selection_file = (
        resolve_workspace_path(args.sample_selection_file)
        if args.sample_selection_file
        else _project_file("04_inference_and_evaluation/input/sample_selection.json")
    )
    module.main(
        summary_file=summary_file,
        predictions_dir=resolve_workspace_path(args.predictions_dir),
        raw_dataset_dir=Path(env["PROJECT_RAW"]) / dataset["name"],
        output_dir=resolve_workspace_path(args.output_dir),
        sample_selection_file=(
            sample_selection_file if sample_selection_file.is_file() else None
        ),
    )
    print(
        f"[OK] Evaluation report written to: "
        f"{workspace_relative_string(resolve_workspace_path(args.output_dir))}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    config = load_project_config()
    dataset = config["dataset"]
    paths = config["paths"]
    runtime = config["runtime"]

    parser = argparse.ArgumentParser(
        description="Standalone BraTS2020 project entrypoint"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check environment and key paths")
    doctor.set_defaults(func=cmd_doctor)

    visualize = subparsers.add_parser(
        "visualize-first-case", help="Regenerate the first-case visualization snapshot"
    )
    visualize.add_argument(
        "--data-root",
        default=paths["archive_root"],
        help="Relative to BraTS project root by default",
    )
    visualize.add_argument(
        "--output-dir",
        default=paths["first_case_output_root"],
        help="Relative to BraTS project root by default",
    )
    visualize.set_defaults(func=cmd_visualize_first_case)

    prepare = subparsers.add_parser(
        "prepare-dataset", help="Convert BraTS2020 raw data into Dataset220_BraTS2020"
    )
    prepare.add_argument("--src-root", default=paths["archive_root"])
    prepare.add_argument("--project-raw", default=paths["project_raw_root"])
    prepare.add_argument("--plans", default=dataset["plans"])
    prepare.add_argument(
        "--force",
        action="store_true",
        help="Rebuild Dataset220_BraTS2020 even if it already exists.",
    )
    prepare.set_defaults(func=cmd_prepare_dataset)

    plan = subparsers.add_parser(
        "plan-preprocess",
        help="Run fingerprint extraction, planning, and preprocessing",
    )
    plan.add_argument("--dataset-id", type=int, default=dataset["id"])
    plan.add_argument("--plans", default=dataset["plans"])
    plan.add_argument(
        "--configurations",
        nargs="+",
        default=["2d", "3d_fullres", "3d_lowres"],
    )
    plan.add_argument(
        "--preprocess-processes",
        nargs="+",
        type=int,
        default=[runtime["default_num_processes_preprocessing"]],
    )
    plan.add_argument(
        "--num-processes", type=int, default=runtime["default_num_processes"]
    )
    plan.add_argument("--gpu-memory-target", type=float, default=None)
    plan.add_argument("--verify-dataset", action="store_true")
    plan.add_argument("--clean", action="store_true")
    plan.add_argument(
        "--recompute-fingerprint",
        action="store_true",
        help="Force regeneration of dataset_fingerprint.json.",
    )
    plan.add_argument(
        "--recompute-plans",
        action="store_true",
        help="Force regeneration of the plans JSON even if it already exists.",
    )
    plan.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Force regeneration of preprocessed outputs for the requested configurations.",
    )
    plan.add_argument("--quiet", action="store_true")
    plan.add_argument("--disable-progress-bar", action="store_true")
    plan.set_defaults(func=cmd_plan_preprocess)

    train = subparsers.add_parser("train", help="Train one fold or validate one fold")
    train.add_argument("--fold", default=0)
    train.add_argument("--configuration", default=dataset["default_configuration"])
    train.add_argument("--trainer", default=dataset["trainer"])
    train.add_argument("--plans", default=dataset["plans"])
    train.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    train.add_argument("--num-gpus", type=int, default=1)
    train.add_argument("--npz", action="store_true")
    train.add_argument("--continue-training", action="store_true")
    train.add_argument(
        "--restart-training",
        action="store_true",
        help="Ignore existing checkpoints and start this fold from scratch.",
    )
    train.add_argument("--validation-only", action="store_true")
    train.add_argument("--val-best", action="store_true")
    train.add_argument("--disable-checkpointing", action="store_true")
    train.add_argument("--pretrained-weights", default=None)
    train.set_defaults(func=cmd_train)

    train_all = subparsers.add_parser(
        "train-all", help="Run the standard 5-fold cross validation"
    )
    for action in train._actions[1:]:
        if action.dest == "fold":
            continue
        train_all._add_action(action)
    train_all.set_defaults(func=cmd_train_all)

    best = subparsers.add_parser(
        "find-best-config", help="Aggregate folds and determine the best configuration"
    )
    best.add_argument(
        "--configurations",
        nargs="+",
        default=[dataset["default_configuration"]],
        help="Configurations to compare. Defaults to the standalone project's default training configuration.",
    )
    best.add_argument(
        "--trainers",
        nargs="+",
        default=[dataset["trainer"]],
        help="Trainer class names to compare.",
    )
    best.add_argument(
        "--plans-identifiers",
        nargs="+",
        default=[dataset["plans"]],
        help="Plans identifiers to compare.",
    )
    best.add_argument("--folds", nargs="+", type=int, default=dataset["default_folds"])
    best.add_argument(
        "--num-processes", type=int, default=runtime["default_num_processes"]
    )
    best.add_argument("--disable-ensembling", action="store_true")
    best.add_argument("--no-overwrite", action="store_true")
    best.set_defaults(func=cmd_find_best_config)

    predict = subparsers.add_parser("predict", help="Run model inference")
    predict.add_argument("--input-dir", default=paths["prediction_input_root"])
    predict.add_argument("--output-dir", default=paths["prediction_output_root"])
    predict.add_argument("--configuration", default=dataset["default_configuration"])
    predict.add_argument("--trainer", default=dataset["trainer"])
    predict.add_argument("--plans", default=dataset["plans"])
    predict.add_argument(
        "--folds", nargs="+", default=[str(fold) for fold in dataset["default_folds"]]
    )
    predict.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    predict.add_argument("--step-size", type=float, default=0.5)
    predict.add_argument(
        "--npp", type=int, default=runtime["default_num_processes_preprocessing"]
    )
    predict.add_argument(
        "--nps", type=int, default=runtime["default_num_processes_segmentation_export"]
    )
    predict.add_argument("--disable-tta", action="store_true")
    predict.add_argument("--verbose", action="store_true")
    predict.add_argument("--save-probabilities", action="store_true")
    predict.add_argument("--continue-prediction", action="store_true")
    predict.add_argument("--disable-progress-bar", action="store_true")
    predict.add_argument("--not-on-device", action="store_true")
    predict.add_argument("--prev-stage-predictions", default=None)
    predict.add_argument(
        "--sample-training-cases",
        type=int,
        default=None,
        help="Before inference, randomly copy this many cases from PROJECT_RAW/<dataset>/imagesTr into the input directory. The input directory must be empty.",
    )
    predict.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when sampling temporary validation cases from the training set.",
    )
    predict.add_argument(
        "--disable-auto-sample-training",
        action="store_true",
        help="Disable the default behavior that auto-populates the empty project input directory with sampled training cases.",
    )
    predict.add_argument(
        "--auto-sample-training-cases",
        type=int,
        default=8,
        help="How many training cases to sample when the default project input directory is empty and auto-sampling is enabled.",
    )
    predict.set_defaults(func=cmd_predict)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a prediction folder against ground truth"
    )
    evaluate.add_argument(
        "--gt-dir",
        default="nnUNet_test/nnUNet_preprocessed/Dataset220_BraTS2020/gt_segmentations",
    )
    evaluate.add_argument("--pred-dir", default=paths["prediction_output_root"])
    evaluate.add_argument(
        "--dataset-json",
        default=None,
        help="Defaults to <pred-dir>/dataset.json and falls back to the trained model metadata if missing.",
    )
    evaluate.add_argument(
        "--plans-json",
        default=None,
        help="Defaults to <pred-dir>/plans.json and falls back to the trained model metadata if missing.",
    )
    evaluate.add_argument("--configuration", default=dataset["default_configuration"])
    evaluate.add_argument("--trainer", default=dataset["trainer"])
    evaluate.add_argument("--plans", default=dataset["plans"])
    evaluate.add_argument(
        "--output-file",
        default=None,
        help="Defaults to BraTS/04_inference_and_evaluation/evaluation/summary.json.",
    )
    evaluate.add_argument(
        "--num-processes", type=int, default=runtime["default_num_processes"]
    )
    evaluate.add_argument("--chill", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    report = subparsers.add_parser(
        "report-evaluation",
        help="Generate a markdown evaluation report with summary figures and case overlays",
    )
    report.add_argument(
        "--summary-file",
        default=None,
        help="Defaults to BraTS/04_inference_and_evaluation/evaluation/summary.json.",
    )
    report.add_argument(
        "--predictions-dir",
        default=paths["prediction_output_root"],
    )
    report.add_argument(
        "--sample-selection-file",
        default="BraTS/04_inference_and_evaluation/input/sample_selection.json",
    )
    report.add_argument(
        "--output-dir",
        default="BraTS/04_inference_and_evaluation/report",
    )
    report.set_defaults(func=cmd_report_evaluation)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args) or 0)
