from __future__ import annotations

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "project_config.json").is_file():
            return parent
    raise RuntimeError(
        "Unable to locate project_config.json from brats_project package"
    )


@lru_cache(maxsize=1)
def get_workspace_root() -> Path:
    return get_project_root()


@lru_cache(maxsize=1)
def get_dataset_root() -> Path:
    config = load_project_config()
    dataset_root = config["paths"].get("dataset_root")
    if dataset_root is None:
        return (get_project_root().parent / "BraTS-Dataset").resolve()
    return resolve_project_path(dataset_root)


@lru_cache(maxsize=1)
def load_project_config() -> dict[str, Any]:
    config_path = get_project_root() / "project_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_project_path(relative_or_absolute: str | os.PathLike[str]) -> Path:
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate.resolve()
    return (get_project_root() / candidate).resolve()


def project_relative_string(path: str | os.PathLike[str]) -> str:
    absolute = Path(path).resolve()
    workspace_root = get_project_root()
    return os.path.relpath(absolute, workspace_root)


def resolve_workspace_path(relative_or_absolute: str | os.PathLike[str]) -> Path:
    return resolve_project_path(relative_or_absolute)


def workspace_relative_string(path: str | os.PathLike[str]) -> str:
    return project_relative_string(path)


def get_workspace_nifti_mirror_root() -> Path:
    config = load_project_config()
    mirror_root = config["paths"].get("workspace_nifti_mirror_root")
    if mirror_root is None:
        return get_dataset_root() / "workspace_nifti"
    return resolve_project_path(mirror_root)


def is_nifti_path(path: str | os.PathLike[str]) -> bool:
    name = Path(path).name
    return name.endswith(".nii") or name.endswith(".nii.gz")


def get_workspace_nifti_mirror_path(path: str | os.PathLike[str]) -> Path:
    source = Path(path).resolve()
    relative = source.relative_to(get_project_root())
    return (get_workspace_nifti_mirror_root() / relative).resolve()


def relocate_workspace_nifti_files(
    search_roots: Iterable[str | os.PathLike[str]],
) -> list[tuple[Path, Path]]:
    moved: list[tuple[Path, Path]] = []
    project_root = get_project_root()
    dataset_root = get_dataset_root()

    for root in search_roots:
        root_path = Path(root).resolve()
        if not root_path.exists():
            continue

        candidates = (
            [root_path]
            if root_path.is_file()
            else sorted(root_path.rglob("*"), key=lambda path: (len(path.parts), str(path)))
        )
        for path in candidates:
            if not path.is_file() or path.is_symlink() or not is_nifti_path(path):
                continue
            try:
                path.relative_to(project_root)
            except ValueError:
                continue
            try:
                path.relative_to(dataset_root)
            except ValueError:
                pass
            else:
                continue

            mirrored_path = get_workspace_nifti_mirror_path(path)
            mirrored_path.parent.mkdir(parents=True, exist_ok=True)
            if not mirrored_path.exists():
                shutil.move(str(path), str(mirrored_path))
            else:
                path.unlink()
            path.symlink_to(mirrored_path)
            moved.append((path, mirrored_path))

    return moved


def get_default_environment_paths() -> dict[str, str]:
    config = load_project_config()
    paths = config["paths"]
    return {
        "PROJECT_RAW": str(resolve_project_path(paths["project_raw_root"])),
        "PROJECT_PREPROCESSED": str(
            resolve_project_path(paths["project_preprocessed_root"])
        ),
        "PROJECT_RESULTS": str(resolve_project_path(paths["project_results_root"])),
    }


def configure_environment() -> dict[str, str]:
    config = load_project_config()
    runtime = config["runtime"]
    defaults = get_default_environment_paths()

    os.environ.setdefault("PROJECT_RAW", defaults["PROJECT_RAW"])
    os.environ.setdefault("PROJECT_PREPROCESSED", defaults["PROJECT_PREPROCESSED"])
    os.environ.setdefault("PROJECT_RESULTS", defaults["PROJECT_RESULTS"])
    os.environ.setdefault("BRATS_DEF_N_PROC", str(runtime["default_num_processes"]))
    os.environ.setdefault("BRATS_N_PROC_DA", str(runtime["default_n_proc_da"]))
    os.environ.setdefault(
        "BRATS_NPP", str(runtime["default_num_processes_preprocessing"])
    )
    os.environ.setdefault(
        "BRATS_NPS", str(runtime["default_num_processes_segmentation_export"])
    )
    return {
        "PROJECT_RAW": os.environ["PROJECT_RAW"],
        "PROJECT_PREPROCESSED": os.environ["PROJECT_PREPROCESSED"],
        "PROJECT_RESULTS": os.environ["PROJECT_RESULTS"],
    }
