from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


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
