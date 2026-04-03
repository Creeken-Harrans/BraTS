from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "03_training_and_results" / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brats_project.cli import main


if __name__ == "__main__":
    argv = None if len(sys.argv) > 1 else ["--help"]
    raise SystemExit(main(argv))
