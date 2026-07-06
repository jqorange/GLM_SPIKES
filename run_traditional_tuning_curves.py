from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from traditional_tuning_curves.pipeline import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()
