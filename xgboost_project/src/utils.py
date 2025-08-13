"""Utility functions and path helpers for the XGBoost project.

Example
-------
Print project paths::

    python src/utils.py --show-paths

Save demo metrics::

    python src/utils.py --demo-metrics reports/demo_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"


def ensure_directories() -> None:
    """Ensure that standard project directories exist."""
    for path in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_json_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    """Save metrics dictionary to JSON file.

    Parameters
    ----------
    path: Path
        Destination file path.
    metrics: Dict[str, Any]
        Metrics to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse command line arguments.

    This helper exists so that other modules can import and extend the parser
    if desired.
    """
    return parser.parse_args()


def main() -> None:
    parser = argparse.ArgumentParser(description="Utility helpers")
    parser.add_argument("--show-paths", action="store_true", help="Display project directories")
    parser.add_argument("--demo-metrics", type=Path, help="Path to save demo metrics JSON")
    args = parse_args(parser)

    if args.show_paths:
        for name, path in {
            "PROJECT_ROOT": PROJECT_ROOT,
            "DATA_DIR": DATA_DIR,
            "MODELS_DIR": MODELS_DIR,
            "REPORTS_DIR": REPORTS_DIR,
        }.items():
            print(f"{name}: {path}")

    if args.demo_metrics:
        save_json_metrics(args.demo_metrics, {"accuracy": 0.0})
        print(f"Demo metrics saved to {args.demo_metrics}")


if __name__ == "__main__":
    main()
