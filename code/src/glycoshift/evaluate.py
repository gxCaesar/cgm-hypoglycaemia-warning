from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .config import load_config, validate_config
from .io import write_json
from .pipeline import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute CGM run metrics")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument(
        "--partition",
        choices=("external_loop", "stress_hupa_ucm"),
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.partition is not None:
        if args.config is None:
            raise ValueError("--config is required for a frozen partition evaluation")
        config = load_config(args.config)
        if (run_dir / "ckpts" / "best.pt").exists():
            from .deep_training import run_frozen_evaluation

            metrics_path = run_frozen_evaluation(
                config,
                run_dir=run_dir,
                partition=args.partition,
            )
        elif (run_dir / "baseline_models.joblib").exists():
            from .baseline_evaluation import run_frozen_baseline_evaluation

            metrics_path = run_frozen_baseline_evaluation(
                config,
                run_dir=run_dir,
                partition=args.partition,
            )
        else:
            raise FileNotFoundError("run has no frozen deep or baseline model")
        print(f"completed frozen evaluation: {metrics_path}")
        return
    required = (
        "config_resolved.yaml",
        "model_summary.json",
        "predictions.parquet",
    )
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"run is missing artifacts: {missing}")
    with (run_dir / "config_resolved.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    with (run_dir / "model_summary.json").open("r", encoding="utf-8") as handle:
        model_summary = json.load(handle)
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    metrics = evaluate_predictions(
        predictions,
        config=config,
        model_summary=model_summary,
    )
    write_json(metrics, run_dir / "metrics.json")
    print(f"validated metrics: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
