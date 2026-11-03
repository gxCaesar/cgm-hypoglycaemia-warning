from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path
from .deep_training import run_deep_training


def seeded_configs(
    config: dict[str, Any],
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    seeds = [int(value) for value in config["training"]["final_seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("training.final_seeds contains duplicates")
    base_run_dir = resolve_path(config, config["output"]["run_dir"])
    if base_run_dir is None:
        raise ValueError("output.run_dir is required")
    output: list[dict[str, Any]] = []
    for seed in seeds:
        run_config = deepcopy(config)
        run_config["training"]["seed"] = seed
        if not bool(run_config["project"].get("synthetic", False)):
            run_config["data"].setdefault(
                "training_window_shards",
                str(Path(base_run_dir) / "window_shards"),
            )
            run_config["data"].setdefault(
                "training_cohort_audit",
                str(Path(base_run_dir) / "cohort_audit.json"),
            )
        run_config["output"]["run_dir"] = str(
            Path(base_run_dir) / f"{model_name}-seed-{seed:03d}"
        )
        output.append(run_config)
    return output


def select_seeded_configs(
    config: dict[str, Any],
    *,
    model_name: str,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    configs = seeded_configs(config, model_name=model_name)
    if seeds is None:
        return configs
    if len(seeds) != len(set(seeds)):
        raise ValueError("requested seeds contain duplicates")
    by_seed = {
        int(item["training"]["seed"]): item
        for item in configs
    }
    unknown = sorted(set(seeds).difference(by_seed))
    if unknown:
        raise ValueError(
            "requested seeds are not in training.final_seeds: "
            f"{unknown}"
        )
    return [by_seed[seed] for seed in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen deep-model seed suite"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model",
        choices=(
            "glycoshift_tiny",
            "glycoshift_base",
            "lstm",
            "tcn",
            "patchtst",
        ),
        default="glycoshift_base",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help=(
            "run only this ordered subset of the frozen "
            "training.final_seeds"
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    architecture = (
        "glycoshift"
        if args.model.startswith("glycoshift_")
        else args.model
    )
    if architecture == "glycoshift":
        variant = args.model.removeprefix("glycoshift_")
    else:
        variant = str(config["model"]["variant"])
    for run_config in select_seeded_configs(
        config,
        model_name=args.model,
        seeds=args.seeds,
    ):
        run_config["model"]["variant"] = variant
        run_dir = run_deep_training(
            run_config,
            architecture=architecture,
        )
        print(f"completed seed {run_config['training']['seed']}: {run_dir}")


if __name__ == "__main__":
    main()
