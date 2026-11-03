from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a prespecified CGM model")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model",
        choices=(
            "baseline",
            "constant_global_train_prevalence",
            "constant_source_conditional_train_prevalence",
            "label_shuffle_logistic",
            "persistence",
            "linear",
            "xgboost",
            "glycoshift_tiny",
            "glycoshift_base",
            "lstm",
            "tcn",
            "patchtst",
            "patchmlp_forecast_mse",
            "patchmlp_forecast_glimmer",
            "asb_erm_lstm",
            "asb_hardmax_zero_lstm",
            "mero_cf_lstm",
            "c1_cf_mero_lstm",
            "c2_source_event_regret_lstm",
        ),
        default="baseline",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model in {
        "baseline",
        "constant_global_train_prevalence",
        "constant_source_conditional_train_prevalence",
        "label_shuffle_logistic",
        "persistence",
        "linear",
        "xgboost",
    }:
        baseline_name = (
            str(config["model"]["baseline"])
            if args.model == "baseline"
            else args.model
        )
        run_dir = run_training(config, baseline_name=baseline_name)
    elif args.model.startswith("patchmlp_forecast_"):
        from .forecast_training import run_forecast_training

        run_dir = run_forecast_training(
            config,
            loss_name=args.model.removeprefix("patchmlp_forecast_"),
        )
    elif args.model in {
        "asb_erm_lstm",
        "asb_hardmax_zero_lstm",
        "mero_cf_lstm",
        "c1_cf_mero_lstm",
        "c2_source_event_regret_lstm",
    }:
        from .mero_training import run_mero_training

        objective_mode = {
            "asb_erm_lstm": "all_source_erm",
            "asb_hardmax_zero_lstm": "hardmax_zero",
            "mero_cf_lstm": "mero",
            "c1_cf_mero_lstm": "c1_cf_mero",
            "c2_source_event_regret_lstm": "c2_source_event_regret",
        }[args.model]
        run_dir = run_mero_training(
            config,
            objective_mode=objective_mode,
        )
    else:
        from .deep_training import run_deep_training

        if args.model.startswith("glycoshift_"):
            config["model"]["variant"] = args.model.removeprefix("glycoshift_")
            architecture = "glycoshift"
        else:
            architecture = args.model
        run_dir = run_deep_training(config, architecture=architecture)
    print(f"completed {args.model} run: {run_dir}")


if __name__ == "__main__":
    main()
