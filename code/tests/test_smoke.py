import json
from pathlib import Path

import pandas as pd
import pytest

from glycoshift.baseline_evaluation import run_frozen_baseline_evaluation
from glycoshift.config import load_config
from glycoshift.pipeline import run_training


@pytest.mark.parametrize(
    ("baseline", "audit_artifact"),
    (
        ("constant_global_train_prevalence", "constant_denominators.json"),
        (
            "constant_source_conditional_train_prevalence",
            "constant_denominators.json",
        ),
        ("label_shuffle_logistic", "label_shuffle_ledger.json"),
    ),
)
def test_sanity_controls_use_the_operational_evaluator(
    tmp_path: Path,
    baseline: str,
    audit_artifact: str,
) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["data"]["participants_per_source"] = 11
    config["data"]["days_per_participant"] = 1
    config["output"]["run_dir"] = str(tmp_path / baseline)

    run_dir = run_training(config, baseline_name=baseline)

    assert (run_dir / audit_artifact).exists()
    operational = json.loads((run_dir / "operational_metrics.json").read_text())
    assert operational["threshold_partition"] == "calibration"
    assert operational["evaluation_partition"] == "selection"
    if baseline == "label_shuffle_logistic":
        ledger = json.loads((run_dir / audit_artifact).read_text())
        assert all(
            task["fit_positive_windows_before"]
            == task["fit_positive_windows_after"]
            for task in ledger["tasks"].values()
        )


def test_end_to_end_smoke(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["data"]["participants_per_source"] = 11
    config["data"]["days_per_participant"] = 1
    config["output"]["run_dir"] = str(tmp_path / "run")
    run_dir = run_training(config)
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "cohort_audit.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "operational_metrics.json").exists()
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    assert {
        "selection",
        "calibration",
        "external_loop",
        "stress_hupa_ucm",
    } == set(predictions["partition"])
    for probability_column in predictions.filter(like="_probability"):
        if "_raw_probability" in probability_column:
            continue
        task = probability_column.removesuffix("_probability")
        label_column = f"{task}_label"
        eligible = predictions[label_column].notna()
        if task.startswith("severe_hypo_"):
            eligible &= predictions["partition"].isin(
                {"selection", "calibration"}
            )
        assert predictions.loc[eligible, probability_column].notna().all()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "severe_hypo_30" in metrics["partitions"]["selection"]
    assert "operational_v1_3" in metrics
    frozen_metrics = run_frozen_baseline_evaluation(
        config,
        run_dir=run_dir,
        partition="stress_hupa_ucm",
    )
    assert frozen_metrics.exists()
    with pytest.raises(FileExistsError, match="refusing to repeat"):
        run_frozen_baseline_evaluation(
            config,
            run_dir=run_dir,
            partition="stress_hupa_ucm",
        )
