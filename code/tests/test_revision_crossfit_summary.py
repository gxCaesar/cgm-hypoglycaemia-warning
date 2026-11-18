import copy
import json

import numpy as np
import pytest
from test_revision_crossfit_evaluation import _inputs

import glycoshift.revision_crossfit_summary as summary_module
from glycoshift.revision_baselines import run_revision_baseline
from glycoshift.revision_crossfit_evaluation import run_crossfit_evaluation
from glycoshift.revision_crossfit_summary import (
    BURDEN,
    METHODS,
    SEEDS,
    SENSITIVITY,
    collect_oof,
    summarize_paired_points,
)
from glycoshift.revision_runner import revision_config, run_revision_deep


def test_seed_and_horizon_contrast_oracle_propagates_undefined():
    points = {}
    for arm in METHODS:
        seeds = SEEDS if arm in {"lstm", "tcn"} else (17,)
        for position, seed in enumerate(seeds):
            for cut in (54, 70):
                for horizon in (30, 60):
                    points[arm, seed, cut, horizon] = {
                        SENSITIVITY: (
                            0.4 + position * 0.01 + (horizon == 60) * 0.1
                            if arm == "lstm" else 0.3
                        ) + (0.2 if cut == 70 and arm == "lstm" else 0),
                        BURDEN: 0.0,
                    }
    result = summarize_paired_points(points, metric=SENSITIVITY)
    assert result["mean_contrast_change_70_minus_54"] == pytest.approx(0.2)
    assert len(result["model_seed_contrasts"]) == 5
    assert result["generalization_ci"] is None
    points["lstm", 17, 70, 30][SENSITIVITY] = np.nan
    assert np.isnan(summarize_paired_points(points, metric=SENSITIVITY)[
        "mean_contrast_change_70_minus_54"])


def test_collect_five_fold_completed_outputs_and_reject_duplicate_fold(tmp_path, monkeypatch):
    prepared, config = _inputs(tmp_path, monkeypatch)
    # Reduce only test dimensions: exercise real five-fold model fitting and
    # frozen outer inference for one LSTM seed and one deterministic control.
    monkeypatch.setattr(summary_module, "METHODS", ("lstm", "persistence"))
    monkeypatch.setattr(summary_module, "SEEDS", (17,))
    config["training"].update(maximum_epochs=1, batch_size=64, effective_batch_size=64,
                              num_workers=0)
    config["evaluation"].update(event_ap_candidate_count=5,
                                selection_operational_candidate_count=5)
    arms = []
    for arm in ("lstm", "persistence"):
        outputs = []
        for fold in range(5):
            run = tmp_path / f"{arm}-{fold}"
            effective = revision_config(config, seed=17, output=run)
            if arm == "lstm":
                run_revision_deep(effective, prepared, architecture=arm, outer_fold=fold)
            else:
                run_revision_baseline(effective, prepared, baseline=arm, outer_fold=fold)
            output = tmp_path / f"outer-{arm}-{fold}"
            run_crossfit_evaluation(prepared, run, outer_fold=fold, output=output)
            outputs.append(str(output))
        arms.append({"arm": arm, "model_seed": 17, "fold_outputs": outputs})
    request = {"prepared_dir": str(prepared), "output_dir": str(tmp_path / "summary"),
               "arms": arms}
    output = collect_oof(request)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["producer_count"] == 10
    assert receipt["condition_count"] == 8
    full = json.loads((output / "all_development.json").read_text())
    subset = json.loads((output / "original_selection.json").read_text())
    assert {row["participant_pooled"]["participant_count"] for row in full["points"]} == {9}
    assert {row["participant_pooled"]["participant_count"] for row in subset["points"]} == {3}
    bad = copy.deepcopy(request)
    bad["output_dir"] = str(tmp_path / "bad")
    bad["arms"][0]["fold_outputs"][1] = bad["arms"][0]["fold_outputs"][0]
    with pytest.raises(ValueError, match="duplicated"):
        collect_oof(bad)
