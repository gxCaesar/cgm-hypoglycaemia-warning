import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_fixed_frame_evaluation import _evaluate, _events, _timeline
from test_revision_crossfit_evaluation import _inputs

import glycoshift.revision_holdout_summary as holdout_module
from glycoshift.revision_baselines import run_revision_baseline
from glycoshift.revision_crossfit_evaluation import (
    _binding,
    aggregate_oof_participants,
    run_crossfit_evaluation,
)
from glycoshift.revision_crossfit_summary import BURDEN, SENSITIVITY, collect_oof
from glycoshift.revision_holdout_summary import (
    _check_prediction_roles,
    collect_holdout,
    summarize_holdout_points,
)
from glycoshift.revision_runner import revision_config, run_revision_deep


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixed-holdout")
    with pytest.MonkeyPatch.context() as monkeypatch:
        prepared, config = _inputs(root, monkeypatch)
        config["training"].update(
            maximum_epochs=1, batch_size=64, effective_batch_size=64, num_workers=0
        )
        config["evaluation"].update(
            event_ap_candidate_count=5, selection_operational_candidate_count=5
        )
        arms = []
        for arm in ("lstm", "persistence"):
            run = root / f"model-{arm}"
            effective = revision_config(config, seed=17, output=run)
            if arm == "lstm":
                run_revision_deep(effective, prepared, architecture=arm, outer_fold=0)
            else:
                run_revision_baseline(effective, prepared, baseline=arm, outer_fold=0)
            output = root / f"outer-{arm}"
            run_crossfit_evaluation(prepared, run, outer_fold=0, output=output)
            arms.append({"arm": arm, "model_seed": 17, "fold_outputs": [str(output)]})
    return {"prepared_dir": str(prepared), "arms": arms}


def _request(completed, tmp_path):
    """Copy only small synthetic evaluator outputs; fitted inputs remain bound."""
    request = copy.deepcopy(completed)
    request["output_dir"] = str(tmp_path / "summary")
    for arm in request["arms"]:
        old = Path(arm["fold_outputs"][0])
        directory = tmp_path / arm["arm"]
        shutil.copytree(old, directory)
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["policy_freeze"] = _binding(directory / "private/policy-freeze.json")
        receipt["outputs"] = [
            _binding(directory / Path(row["path"]).relative_to(old))
            for row in receipt["outputs"]
        ]
        receipt_path.write_text(json.dumps(receipt))
        arm["fold_outputs"] = [str(directory)]
    return request


def _change_receipt(request, change):
    directory = Path(request["arms"][0]["fold_outputs"][0])
    path = directory / "receipt.json"
    receipt = json.loads(path.read_text())
    change(receipt, directory)
    path.write_text(json.dumps(receipt))


def test_participant_source_horizon_contrast_oracle_preserves_undefined():
    # Source A has three participants; B has one. The equal-source sensitivity
    # therefore differs from participant pooling (2/3 vs 1/2 below).
    identities = [("A", "a"), ("A", "b"), ("A", "c"), ("B", "d")]
    events = _events([(s, p, "event", 10) for s, p in identities])
    points = {}
    for arm in ("lstm", "persistence"):
        for cut in (54, 70):
            for horizon in (30, 60):
                detected = {"a", "d"} if arm == "lstm" else {"a"}
                if arm == "lstm" and cut == 70:
                    detected = {"a", "b", "c", "d"} if horizon == 60 else {"a", "b"}
                timeline = _timeline([
                    (s, p, 0, 100, float(p in detected), True) for s, p in identities
                ])
                table = _evaluate(
                    timeline, events, cut=cut, horizon=horizon
                ).participants
                points[arm, cut, horizon] = aggregate_oof_participants(
                    table, ("A", "B")
                )
    assert points["lstm", 54, 30][SENSITIVITY] == pytest.approx(2 / 3)
    assert points["lstm", 54, 30]["participant_pooled"]["fixed_sensitivity"] == 0.5
    result = summarize_holdout_points(points, metric=SENSITIVITY)
    assert [row["contrast_change_70_minus_54"] for row in result[
        "horizon_contrasts"
    ]] == pytest.approx([-1 / 3, 1 / 3])
    assert result["equal_horizon_contrast"][
        "contrast_change_70_minus_54"
    ] == pytest.approx(0)
    assert result["source_horizon_contrasts"][0][
        "contrast_change_70_minus_54"
    ] == pytest.approx(1 / 3)
    assert result["generalization_ci"] is None
    # All-source arithmetic must propagate an unsupported source, never nanmean.
    for point in points.values():
        point["source_order"] = ("A", "B", "C")
        point["sources"].append({
            "source_id": "C", "fixed_sensitivity": np.nan,
            "participant_macro_unmatched_alarms_per_patient_day": np.nan,
        })
        point[SENSITIVITY] = np.nan
        point[BURDEN] = np.nan
    for metric in (SENSITIVITY, BURDEN):
        result = summarize_holdout_points(points, metric=metric)
        assert np.isnan(result["equal_horizon_contrast"]["contrast_change_70_minus_54"])
        assert np.isnan(result["source_horizon_contrasts"][-1][
            "contrast_change_70_minus_54"
        ])


def test_collect_completed_holdout_and_cli(completed, tmp_path, monkeypatch):
    request = _request(completed, tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr("sys.argv", ["summary", "--config", str(path)])
    holdout_module.main()
    output = tmp_path / "summary"
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["producer_count"] == 2 and receipt["condition_count"] == 8
    assert receipt["outer_fold"] == 0 and receipt["model_seed"] == 17
    assert not receipt["overall_generalization_ci_computed"]
    result = json.loads((output / "holdout.json").read_text())
    roles = json.loads((Path(completed["prepared_dir"]) /
                       "folds/fold-0/participant-roles.private.json").read_text())
    expected = sum(row["role"] == "outer_evaluation" for row in roles["participants"])
    assert result["outer_participant_count"] == expected
    assert {row["participant_pooled"]["participant_count"]
            for row in result["points"]} == {expected}
    assert result["source_order"] == ["s"]
    assert result["sensitivity"]["generalization_ci"] is None
    with pytest.raises(FileExistsError, match="overwrite"):
        collect_holdout(request)
    # The dedicated small collector must not weaken the existing OOF collector.
    request["output_dir"] = str(tmp_path / "full-oof")
    with pytest.raises(ValueError, match="all six methods"):
        collect_oof(request)


@pytest.mark.parametrize(
    "defect", ["missing_arm", "wrong_arm", "wrong_seed", "two_folds"]
)
def test_reject_wrong_declared_scope(completed, tmp_path, defect):
    request = _request(completed, tmp_path)
    if defect == "missing_arm":
        request["arms"].pop()
    elif defect == "wrong_arm":
        request["arms"][0]["arm"] = "tcn"
    elif defect == "wrong_seed":
        request["arms"][0]["model_seed"] = 23
    else:
        request["arms"][0]["fold_outputs"] *= 2
    with pytest.raises(ValueError, match="only LSTM/persistence seed17 fold0"):
        collect_holdout(request)
    assert not Path(request["output_dir"]).exists()


@pytest.mark.parametrize("field,value", [
    ("outer_fold", 1), ("outer_fold", False), ("status", "in_progress"),
    ("outer_refitting", True), ("prepared_dir", "/another/pool"),
])
def test_reject_incomplete_wrong_fold_or_pool_receipt(
    completed, tmp_path, field, value
):
    request = _request(completed, tmp_path)
    _change_receipt(request, lambda receipt, _: receipt.update({field: value}))
    with pytest.raises(ValueError, match="incomplete or has wrong fold/pool"):
        collect_holdout(request)


@pytest.mark.parametrize("defect", ["wrong_fold", "opened_outer", "wrong_fit_role"])
def test_reject_policy_isolation_defects(completed, tmp_path, defect):
    request = _request(completed, tmp_path)

    def change(receipt, directory):
        path = directory / "private/policy-freeze.json"
        freeze = json.loads(path.read_text())
        if defect == "wrong_fold":
            freeze["outer_fold"] = 1
        elif defect == "opened_outer":
            freeze["outer_values_opened"] = True
        else:
            freeze["policy_fit_role"] = "outer_evaluation"
        path.write_text(json.dumps(freeze))
        receipt["policy_freeze"] = _binding(path)

    _change_receipt(request, change)
    with pytest.raises(ValueError, match="freeze differs from run/fold/isolation"):
        collect_holdout(request)


@pytest.mark.parametrize("defect", ["missing_participant", "base_events", "exposure"])
def test_reject_missing_participant_or_mismatched_truth(completed, tmp_path, defect):
    request = _request(completed, tmp_path)

    def change(receipt, directory):
        path = directory / "private/evaluations/70/30/participants.parquet"
        table = pd.read_parquet(path)
        if defect == "missing_participant":
            table = table.iloc[1:]
        elif defect == "base_events":
            table.loc[0, "base_event_count"] += 1
        else:
            table.loc[0, "observation_minutes"] += 5
        table.to_parquet(path, index=False)
        receipt["outputs"] = [
            _binding(path) if row["path"] == str(path) else row
            for row in receipt["outputs"]
        ]

    _change_receipt(request, change)
    match = {"missing_participant": "full outer assignment", "base_events": "truth",
             "exposure": "truth/exposure"}[defect]
    with pytest.raises(ValueError, match=match):
        collect_holdout(request)


def test_identity_scan_rejects_inner_outer_leakage_and_missing_roles(tmp_path):
    roles = pd.DataFrame([
        ("s", "i", "selection"), ("s", "c", "calibration"),
        ("s", "o", "outer_evaluation"),
    ], columns=["source_id", "participant_id", "role"])
    path = tmp_path / "predictions.parquet"
    roles.iloc[:2].rename(columns={"role": "partition"}).to_parquet(path, index=False)
    _check_prediction_roles(path, roles, outer=False)
    roles.rename(columns={"role": "partition"}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="role isolation"):
        _check_prediction_roles(path, roles, outer=False)
    roles.iloc[:1].rename(columns={"role": "partition"}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="omit assigned"):
        _check_prediction_roles(path, roles, outer=False)
