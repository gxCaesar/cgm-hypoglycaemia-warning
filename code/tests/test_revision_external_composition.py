"""Synthetic source/development freezes only; no real model/data acceptance."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_revision_external import _binding, _deep_loader, _fixture, _forbid_fit

from glycoshift import revision_composition as composition
from glycoshift import revision_external as external
from glycoshift import revision_external_composition as consumer
from glycoshift.revision_composition import (
    FrozenRevisionComposition,
    FrozenStepECDF,
    compose_frozen_scores,
    load_frozen_composition,
    save_frozen_composition,
)
from glycoshift.sequences import REVISION_CONTRACT


def _prepared(tmp_path, monkeypatch, *, cut70=False):
    freeze_path, runs, _ = _fixture(tmp_path)
    freeze = json.loads(freeze_path.read_text())
    metadata = {}
    for entry in freeze["runs"]:
        run = Path(entry["run_dir"])
        summary = json.loads((run / "model_summary.json").read_text())
        summary["training_seed"] = entry["config"]["snapshot"]["training"]["seed"]
        (run / "model_summary.json").write_text(json.dumps(summary))
        entry["model_summary"] = {
            "binding": _binding(run / "model_summary.json"),
            "snapshot": summary,
        }
        member = "lstm" if summary.get("architecture") == "lstm" else "persistence"
        metadata[member] = {
            "run_dir": str(run),
            "training_seed": summary["training_seed"],
            "comparison_lstm_seed": 42,
            "shared_control_fit": member == "persistence",
            "revision_contract": REVISION_CONTRACT,
            "input_feature_mode": "cgm_mask_clock",
            "score_columns": {str(h): f"severe_hypo_{h}_probability" for h in (30, 60)},
        }
    source_order = tuple(
        sorted(freeze["runs"][0]["config"]["snapshot"]["data"]["development_sources"])
    )
    policies = {}
    for horizon in (30, 60):
        policy = copy.deepcopy(
            freeze["runs"][0]["model_summary"]["snapshot"]["tasks"][f"hypo_{horizon}"][
                "fixed_frame_policies"
            ]["0.5"]
        )
        policy["endpoint"]["score_column"] = f"composition_{horizon}_score"
        policies[horizon] = policy
    # At the frozen member probability ~.634, ranks are .5 and .75: mean .625.
    mappings = {
        (member, h): FrozenStepECDF((0.2, 0.5, 0.8), counts, 4)
        for member, counts in (("lstm", (1, 2, 4)), ("persistence", (1, 3, 4)))
        for h in (30, 60)
    }
    frozen = FrozenRevisionComposition(42, source_order, mappings, policies, metadata)
    recipe = save_frozen_composition(frozen, tmp_path / "composition-recipe")
    entry = {
        "recipe": _binding(recipe),
        "accepted_development_evidence": freeze["accepted_development_evidence"],
    }
    if cut70:
        entry["cut70_policies"] = {}
        for horizon in (30, 60):
            policy = copy.deepcopy(policies[horizon])
            policy["glucose_cut"] = 70
            path = recipe.parent / f"synthetic-development-cut70-{horizon}.json"
            path.write_text(
                json.dumps(
                    {
                        "policy": policy,
                        "policy_origin": "calibration_refit_at_cut",
                        "base54_policy_parity_checked": True,
                    }
                )
            )
            entry["cut70_policies"][str(horizon)] = {
                "policy": policy,
                "evidence": _binding(path),
            }
    freeze["compositions"] = [entry]
    freeze_path.write_text(json.dumps(freeze))
    _forbid_fit(monkeypatch)
    monkeypatch.setattr(external, "_load_frozen_model", _deep_loader)

    def forbidden(*args, **kwargs):
        raise AssertionError("composition cannot fit external mappings/policies")

    monkeypatch.setattr(FrozenStepECDF, "fit", forbidden)
    monkeypatch.setattr(composition, "fit_revision_composition", forbidden)
    outputs = {}
    for member in ("lstm", "persistence"):
        output = tmp_path / f"external-{member}"
        external.run_revision_external(
            freeze_path,
            run_dir=Path(metadata[member]["run_dir"]),
            partition="external_loop",
            output_dir=output,
            batch_rows=17,
        )
        outputs[member] = output
    return freeze_path, recipe, outputs


def _run(tmp_path, freeze, recipe, outputs):
    return consumer.run_external_composition(
        freeze,
        recipe,
        lstm_output=outputs["lstm"],
        persistence_output=outputs["persistence"],
        partition="external_loop",
        output_dir=tmp_path / "composed",
    )


def test_frozen_score_mapping_full_tail_zero_event_burden_and_typed_shards(
    tmp_path, monkeypatch
):
    freeze, recipe, outputs = _prepared(tmp_path, monkeypatch)
    output = _run(tmp_path, freeze, recipe, outputs)
    predictions = pd.read_parquet(output / "private/predictions")
    assert len(predictions) == 140 and (~predictions.base_evaluable).sum() == 24
    for horizon in (30, 60):
        np.testing.assert_array_equal(
            predictions[f"composition_{horizon}_score"], np.full(140, 0.625)
        )
    assert not any(
        column.endswith(("_probability", "_logit", "_prediction", "_prediction_scaled"))
        for column in predictions
    )
    events = pd.read_parquet(output / "private/endpoints/events-30")
    assert len(events) == 1  # First participant shard has no event.
    metrics = pd.read_parquet(output / "private/evaluations/54/30/0.5/participants")
    assert len(metrics) == 2
    assert (
        metrics.loc[metrics.base_event_count.eq(0), "delivered_alarm_count"].iloc[0] > 0
    )
    report = json.loads((output / "exportable/fixed-frame-summary.json").read_text())
    assert len(report["policies"]) == 2  # Member cut70 policies are not inherited.
    assert {item["participant_count"] for item in report["policies"]} == {2}
    assert {
        item["event_positive_participant_count"] for item in report["policies"]
    } == {1}
    assert report["fitted_on_external"] is False
    assert report["inherited_probability_or_regression_diagnostics"] is False
    assert (
        report["score_interpretation"] == "rank_mean_not_calibrated_event_probability"
    )
    assert (
        "participant_id"
        not in (output / "exportable/fixed-frame-summary.json").read_text()
    )
    glucose = json.loads((output / "exportable/alarm-glucose-summary.json").read_text())
    assert len(glucose["rows"]) == 16
    assert glucose["participant_quantiles_averaged"] is False
    emitted = next(
        row
        for row in glucose["rows"]
        if row["scope"] == "pooled_alarms"
        and row["alarm_category"] == "emitted"
        and row["horizon_minutes"] == 30
    )
    assert emitted["alarm_count"] == report["policies"][0]["emitted_alarm_count"]
    assert emitted["median"] == 120.0
    receipt = json.loads((output / "run-receipt.json").read_text())
    assert receipt["producer_kind"] == "frozen_step_ecdf_composition"
    assert receipt["recipe"] == _binding(recipe)
    assert receipt["member_receipts"] == {
        member: _binding(root / "run-receipt.json") for member, root in outputs.items()
    }
    bound = {item["relative_path"] for item in receipt["files"]}
    assert {
        str(path.relative_to(output))
        for path in (output / "private").rglob("*.parquet")
    }.issubset(bound)
    assert "private/composition-input-receipt.json" in bound
    assert not (output / "run-receipt.pending.json").exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(tmp_path, freeze, recipe, outputs)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "truth"])
def test_exact_member_joins_never_intersect_or_silently_accept_truth(
    tmp_path, monkeypatch, defect
):
    _, recipe, outputs = _prepared(tmp_path, monkeypatch)
    paths = {
        member: sorted((outputs[member] / "private/predictions").glob("*.parquet"))[0]
        for member in outputs
    }
    summaries = {
        member: json.loads((outputs[member] / "run-receipt.json").read_text())[
            "model_summary_snapshot"
        ]
        for member in outputs
    }
    frames = consumer._member_frames(paths, summaries, "external_loop")
    right = frames["persistence"]
    if defect == "missing":
        frames["persistence"] = right.iloc[1:].copy()
    elif defect == "duplicate":
        frames["persistence"] = pd.concat([right, right.iloc[[0]]], ignore_index=True)
    else:
        frames["persistence"] = right.copy()
        frames["persistence"].loc[0, "cgm_last"] += 1
    with pytest.raises(
        ValueError, match="populations differ|duplicate decision keys|truth differs"
    ):
        compose_frozen_scores(load_frozen_composition(recipe), frames)


def test_producer_receipt_missing_or_changed_file_is_rejected(tmp_path, monkeypatch):
    freeze, recipe, outputs = _prepared(tmp_path, monkeypatch)
    receipt_path = outputs["lstm"] / "run-receipt.json"
    content = receipt_path.read_text()
    receipt_path.rename(outputs["lstm"] / "hidden-synthetic-receipt.json")
    with pytest.raises(ValueError, match="native completed external producer receipt"):
        _run(tmp_path, freeze, recipe, outputs)
    receipt_path.write_text(content)
    path = sorted((outputs["lstm"] / "private/predictions").glob("*.parquet"))[0]
    frame = pd.read_parquet(path)
    frame.loc[0, "cgm_last"] += 1
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="size/mtime binding changed"):
        _run(tmp_path, freeze, recipe, outputs)


@pytest.mark.parametrize("defect", ["changed_allocation", "missing_binding"])
def test_evaluation_ledgers_require_their_exact_native_binding(
    tmp_path, monkeypatch, defect
):
    freeze, recipe, outputs = _prepared(tmp_path, monkeypatch)
    root = outputs["lstm"]
    if defect == "changed_allocation":
        path = sorted(
            (root / "private/evaluations/54/30/0.5/participants").glob("*.parquet")
        )[0]
        frame = pd.read_parquet(path)
        frame.loc[0, "detected_event_count"] += 1
        frame.to_parquet(path, index=False)
        message = "size/mtime binding changed"
    else:
        path = root / "run-receipt.json"
        receipt = json.loads(path.read_text())
        receipt["files"] = [
            item
            for item in receipt["files"]
            if not item["relative_path"].startswith("private/evaluations/")
        ]
        path.write_text(json.dumps(receipt))
        message = "evaluation files differ"
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, freeze, recipe, outputs)


def test_independent_event_identity_and_exposure_are_paired_exactly(
    tmp_path, monkeypatch
):
    _, _, outputs = _prepared(tmp_path, monkeypatch)
    paths = {
        member: sorted((outputs[member] / "private/predictions").glob("*.parquet"))[1]
        for member in outputs
    }
    event = (
        outputs["persistence"]
        / "private/endpoints/events-30"
        / paths["persistence"].name
    )
    frame = pd.read_parquet(event)
    frame.event_id = "same-count-wrong-event"
    frame.to_parquet(event, index=False)
    with pytest.raises(ValueError, match="base event/exposure truth differs"):
        consumer._paired_truth(
            outputs, paths, ("Loop Observational Study", "synthetic-A"), 30
        )


def test_cut70_only_uses_explicit_frozen_development_composition_policy(
    tmp_path, monkeypatch
):
    freeze, recipe, outputs = _prepared(tmp_path, monkeypatch, cut70=True)
    output = _run(tmp_path, freeze, recipe, outputs)
    result = json.loads((output / "exportable/fixed-frame-summary.json").read_text())
    assert len(result["policies"]) == 4
    assert {item["glucose_cut"] for item in result["policies"]} == {54, 70}
    assert {item["threshold"] for item in result["policies"]} == {0.6}


def test_ohio_and_changed_recipe_are_rejected(tmp_path, monkeypatch):
    freeze, recipe, outputs = _prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="already exposed Loop/HUPA"):
        consumer.run_external_composition(
            freeze,
            recipe,
            lstm_output=outputs["lstm"],
            persistence_output=outputs["persistence"],
            partition="confirmatory_ohio",
            output_dir=tmp_path / "bad",
        )
    payload = json.loads(recipe.read_text())
    payload["policies"]["30"]["threshold"] = 0.7
    recipe.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="size/mtime binding changed"):
        _run(tmp_path, freeze, recipe, outputs)


def test_changed_member_receipt_during_final_export_cannot_complete(
    tmp_path, monkeypatch
):
    freeze, recipe, outputs = _prepared(tmp_path, monkeypatch)
    export = consumer._export_alarm_glucose

    def change_receipt_after_export(*args, **kwargs):
        export(*args, **kwargs)
        receipt = outputs["lstm"] / "run-receipt.json"
        receipt.write_text(receipt.read_text() + "\n")

    monkeypatch.setattr(consumer, "_export_alarm_glucose", change_receipt_after_export)
    with pytest.raises(ValueError, match="size/mtime binding changed"):
        _run(tmp_path, freeze, recipe, outputs)
    status = json.loads((tmp_path / "composed/status.json").read_text())
    assert status["status"] == "failed"
    assert not (tmp_path / "composed/run-receipt.json").exists()
