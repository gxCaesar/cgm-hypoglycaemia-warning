from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import glycoshift.revision_baselines as baseline_module
from glycoshift.config import load_config
from glycoshift.revision_baselines import (
    CGM_SUMMARY_FEATURES,
    REVISION_SUMMARY_FEATURES,
    build_revision_features,
    run_revision_baseline,
)
from glycoshift.revision_prepare import prepare_replays
from glycoshift.revision_runner import load_replay_bundle, revision_config
from glycoshift.schema import ACTION_FIELDS
from glycoshift.windows import build_windows


def _bundle(tmp_path, monkeypatch, *, heldout_shift=0):
    from test_revision_inputs import fixture

    raw, protocol = fixture()
    config = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
    config["protocol"].update(protocol["protocol"])
    config["data"].update(protocol["data"], development_sources=["s"])
    config["training"].update(num_workers=1, negative_ratio=0)
    config["evaluation"].update(threshold_candidate_count=5)
    normalized, legacy = tmp_path / "normalized", tmp_path / "legacy"
    normalized.mkdir(parents=True)
    legacy.mkdir()
    parts = []
    for index, partition in enumerate(("train", "selection", "calibration")):
        part = raw.copy(deep=True)
        part["participant_id"] = str(index)
        part["cgm_mg_dl"] = 100 + np.arange(len(part)) * 0.25
        if partition != "train":
            part["cgm_mg_dl"] += heldout_shift
        part.loc[22:25, "cgm_mg_dl"] = 65
        part.loc[35:38, "cgm_mg_dl"] = 50
        part.loc[50:54, "cgm_mg_dl"] = 230
        part.loc[20, ["cgm_observed", "cgm_interpolated"]] = [False, True]
        part["timestamp_local"] = (
            part["timestamp"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        )
        part.to_parquet(normalized / f"part-{index}.parquet", index=False)
        old = build_windows(part, config)
        old["partition"] = partition
        old.to_parquet(legacy / f"part-{index}.parquet", index=False)
        parts.append(part)
    config = revision_config(config, seed=17, output=tmp_path / "run")
    prepared = tmp_path / "prepared"
    prepare_replays(normalized, legacy, prepared, config)
    normalized_frame = pd.concat(parts, ignore_index=True)

    def load_frame(_config, *, sources, participant_keys):
        assert sources == {"s"}
        assert participant_keys == {("s", str(index)) for index in range(3)}
        return normalized_frame.copy(deep=True)

    monkeypatch.setattr("glycoshift.revision_runner.load_frame", load_frame)
    return config, prepared, normalized_frame


def test_features_have_hand_calculated_mask_and_recorded_clock_and_no_actions(
    tmp_path,
    monkeypatch,
):
    config, prepared, _ = _bundle(tmp_path, monkeypatch)
    (frame, windows, _), _ = load_replay_bundle(prepared, config, sequences=True)
    features = build_revision_features(frame, windows, config)
    assert tuple(features) == REVISION_SUMMARY_FEATURES
    row = windows.loc[
        windows.participant_id.eq("0") & windows.participant_end_index.eq(21)
    ].index.item()
    assert features.loc[row, "cgm_observed_fraction"] == pytest.approx(5 / 6)
    assert features.attrs["clock_column"] == "timestamp_local"
    minute = 8 * 60 + 21 * 5
    assert features.loc[row, "time_of_day_sin"] == pytest.approx(
        np.sin(2 * np.pi * minute / 1440)
    )
    assert features.loc[row, "time_of_day_cos"] == pytest.approx(
        np.cos(2 * np.pi * minute / 1440)
    )
    for name in CGM_SUMMARY_FEATURES:
        np.testing.assert_allclose(features[name], windows[name])
    changed = frame.copy()
    for field in ACTION_FIELDS:
        changed[field] = 999999.0
        changed[f"{field}_observed"] = True
        changed[f"{field}_available"] = True
        changed[f"{field}_zero_ambiguous"] = False
    pd.testing.assert_frame_equal(
        features, build_revision_features(changed, windows, config)
    )


def test_future_normalized_values_and_masks_cannot_change_past_features(
    tmp_path,
    monkeypatch,
):
    config, prepared, _ = _bundle(tmp_path, monkeypatch)
    (frame, windows, _), _ = load_replay_bundle(prepared, config, sequences=True)
    past = windows.loc[windows.participant_end_index < 40].copy()
    original = build_revision_features(frame, past, config)
    changed = frame.copy(deep=True)
    future = changed.groupby(["source_id", "participant_id"]).cumcount() >= 40
    changed.loc[future, "cgm_mg_dl"] = 400
    changed.loc[future, "cgm_observed"] = False
    changed.loc[future, "cgm_interpolated"] = True
    pd.testing.assert_frame_equal(
        original, build_revision_features(changed, past, config)
    )


@pytest.mark.parametrize(
    "baseline", ["persistence", "linear", "summary_logistic", "xgboost"]
)
def test_complete_baseline_training_calibration_and_fixed_report(
    tmp_path,
    monkeypatch,
    baseline,
):
    if baseline == "xgboost":
        pytest.importorskip("xgboost")
    config, prepared, _ = _bundle(tmp_path, monkeypatch)
    run = run_revision_baseline(config, prepared, baseline=baseline)
    summary = json.loads((run / "model_summary.json").read_text())
    target_audit = json.loads((run / "training_target_audit.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    contract = json.loads((run / "feature_contract.json").read_text())
    predictions = pd.read_parquet(run / "predictions.parquet")
    models = joblib.load(run / "baseline_models.joblib")
    (_, windows, _), _ = load_replay_bundle(prepared, config, sequences=True)
    train = windows.loc[windows.partition.eq("train")]
    assert summary["baseline"] == baseline
    assert (
        target_audit["training_sampling"] == "all_masked_rows_no_negative_subsampling"
    )
    assert not contract["action_features_used"]
    assert contract["allowed_summary_features"] == list(REVISION_SUMMARY_FEATURES)
    assert contract["raw_score_transform"] == "predict_probability_clip_1e-6_then_logit"
    assert predictions.partition.value_counts().to_dict() == {
        partition: int(windows.partition.eq(partition).sum())
        for partition in ("selection", "calibration")
    }
    assert (~predictions.base_evaluable).any()
    for task in ("hypo_30", "hypo_60", "hyper_30", "hyper_60"):
        mask = train[f"{task}_training_mask"]
        expected = train.loc[mask, f"{task}_training_target"]
        assert target_audit["tasks"][task]["retained_count"] == int(mask.sum())
        assert target_audit["tasks"][task]["positive_count"] == int(expected.sum())
        assert summary["tasks"][task]["temperature_fit_partition"] == "calibration"
        if baseline == "xgboost":
            params = models[task]["model"].get_params()
            assert {
                key: params[key]
                for key in (
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                )
            } == {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
            assert summary["tasks"][task]["boosted_rounds"] == 300
        else:
            params = models[task]["model"].named_steps["model"].get_params()
            assert (params["class_weight"], params["max_iter"], params["solver"]) == (
                "balanced",
                500,
                "liblinear",
            )
            assert summary["tasks"][task]["fit_status"] == "converged"
            assert len(summary["tasks"][task]["coefficients"]) == len(
                models[task]["features"]
            )
    assert (
        train.loc[train.hypo_30_training_mask, "hypo_30_label"].sum()
        != (target_audit["tasks"]["hypo_30"]["positive_count"])
    )
    for horizon in (30, 60):
        np.testing.assert_array_equal(
            predictions[f"hypo_{horizon}_probability"],
            predictions[f"severe_hypo_{horizon}_probability"],
        )
        for partition in ("selection", "calibration"):
            reported = metrics["partitions"][partition]
            assert ("continuous_forecast" in reported) == (
                baseline in {"persistence", "linear"}
            )
            assert reported["fixed_frame"][str(horizon)]["0.5"]["base_event_count"] == 1
    assert summary["continuous_prediction_available"] == (
        baseline in {"persistence", "linear"}
    )
    if baseline in {"summary_logistic", "xgboost"}:
        assert not any(f"cgm_{h}_prediction" in predictions for h in (30, 60))
    else:
        for horizon in (30, 60):
            expected = predictions.cgm_last.to_numpy()
            if baseline == "linear":
                expected = expected + predictions.cgm_slope_30.to_numpy() * (
                    horizon // 5
                )
            np.testing.assert_allclose(
                predictions[f"cgm_{horizon}_prediction"], expected
            )
    for name in (
        "config_resolved.yaml",
        "operational_metrics.json",
        "training_summary.json",
    ):
        assert (run / name).is_file()
    with pytest.raises(FileExistsError):
        run_revision_baseline(config, prepared, baseline=baseline)


def test_only_training_rows_fit_scalers_and_coefficients(tmp_path, monkeypatch):
    first_config, prepared, _ = _bundle(tmp_path / "first", monkeypatch)
    first = run_revision_baseline(first_config, prepared, baseline="summary_logistic")
    second_config, prepared, _ = _bundle(
        tmp_path / "second", monkeypatch, heldout_shift=30
    )
    second = run_revision_baseline(second_config, prepared, baseline="summary_logistic")
    first_models = joblib.load(first / "baseline_models.joblib")
    second_models = joblib.load(second / "baseline_models.joblib")
    for task in first_models:
        one, two = first_models[task]["model"], second_models[task]["model"]
        np.testing.assert_array_equal(
            one.named_steps["scale"].mean_, two.named_steps["scale"].mean_
        )
        np.testing.assert_array_equal(
            one.named_steps["model"].coef_, two.named_steps["model"].coef_
        )
        np.testing.assert_array_equal(
            one.named_steps["model"].intercept_, two.named_steps["model"].intercept_
        )


def test_action_values_and_provenance_cannot_change_fitted_summary_model_or_predictions(
    tmp_path,
    monkeypatch,
):
    config, prepared, _ = _bundle(tmp_path, monkeypatch)
    first = run_revision_baseline(config, prepared, baseline="summary_logistic")
    original_loader = baseline_module.load_replay_bundle

    def changed_actions(*args, **kwargs):
        (frame, windows, audit), replays = original_loader(*args, **kwargs)
        for field in ACTION_FIELDS:
            frame[field] = -123456.0
            for suffix in ("observed", "available", "zero_ambiguous"):
                frame[f"{field}_{suffix}"] = True
        for column in windows:
            if any(column.startswith(f"{field}_") for field in ACTION_FIELDS):
                windows[column] = 12345
        return (frame, windows, audit), replays

    monkeypatch.setattr(baseline_module, "load_replay_bundle", changed_actions)
    config["output"]["run_dir"] = str(tmp_path / "changed-actions")
    second = run_revision_baseline(config, prepared, baseline="summary_logistic")
    first_models = joblib.load(first / "baseline_models.joblib")
    second_models = joblib.load(second / "baseline_models.joblib")
    for task in first_models:
        np.testing.assert_array_equal(
            first_models[task]["model"].named_steps["model"].coef_,
            second_models[task]["model"].named_steps["model"].coef_,
        )
    one, two = (
        pd.read_parquet(first / "predictions.parquet"),
        pd.read_parquet(second / "predictions.parquet"),
    )
    score_columns = [
        column for column in one if column.endswith(("_logit", "_probability"))
    ]
    pd.testing.assert_frame_equal(one[score_columns], two[score_columns])


@pytest.mark.parametrize(
    "kind", ["end_index", "start", "cgm", "observed", "censored", "summary"]
)
def test_bad_normalized_context_binding_fails_closed(tmp_path, monkeypatch, kind):
    config, prepared, _ = _bundle(tmp_path, monkeypatch)
    (frame, windows, _), _ = load_replay_bundle(prepared, config, sequences=True)
    row = windows.loc[windows.participant_id.eq("0")].index[0]
    anchor = int(windows.loc[row, "participant_end_index"])
    if kind == "end_index":
        windows.loc[row, "participant_end_index"] += 1
    elif kind == "start":
        windows.loc[row, "context_start"] += pd.Timedelta(minutes=5)
    elif kind == "cgm":
        frame.loc[anchor, "cgm_mg_dl"] += 10
    elif kind == "observed":
        frame.loc[anchor, "cgm_observed"] = False
    elif kind == "censored":
        frame.loc[anchor - 1, "cgm_lower_censored"] = True
    else:
        windows.loc[row, "cgm_mean"] += 1
    with pytest.raises(ValueError):
        build_revision_features(frame, windows, config)
