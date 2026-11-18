import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_fixed_frame_evaluation import _evaluate, _events, _timeline
from test_revision_prepare import crossfit_inputs

import glycoshift.revision_crossfit_evaluation as evaluation_module
from glycoshift.config import load_config
from glycoshift.revision_baselines import run_revision_baseline
from glycoshift.revision_crossfit_evaluation import (
    aggregate_oof_participants,
    freeze_inner_policies,
    run_crossfit_evaluation,
)
from glycoshift.revision_crossfit_prepare import prepare_crossfit_replays
from glycoshift.revision_inputs import CONFIRMATION_CONTRACT
from glycoshift.revision_runner import revision_config
from glycoshift.windows import build_windows


def _inputs(tmp_path, monkeypatch):
    normalized, legacy, fixture = crossfit_inputs(tmp_path)
    for index in range(9):
        path = normalized / f"part-{index}.parquet"
        frame = pd.read_parquet(path)
        frame.loc[50:54, "cgm_mg_dl"] = 230.0
        frame.to_parquet(path, index=False)
        windows = build_windows(frame, fixture)
        windows["partition"] = ("train", "selection", "calibration")[index % 3]
        windows.to_parquet(legacy / f"part-{index}.parquet", index=False)
    config = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
    config["data"].update(fixture["data"])
    config["protocol"].update(
        fixture["protocol"], ascertainment_contract=CONFIRMATION_CONTRACT
    )
    config["training"].update(num_workers=1, device="cpu")
    config["evaluation"].update(threshold_candidate_count=5)
    config = revision_config(config, seed=17, output=tmp_path / "model")
    prepared = tmp_path / "prepared"
    prepare_crossfit_replays(normalized, legacy, prepared, config)

    def load_frame(_config, *, sources, participant_keys):
        assert sources == {"s"}
        return pd.concat(
            [
                pd.read_parquet(normalized / f"part-{participant}.parquet")
                for _, participant in sorted(participant_keys)
            ],
            ignore_index=True,
        )

    monkeypatch.setattr("glycoshift.revision_runner.load_frame", load_frame)
    return prepared, config


def _fit(tmp_path, monkeypatch):
    prepared, config = _inputs(tmp_path, monkeypatch)
    run = run_revision_baseline(
        config, prepared, baseline="summary_logistic", outer_fold=0
    )
    return prepared, run


def test_outer_values_open_only_after_policy_freeze_and_no_refit(tmp_path, monkeypatch):
    prepared, run = _fit(tmp_path, monkeypatch)
    output = tmp_path / "outer"
    loader = evaluation_module.load_crossfit_outer_bundle

    def guarded_outer(*args, **kwargs):
        freeze = json.loads((output / "private/policy-freeze.json").read_text())
        assert not freeze["outer_values_opened"]
        assert freeze["base54_policy_parity_checked"]
        assert freeze["policy_fit_role"] == "calibration"
        assert set(freeze["policies"]) == {"54/30", "54/60", "70/30", "70/60"}

        # Everything that fits has finished when this entrypoint first opens.
        def forbidden(*args, **kwargs):
            pytest.fail("an outer value was used before fitting had finished")

        monkeypatch.setattr(
            evaluation_module, "fit_fixed_frame_alarm_policy", forbidden
        )
        monkeypatch.setattr("glycoshift.calibration.fit_temperature", forbidden)
        monkeypatch.setattr("sklearn.linear_model.LogisticRegression.fit", forbidden)
        return loader(*args, **kwargs)

    monkeypatch.setattr(evaluation_module, "load_crossfit_outer_bundle", guarded_outer)
    run_crossfit_evaluation(prepared, run, outer_fold=0, output=output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete" and not receipt["outer_refitting"]
    predictions = pd.read_parquet(output / "private/predictions.parquet")
    assert predictions.partition.eq("outer_evaluation").all()
    inner = pd.read_parquet(run / "predictions.parquet")
    assert not set(predictions.participant_id).intersection(inner.participant_id)
    assert (~predictions.base_evaluable).any()
    assert predictions.outer_fold.eq(0).all()
    assert len(receipt["outputs"]) == 17


def test_wrong_fold_fit_rejected_before_loading_outer_values(tmp_path, monkeypatch):
    prepared, run = _fit(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outer-fold complement"):
        freeze_inner_policies(prepared, run, outer_fold=1, output=tmp_path / "bad.json")


def test_oof_aggregation_uses_participants_and_retains_missing_sources():
    timeline = _timeline(
        [
            ("A", "a", 0, 100, 1.0, True),
            ("A", "b", 0, 100, 0.0, True),
            ("A", "c", 0, 100, 0.0, True),
            ("B", "d", 0, 100, 1.0, True),
        ]
    )
    events = _events(
        [
            (s, p, "event", 10)
            for s, p in [("A", "a"), ("A", "b"), ("A", "c"), ("B", "d")]
        ]
    )
    table = _evaluate(timeline, events).participants
    table["outer_fold"] = [0, 1, 1, 0]
    result = aggregate_oof_participants(table, ("A", "B"))
    assert result["equal_source_participant_macro_fixed_sensitivity"] == pytest.approx(
        2 / 3
    )
    assert result["participant_pooled"]["fixed_sensitivity"] == 0.5
    assert result["sources"][0]["fixed_sensitivity"] == pytest.approx(1 / 3)
    missing = aggregate_oof_participants(table, ("A", "B", "C"))
    assert missing["undefined_sensitivity_sources"] == ["C"]
    assert np.isnan(missing["equal_source_participant_macro_fixed_sensitivity"])
    with pytest.raises(ValueError, match="unique"):
        aggregate_oof_participants(pd.concat([table, table]), ("A", "B"))
