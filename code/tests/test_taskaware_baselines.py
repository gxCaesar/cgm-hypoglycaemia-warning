from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from glycoshift.schema import ACTION_FIELDS
from glycoshift.sequences import (
    SEQUENCE_FEATURES,
    TASKS,
    ForecastWindowSequenceDataset,
    SequenceScaler,
)
from glycoshift.taskaware_baselines import (
    PatchMLPForecaster,
    PatchMLPForecasterConfig,
    forecast_task_probabilities,
    glimmer_weighted_mae,
    glimmer_weighted_mae_per_example,
    inverse_standardized_cgm,
)


def test_patchmlp_forecaster_shapes_and_gradients() -> None:
    torch.manual_seed(17)
    config = PatchMLPForecasterConfig(
        context_samples=72,
        prediction_samples=12,
        d_model=128,
        encoder_layers=1,
        dropout=0.0,
    )
    model = PatchMLPForecaster(config)
    sequence = torch.randn(3, 72, len(SEQUENCE_FEATURES))
    output = model(sequence)

    assert output["forecast"].shape == (3, 12)
    assert output["all_channel_forecasts"].shape == (
        3,
        12,
        len(SEQUENCE_FEATURES),
    )
    loss = output["forecast"].square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_patchmlp_rejects_contract_drift() -> None:
    config = PatchMLPForecasterConfig(
        context_samples=72,
        prediction_samples=12,
        d_model=128,
    )
    model = PatchMLPForecaster(config)
    with pytest.raises(ValueError, match="frozen context contract"):
        model(torch.randn(2, 71, len(SEQUENCE_FEATURES)))
    with pytest.raises(ValueError, match="non-finite"):
        invalid = torch.randn(2, 72, len(SEQUENCE_FEATURES))
        invalid[0, 0, 0] = float("nan")
        model(invalid)


def test_patchmlp_encoder_reuses_public_commit_residual_norm() -> None:
    config = PatchMLPForecasterConfig(
        context_samples=72,
        prediction_samples=12,
        d_model=128,
        dropout=0.0,
    )
    encoder = PatchMLPForecaster(config).seasonal_layers[0]
    encoder.eval()
    values = torch.randn(2, len(SEQUENCE_FEATURES), 128)
    with torch.no_grad():
        semantic = encoder.residual_norm(values + encoder.semantic_mlp(values))
        mixed = encoder.channel_mlp(semantic.transpose(1, 2)).transpose(1, 2)
        expected = encoder.residual_norm(values + semantic * mixed)
        observed = encoder(values)
    assert torch.equal(observed, expected)


def test_glimmer_weighted_mae_matches_region_oracle() -> None:
    target = torch.tensor([[50.0, 100.0, 200.0]])
    forecast = torch.tensor([[60.0, 90.0, 190.0]], requires_grad=True)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = glimmer_weighted_mae(forecast, target, mask)
    expected = 10.0 * (3.296363582 + 1.0 + 2.382397706) / 3.0
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert forecast.grad is not None
    assert torch.isfinite(forecast.grad).all()


def test_glimmer_loss_fails_closed_on_empty_or_nonfinite_targets() -> None:
    forecast = torch.tensor([[100.0]])
    target = torch.tensor([[100.0]])
    with pytest.raises(ValueError, match="at least one"):
        glimmer_weighted_mae(
            forecast,
            target,
            torch.zeros_like(target, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="finite"):
        glimmer_weighted_mae(
            torch.tensor([[float("nan")]]),
            target,
            torch.ones_like(target, dtype=torch.bool),
        )


def test_glimmer_per_example_mask_and_mg_dl_inverse_transform() -> None:
    standardized = torch.tensor([[-1.0, 0.0, 1.0]])
    forecast_mg_dl = inverse_standardized_cgm(
        standardized,
        mean_mg_dl=100.0,
        std_mg_dl=50.0,
    )
    assert torch.equal(forecast_mg_dl, torch.tensor([[50.0, 100.0, 150.0]]))
    target = torch.tensor([[50.0, 110.0, 200.0]])
    mask = torch.tensor([[True, False, True]])
    losses = glimmer_weighted_mae_per_example(forecast_mg_dl, target, mask)
    expected = 50.0 * 2.382397706 / 2.0
    assert losses.tolist() == pytest.approx([expected])
    with pytest.raises(ValueError, match="positive"):
        inverse_standardized_cgm(
            standardized,
            mean_mg_dl=100.0,
            std_mg_dl=0.0,
        )


def test_forecast_task_probabilities_are_horizon_and_direction_aware() -> None:
    forecast = torch.tensor(
        [
            [100.0, 90.0, 80.0, 75.0, 72.0, 68.0, 65.0, 60.0, 55.0, 50.0, 45.0, 40.0],
            [
                100.0,
                110.0,
                120.0,
                130.0,
                150.0,
                170.0,
                185.0,
                190.0,
                200.0,
                210.0,
                220.0,
                230.0,
            ],
        ]
    )
    probabilities = forecast_task_probabilities(forecast)
    assert set(probabilities) == set(TASKS)
    assert probabilities["hypo_60"][0] > probabilities["hypo_30"][0]
    assert probabilities["hyper_60"][1] > probabilities["hyper_30"][1]
    assert probabilities["hypo_60"][0] > probabilities["hypo_60"][1]
    assert probabilities["hyper_60"][1] > probabilities["hyper_60"][0]
    assert all(
        torch.all((values > 0) & (values < 1))
        for values in probabilities.values()
    )


def test_forecast_dataset_reads_only_contiguous_same_participant_future() -> None:
    timestamps = pd.date_range("2026-01-01", periods=90, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "source_id": "source-a",
            "participant_id": "participant-a",
            "timestamp": timestamps,
            "timestamp_local": timestamps.tz_localize(None),
            "cgm_mg_dl": np.arange(90, dtype=float) + 80.0,
            "cgm_observed": True,
            "cgm_lower_censored": False,
        }
    )
    for field in ACTION_FIELDS:
        frame[field] = 0.0
        frame[f"{field}_available"] = True
        frame[f"{field}_observed"] = True
        frame[f"{field}_zero_ambiguous"] = False
    frame.loc[75, "cgm_lower_censored"] = True
    window: dict[str, object] = {
        "source_id": "source-a",
        "participant_id": "participant-a",
        "context_start": timestamps[0],
        "context_end": timestamps[71],
        "patient_day": timestamps[71].floor("D"),
        "participant_end_index": 71,
        "sampling_weight": 1.0,
    }
    for task in TASKS:
        kind = task.split("_")[0]
        window[f"{kind}_currently_active"] = False
        window[f"{task}_label"] = 0
    for horizon, target_index in ((30, 77), (60, 83)):
        window[f"cgm_{horizon}_target"] = float(frame.loc[target_index, "cgm_mg_dl"])
        window[f"cgm_{horizon}_target_available"] = True
    windows = pd.DataFrame([window])
    scaler = SequenceScaler.fit(frame, {("source-a", "participant-a")})
    dataset = ForecastWindowSequenceDataset(
        frame,
        windows,
        scaler=scaler,
        context_samples=72,
        prediction_samples=12,
        sample_minutes=5,
    )
    item = dataset[0]

    assert item["forecast_targets"].shape == (12,)
    assert item["forecast_targets_mg_dl"].shape == (12,)
    assert item["forecast_mask"].sum().item() == 11
    assert not bool(item["forecast_mask"][3])
    assert float(item["forecast_targets_mg_dl"][0]) == 152.0


def test_forecast_dataset_fails_on_future_gap() -> None:
    timestamps = pd.date_range("2026-01-01", periods=14, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "source_id": "s",
            "participant_id": "p",
            "timestamp": timestamps,
            "cgm_mg_dl": np.arange(14, dtype=float) + 100.0,
            "cgm_observed": True,
        }
    )
    for field in ACTION_FIELDS:
        frame[field] = 0.0
        frame[f"{field}_available"] = True
        frame[f"{field}_observed"] = True
        frame[f"{field}_zero_ambiguous"] = False
    frame.loc[3:, "timestamp"] = frame.loc[3:, "timestamp"] + pd.Timedelta(minutes=5)
    window: dict[str, object] = {
        "source_id": "s",
        "participant_id": "p",
        "context_start": timestamps[0],
        "context_end": timestamps[1],
        "patient_day": timestamps[1].floor("D"),
        "participant_end_index": 1,
    }
    for task in TASKS:
        kind = task.split("_")[0]
        window[f"{kind}_currently_active"] = False
        window[f"{task}_label"] = 0
    for horizon in (30, 60):
        window[f"cgm_{horizon}_target"] = 100.0
        window[f"cgm_{horizon}_target_available"] = True
    scaler = SequenceScaler.fit(frame, {("s", "p")})
    dataset = ForecastWindowSequenceDataset(
        frame,
        pd.DataFrame([window]),
        scaler=scaler,
        context_samples=2,
        prediction_samples=12,
        sample_minutes=5,
    )
    with pytest.raises(ValueError, match="not contiguous"):
        dataset[0]
