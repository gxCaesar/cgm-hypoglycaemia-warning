import numpy as np
import torch

from glycoshift.baselines import (
    build_temporal_baseline,
    linear_extrapolation_forecast,
    persistence_forecast,
)
from glycoshift.models import GlycoShift, GlycoShiftModelConfig, count_parameters
from glycoshift.sequences import SEQUENCE_FEATURES, TASKS


def test_tiny_model_shapes_and_finite_population_memory() -> None:
    config = GlycoShiftModelConfig.variant(
        "tiny",
        context_samples=12,
        memory_k=4,
    )
    model = GlycoShift(config).eval()
    sequence = torch.randn(3, 12, len(SEQUENCE_FEATURES))
    memory = torch.zeros(3, 4, config.d_model)
    padding = torch.ones(3, 4, dtype=torch.bool)
    output = model(
        sequence,
        memory_embeddings=memory,
        memory_padding_mask=padding,
    )
    assert set(output["logits"]) == set(TASKS)
    assert output["representation"].shape == (3, config.d_model)
    assert torch.isfinite(output["fused_representation"]).all()
    assert count_parameters(model) > 500_000


def test_causal_encoder_is_invariant_to_future_values() -> None:
    torch.manual_seed(7)
    config = GlycoShiftModelConfig.variant(
        "tiny",
        context_samples=12,
        memory_k=4,
    )
    model = GlycoShift(config).eval()
    first = torch.randn(1, 12, len(SEQUENCE_FEATURES))
    second = first.clone()
    second[:, 6:] = torch.randn_like(second[:, 6:]) * 50
    with torch.no_grad():
        states_first = model.encode_sequence(first)
        states_second = model.encode_sequence(second)
    torch.testing.assert_close(
        states_first[:, :6],
        states_second[:, :6],
        atol=1e-5,
        rtol=1e-5,
    )


def test_temporal_baseline_registry_shapes() -> None:
    sequence = torch.randn(2, 12, len(SEQUENCE_FEATURES))
    for name in ("lstm", "tcn", "patchtst"):
        model = build_temporal_baseline(name).eval()
        output = model(sequence)
        assert set(output["logits"]) == set(TASKS)
        assert output["representation"].shape[0] == 2


def test_naive_forecasts_are_explicit() -> None:
    values = torch.tensor([100.0, 102.0, 104.0]).numpy()
    assert persistence_forecast(values, horizon_samples=2).tolist() == [104, 104]
    forecast = linear_extrapolation_forecast(values, horizon_samples=2)
    np.testing.assert_allclose(forecast, [106, 108])
