from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .sequences import SEQUENCE_FEATURES, TASKS


@dataclass(frozen=True)
class PatchMLPForecasterConfig:
    """Configuration for the clean-room PatchMLP forecasting adapter.

    The adapter follows the public PatchMLP architecture at commit
    ``b36bbc92ecfc4732acaabb6d5e8c4ff487876f5d``: multi-scale overlapping
    patches, channel mixing, series decomposition, and a linear multi-horizon
    projector. It predicts every standardized input channel and exposes the
    CGM channel as the glucose forecast. The surrounding data and evaluation
    contract remain GlycoShift's frozen six-hour participant protocol.
    """

    context_samples: int
    prediction_samples: int = 12
    input_dim: int = len(SEQUENCE_FEATURES)
    d_model: int = 1024
    encoder_layers: int = 1
    moving_average_kernel: int = 13
    dropout: float = 0.1
    patch_lengths: tuple[int, ...] = (48, 24, 12, 6)
    use_instance_normalization: bool = True
    cgm_feature_index: int = 0

    def validate(self) -> None:
        if self.context_samples <= 0 or self.prediction_samples <= 0:
            raise ValueError("context and prediction lengths must be positive")
        if self.input_dim <= 0 or self.d_model <= 0:
            raise ValueError("input_dim and d_model must be positive")
        if self.encoder_layers <= 0:
            raise ValueError("encoder_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.moving_average_kernel <= 0 or self.moving_average_kernel % 2 == 0:
            raise ValueError("moving_average_kernel must be a positive odd integer")
        if not self.patch_lengths:
            raise ValueError("at least one patch length is required")
        if self.d_model % len(self.patch_lengths) != 0:
            raise ValueError("d_model must be divisible by the number of patch scales")
        for patch_length in self.patch_lengths:
            if patch_length <= 1 or patch_length > self.context_samples:
                raise ValueError("every patch length must be in [2, context_samples]")
            if patch_length // 2 <= 0:
                raise ValueError(
                    "every patch scale requires a positive half-patch stride"
                )
        if not 0 <= self.cgm_feature_index < self.input_dim:
            raise ValueError("cgm_feature_index is outside the input feature range")


class _MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.average = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        left = values[:, :, :1].expand(-1, -1, padding)
        right = values[:, :, -1:].expand(-1, -1, padding)
        return self.average(torch.cat((left, values, right), dim=-1))


class _SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_average = _MovingAverage(kernel_size)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(values)
        return values - trend, trend


class _PatchScale(nn.Module):
    def __init__(
        self,
        *,
        context_samples: int,
        patch_length: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        stride = patch_length // 2
        patch_count = (context_samples - patch_length) // stride + 1
        if patch_count <= 0:
            raise ValueError("patch scale produces no patches")
        patch_dim = output_dim // patch_count
        if patch_dim <= 0:
            raise ValueError(
                "per-scale d_model is too small for the requested patch count"
            )
        self.patch_length = int(patch_length)
        self.stride = int(stride)
        self.patch_projection = nn.Linear(patch_length, patch_dim)
        self.scale_projection = nn.Linear(patch_dim * patch_count, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        patches = values.unfold(
            dimension=-1,
            size=self.patch_length,
            step=self.stride,
        )
        embedded = self.patch_projection(patches)
        flattened = embedded.flatten(start_dim=-2)
        return self.scale_projection(flattened)


class _MultiScalePatchEmbedding(nn.Module):
    def __init__(self, config: PatchMLPForecasterConfig) -> None:
        super().__init__()
        scale_dim = config.d_model // len(config.patch_lengths)
        self.scales = nn.ModuleList(
            [
                _PatchScale(
                    context_samples=config.context_samples,
                    patch_length=patch_length,
                    output_dim=scale_dim,
                )
                for patch_length in config.patch_lengths
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat([scale(values) for scale in self.scales], dim=-1)


class _PatchMLPEncoder(nn.Module):
    def __init__(self, *, d_model: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.semantic_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # The public PatchMLP commit defines two norms but calls the first norm
        # after both residual blocks. Reusing one module here preserves that
        # observable upstream behaviour; the second unused module is omitted.
        self.residual_norm = nn.LayerNorm(d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        semantic = self.residual_norm(values + self.semantic_mlp(values))
        mixed = self.channel_mlp(semantic.transpose(1, 2)).transpose(1, 2)
        return self.residual_norm(values + semantic * mixed)


class PatchMLPForecaster(nn.Module):
    """Information-matched continuous-forecast baseline for task-aware alarms."""

    supports_memory = False

    def __init__(self, config: PatchMLPForecasterConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = _MultiScalePatchEmbedding(config)
        self.decomposition = _SeriesDecomposition(config.moving_average_kernel)
        self.seasonal_layers = nn.ModuleList(
            [
                _PatchMLPEncoder(
                    d_model=config.d_model,
                    channels=config.input_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.trend_layers = nn.ModuleList(
            [
                _PatchMLPEncoder(
                    d_model=config.d_model,
                    channels=config.input_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.projector = nn.Linear(config.d_model, config.prediction_samples)

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        if sequence.ndim != 3:
            raise ValueError("PatchMLP input must have shape [batch, time, feature]")
        if sequence.shape[1:] != (
            self.config.context_samples,
            self.config.input_dim,
        ):
            raise ValueError(
                "PatchMLP input shape does not match the frozen context contract"
            )
        if not torch.isfinite(sequence).all():
            raise ValueError("PatchMLP input contains non-finite values")

        values = sequence
        if self.config.use_instance_normalization:
            mean = values.mean(dim=1, keepdim=True).detach()
            scale = torch.sqrt(
                values.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            values = (values - mean) / scale
        else:
            mean = torch.zeros_like(values[:, :1])
            scale = torch.ones_like(values[:, :1])

        embedded = self.embedding(values.transpose(1, 2))
        seasonal, trend = self.decomposition(embedded)
        for layer in self.seasonal_layers:
            seasonal = layer(seasonal)
        for layer in self.trend_layers:
            trend = layer(trend)
        all_forecasts = self.projector(seasonal + trend).transpose(1, 2)
        all_forecasts = all_forecasts * scale + mean
        cgm_forecast = all_forecasts[:, :, self.config.cgm_feature_index]
        return {
            "forecast": cgm_forecast,
            "all_channel_forecasts": all_forecasts,
        }


def glimmer_weighted_mae(
    forecast_mg_dl: torch.Tensor,
    target_mg_dl: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    hypoglycemia_threshold: float = 70.0,
    hyperglycemia_threshold: float = 180.0,
    normal_weight: float = 1.0,
    hypoglycemia_weight: float = 3.296363582,
    hyperglycemia_weight: float = 2.382397706,
) -> torch.Tensor:
    """GLIMMER's public region-weighted MAE with explicit masking."""

    if forecast_mg_dl.shape != target_mg_dl.shape:
        raise ValueError("forecast and target shapes must match")
    if target_mask.shape != target_mg_dl.shape:
        raise ValueError("target_mask shape must match the targets")
    if not target_mask.dtype == torch.bool:
        raise ValueError("target_mask must be boolean")
    if not target_mask.any():
        raise ValueError("GLIMMER loss requires at least one available target")
    if not torch.isfinite(forecast_mg_dl[target_mask]).all() or not torch.isfinite(
        target_mg_dl[target_mask]
    ).all():
        raise ValueError("GLIMMER loss inputs must be finite on available targets")
    if not (
        0 < normal_weight and 0 < hypoglycemia_weight and 0 < hyperglycemia_weight
    ):
        raise ValueError("GLIMMER region weights must be positive")

    weights = torch.full_like(target_mg_dl, float(normal_weight))
    weights = torch.where(
        target_mg_dl < hypoglycemia_threshold,
        torch.as_tensor(
            hypoglycemia_weight,
            device=weights.device,
            dtype=weights.dtype,
        ),
        weights,
    )
    weights = torch.where(
        target_mg_dl > hyperglycemia_threshold,
        torch.as_tensor(
            hyperglycemia_weight,
            device=weights.device,
            dtype=weights.dtype,
        ),
        weights,
    )
    errors = torch.abs(forecast_mg_dl - target_mg_dl) * weights
    return errors[target_mask].mean()


def glimmer_weighted_mae_per_example(
    forecast_mg_dl: torch.Tensor,
    target_mg_dl: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    hypoglycemia_threshold: float = 70.0,
    hyperglycemia_threshold: float = 180.0,
    normal_weight: float = 1.0,
    hypoglycemia_weight: float = 3.296363582,
    hyperglycemia_weight: float = 2.382397706,
) -> torch.Tensor:
    """Per-example form of the audited GLIMMER region-weighted MAE."""

    if forecast_mg_dl.ndim != 2:
        raise ValueError("forecast and target must have shape [batch, future_step]")
    if forecast_mg_dl.shape != target_mg_dl.shape:
        raise ValueError("forecast and target shapes must match")
    if target_mask.shape != target_mg_dl.shape or target_mask.dtype != torch.bool:
        raise ValueError("target_mask must be boolean and match the targets")
    available = target_mask.sum(dim=1)
    if (available == 0).any():
        raise ValueError("every GLIMMER example requires an available target")
    if not torch.isfinite(forecast_mg_dl[target_mask]).all() or not torch.isfinite(
        target_mg_dl[target_mask]
    ).all():
        raise ValueError("GLIMMER loss inputs must be finite on available targets")
    if not (
        0 < normal_weight and 0 < hypoglycemia_weight and 0 < hyperglycemia_weight
    ):
        raise ValueError("GLIMMER region weights must be positive")

    weights = torch.full_like(target_mg_dl, float(normal_weight))
    weights = torch.where(
        target_mg_dl < hypoglycemia_threshold,
        torch.as_tensor(
            hypoglycemia_weight,
            device=weights.device,
            dtype=weights.dtype,
        ),
        weights,
    )
    weights = torch.where(
        target_mg_dl > hyperglycemia_threshold,
        torch.as_tensor(
            hyperglycemia_weight,
            device=weights.device,
            dtype=weights.dtype,
        ),
        weights,
    )
    errors = torch.abs(forecast_mg_dl - target_mg_dl) * weights
    errors = torch.where(target_mask, errors, torch.zeros_like(errors))
    return errors.sum(dim=1) / available.to(errors.dtype)


def inverse_standardized_cgm(
    forecast: torch.Tensor,
    *,
    mean_mg_dl: float,
    std_mg_dl: float,
) -> torch.Tensor:
    """Convert SequenceScaler CGM units to mg/dL before clinical thresholds."""

    if not math.isfinite(mean_mg_dl) or not math.isfinite(std_mg_dl):
        raise ValueError("CGM scaler values must be finite")
    if std_mg_dl <= 0:
        raise ValueError("CGM scaler standard deviation must be positive")
    if not torch.isfinite(forecast).all():
        raise ValueError("standardized CGM forecast contains non-finite values")
    return forecast * float(std_mg_dl) + float(mean_mg_dl)


def forecast_task_probabilities(
    forecast_mg_dl: torch.Tensor,
    *,
    sample_minutes: int = 5,
    hypoglycemia_threshold: float = 70.0,
    hyperglycemia_threshold: float = 180.0,
    temperature_mg_dl: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Map a continuous forecast path to monotone task-risk scores.

    The logistic map is a calibration adapter, not a claim that a deterministic
    forecast is probabilistic. Its fixed temperature preserves the ranking of
    the minimum/maximum forecast and allows the frozen alarm evaluator to use
    the same score interface for every arm.
    """

    if forecast_mg_dl.ndim != 2:
        raise ValueError("forecast must have shape [batch, future_step]")
    if sample_minutes <= 0 or temperature_mg_dl <= 0:
        raise ValueError("sample_minutes and temperature_mg_dl must be positive")
    if not torch.isfinite(forecast_mg_dl).all():
        raise ValueError("forecast contains non-finite values")
    output: dict[str, torch.Tensor] = {}
    for task in TASKS:
        kind, raw_horizon = task.split("_")
        steps = int(raw_horizon) // sample_minutes
        if steps <= 0 or steps > forecast_mg_dl.shape[1]:
            raise ValueError(f"forecast does not cover the {raw_horizon}-minute task")
        values = forecast_mg_dl[:, :steps]
        if kind == "hypo":
            margin = hypoglycemia_threshold - values.min(dim=1).values
        elif kind == "hyper":
            margin = values.max(dim=1).values - hyperglycemia_threshold
        else:
            raise ValueError(f"unsupported task kind: {kind}")
        output[task] = torch.sigmoid(margin / temperature_mg_dl)
    return output
