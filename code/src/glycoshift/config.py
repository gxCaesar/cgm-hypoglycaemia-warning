from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {
        "project",
        "data",
        "protocol",
        "split",
        "model",
        "evaluation",
        "output",
    }
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"missing configuration sections: {sorted(missing)}")

    sample_minutes = int(config["protocol"]["sample_minutes"])
    if sample_minutes <= 0:
        raise ValueError("sample_minutes must be positive")
    horizons = [int(value) for value in config["protocol"]["horizons_minutes"]]
    if not horizons or any(value <= 0 or value % sample_minutes for value in horizons):
        raise ValueError("all horizons must be positive multiples of sample_minutes")

    development = set(config["data"].get("development_sources", []))
    forbidden = set(config["data"].get("forbidden_development_sources", []))
    overlap = development.intersection(forbidden)
    if overlap:
        raise ValueError(
            "held-out sources cannot be development sources: "
            f"{sorted(overlap)}"
        )

    synthetic = bool(config["project"].get("synthetic", False))
    glucose_range = config["data"].get("glucose_range")
    if not synthetic and not isinstance(glucose_range, dict):
        raise ValueError("real-data config requires data.glucose_range")
    if isinstance(glucose_range, dict):
        policy = str(glucose_range.get("policy", "fail"))
        if policy not in {
            "fail",
            "exclude_noninterpolable",
            "left_censor",
        }:
            raise ValueError("unsupported data.glucose_range.policy")
        minimum = float(glucose_range.get("minimum_mg_dl", 20))
        maximum = float(glucose_range.get("maximum_mg_dl", 600))
        proxy = float(glucose_range.get("lower_censor_proxy_mg_dl", 39))
        if minimum >= maximum:
            raise ValueError("glucose range minimum must be below maximum")
        if not minimum <= proxy <= maximum:
            raise ValueError(
                "lower-censor proxy must be inside the glucose range"
            )
        maximum_rows = int(glucose_range.get("maximum_rows", 0))
        if maximum_rows < 0:
            raise ValueError("glucose range maximum_rows cannot be negative")
        if policy != "fail":
            allowed_sources = set(
                map(str, glucose_range.get("allowed_sources", []))
            )
            if not allowed_sources or not allowed_sources.issubset(development):
                raise ValueError(
                    "approved glucose-range sources must be development sources"
                )
            approved_values = [
                float(value)
                for value in glucose_range.get(
                    "approved_values_mg_dl",
                    [],
                )
            ]
            if (
                maximum_rows <= 0
                or not approved_values
                or any(value >= minimum for value in approved_values)
            ):
                raise ValueError(
                    "approved lower-range exception requires a positive row "
                    "limit and values below the minimum"
                )
            release_id = str(
                glucose_range.get("approved_release_id", "")
            )
            release_hash = str(
                glucose_range.get("approved_release_sha256", "")
            ).lower()
            invalid_hash_character = any(
                character not in "0123456789abcdef"
                for character in release_hash
            )
            if (
                not release_id
                or len(release_hash) != 64
                or invalid_hash_character
            ):
                raise ValueError(
                    "approved glucose-range exception requires release ID "
                    "and SHA-256"
                )

    split = config["split"]
    if "validation_fraction" in split:
        validation_fraction = float(split["validation_fraction"])
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
    else:
        fractions = {
            name: float(split[name])
            for name in (
                "train_fraction",
                "selection_fraction",
                "calibration_fraction",
            )
        }
        if any(value <= 0 or value >= 1 for value in fractions.values()):
            raise ValueError("all split fractions must be between zero and one")
        if abs(sum(fractions.values()) - 1.0) > 1e-9:
            raise ValueError("train/selection/calibration fractions must sum to one")

    external_source = config["data"].get("external_source")
    stress_sources = set(config["data"].get("stress_sources", []))
    if external_source in stress_sources:
        raise ValueError("external source cannot also be a stress source")
    if not synthetic:
        evaluation_inputs = config["data"].get("evaluation_inputs")
        if not isinstance(evaluation_inputs, dict):
            raise ValueError(
                "real-data config requires isolated data.evaluation_inputs"
            )
        required_partitions = {
            "external_loop": (
                str(external_source),
                "metabonet-public",
            ),
        }
        for source in stress_sources:
            partition = (
                f"stress_{str(source).lower().replace('-', '_').replace(' ', '_')}"
            )
            required_partitions[partition] = (
                str(source),
                "mendeley-direct",
            )
        development_input = str(config["data"].get("input_path", ""))
        development_manifest = str(
            config["data"].get("source_manifest", "")
        )
        for partition, (
            expected_source,
            required_origin,
        ) in required_partitions.items():
            specification = evaluation_inputs.get(partition)
            if not isinstance(specification, dict):
                raise ValueError(
                    f"real-data config is missing evaluation input: {partition}"
                )
            missing_fields = {
                "input_path",
                "source_manifest",
                "expected_source",
                "required_origin",
            }.difference(specification)
            if missing_fields:
                raise ValueError(
                    f"{partition} evaluation input is missing fields: "
                    f"{sorted(missing_fields)}"
                )
            if str(specification["expected_source"]) != expected_source:
                raise ValueError(
                    f"{partition} expected_source conflicts with protocol"
                )
            if str(specification["required_origin"]) != required_origin:
                raise ValueError(
                    f"{partition} required_origin conflicts with protocol"
                )
            if str(specification["input_path"]) == development_input:
                raise ValueError(
                    f"{partition} must not share the development input path"
                )
            if str(specification["source_manifest"]) == development_manifest:
                raise ValueError(
                    f"{partition} must not share the development manifest"
                )

    model = config["model"]
    if "variant" in model and model["variant"] not in {"tiny", "base"}:
        raise ValueError("model.variant must be tiny or base")
    forecast = model.get("taskaware_forecast")
    if forecast is not None:
        if not isinstance(forecast, dict):
            raise ValueError("model.taskaware_forecast must be a mapping")
        positive_integer_fields = (
            "d_model",
            "encoder_layers",
            "moving_average_kernel",
        )
        for field in positive_integer_fields:
            if field in forecast and int(forecast[field]) <= 0:
                raise ValueError(f"model.taskaware_forecast.{field} must be positive")
        if "moving_average_kernel" in forecast and int(
            forecast["moving_average_kernel"]
        ) % 2 == 0:
            raise ValueError(
                "model.taskaware_forecast.moving_average_kernel must be odd"
            )
        if "patch_lengths" in forecast:
            patch_lengths = forecast["patch_lengths"]
            if not isinstance(patch_lengths, list) or not patch_lengths:
                raise ValueError(
                    "model.taskaware_forecast.patch_lengths must be a nonempty list"
                )
            if any(int(value) <= 1 for value in patch_lengths):
                raise ValueError(
                    "model.taskaware_forecast.patch_lengths must exceed one"
                )

    negative_ratio = float(config.get("training", {}).get("negative_ratio", 20))
    if negative_ratio <= 0:
        raise ValueError("training.negative_ratio must be positive")
    training = config.get("training", {})
    event_objective = str(training.get("event_objective", "bce"))
    if event_objective not in {
        "bce",
        "tls_exp",
        "severe_retarget",
        "lead_time_weighted",
    }:
        raise ValueError(
            "training.event_objective must be bce, tls_exp, severe_retarget, "
            "or lead_time_weighted"
        )
    if event_objective == "tls_exp":
        gamma_per_hour = float(
            training.get("tls_gamma_per_hour", float("nan"))
        )
        if not math.isfinite(gamma_per_hour) or gamma_per_hour <= 0:
            raise ValueError(
                "TLS requires a positive training.tls_gamma_per_hour"
            )
    rule = str(training.get("checkpoint_selection_rule", "area"))
    if rule not in {"area", "budget"}:
        raise ValueError(
            "training.checkpoint_selection_rule must be area or budget"
        )
    budget = float(training.get("checkpoint_budget", 0.5))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("training.checkpoint_budget must be positive and finite")
    positive_weight_mode = str(training.get("positive_weight_mode", "prevalence"))
    if positive_weight_mode not in {"prevalence", "uniform"}:
        raise ValueError(
            "training.positive_weight_mode must be prevalence or uniform"
        )
    temporal_memory_k = int((config.get("model") or {}).get("temporal_memory_k", 0))
    if temporal_memory_k < 0:
        raise ValueError("model.temporal_memory_k must be non-negative")
    if "anchor_weight" in training:
        anchor_weight = float(training["anchor_weight"])
        if not math.isfinite(anchor_weight) or not 0.0 < anchor_weight <= 1.0:
            raise ValueError(
                "training.anchor_weight must lie in (0, 1]; omit the key for an "
                "un-anchored arm rather than setting it to zero, so the control "
                "runs the code path it has always run"
            )
    data_sampling_seed = int(
        training.get("data_sampling_seed", training.get("seed", 0))
    )
    if data_sampling_seed < 0:
        raise ValueError("training.data_sampling_seed must be non-negative")


def resolve_path(config: dict[str, Any], value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_dir = Path(config["_config_path"]).parent
    return (config_dir / path).resolve()


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {
        key: value
        for key, value in config.items()
        if key != "_config_path"
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)
