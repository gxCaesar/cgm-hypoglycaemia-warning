from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from .schema import ACTION_FIELDS, validate_frame

METABONET_ALIASES: dict[str, tuple[str, ...]] = {
    "participant_id": ("id", "participant_id", "subject_id"),
    "timestamp": ("date", "timestamp", "datetime"),
    "cgm_mg_dl": ("CGM", "cgm_mg_dl", "glucose"),
    "basal_insulin_u": ("basal", "basal_insulin_u"),
    "bolus_insulin_u": ("bolus", "bolus_insulin_u"),
    "carbs_g": ("carbs", "carbs_g", "carbohydrates"),
    "activity_value": ("steps", "activity_value", "workout_duration"),
}


@dataclass(frozen=True)
class SourceRule:
    source_id: str
    value: str | None = None
    pattern: str | None = None

    def matches(self, raw_value: str) -> bool:
        if self.value is not None:
            return raw_value == self.value
        if self.pattern is not None:
            return re.fullmatch(self.pattern, raw_value) is not None
        return False


@dataclass(frozen=True)
class SourceMap:
    source_column: str
    rules: tuple[SourceRule, ...]

    def map_value(self, raw_value: object) -> str:
        value = str(raw_value)
        matches = [rule.source_id for rule in self.rules if rule.matches(value)]
        if len(matches) != 1:
            raise ValueError(
                f"source value {value!r} matched {len(matches)} reviewed rules"
            )
        return matches[0]


def load_source_map(path: str | Path) -> SourceMap:
    mapping_path = Path(path).expanduser().resolve()
    with mapping_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("source map must be a YAML mapping")
    source_column = str(payload.get("source_column", "source_file"))
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("source map requires a non-empty rules list")
    rules: list[SourceRule] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("each source rule must be a mapping")
        source_id = str(raw_rule["source_id"])
        value = raw_rule.get("value")
        pattern = raw_rule.get("pattern")
        if (value is None) == (pattern is None):
            raise ValueError("each source rule needs exactly one value or pattern")
        rules.append(
            SourceRule(
                source_id=source_id,
                value=None if value is None else str(value),
                pattern=None if pattern is None else str(pattern),
            )
        )
    return SourceMap(source_column=source_column, rules=tuple(rules))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_column(frame: pd.DataFrame, canonical: str) -> str | None:
    for candidate in METABONET_ALIASES[canonical]:
        if candidate in frame.columns:
            return candidate
    return None


def _interpolate_short_runs(
    values: pd.Series,
    *,
    limit_samples: int,
    blocked: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = numeric.copy()
    interpolated = pd.Series(False, index=values.index, dtype=bool)
    blocked_mask = (
        pd.Series(False, index=values.index, dtype=bool)
        if blocked is None
        else blocked.reindex(values.index, fill_value=False).astype(bool)
    )
    missing = numeric.isna().to_numpy()
    padded = np.concatenate(([False], missing, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    raw = numeric.to_numpy(dtype=float)
    for start, end in zip(starts, ends):  # noqa: B905
        length = int(end - start + 1)
        if (
            length > limit_samples
            or blocked_mask.iloc[start : end + 1].any()
            or start == 0
            or end == len(raw) - 1
            or not np.isfinite(raw[start - 1])
            or not np.isfinite(raw[end + 1])
        ):
            continue
        positions = np.arange(start, end + 1)
        filled = np.interp(
            positions,
            np.array([start - 1, end + 1]),
            np.array([raw[start - 1], raw[end + 1]]),
        )
        output.iloc[positions] = filled
        interpolated.iloc[positions] = True
    return output, interpolated


def _normalize_timestamps(
    raw_timestamp: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    parsed = pd.to_datetime(raw_timestamp, errors="coerce")
    if parsed.isna().any():
        raise ValueError("timestamp column contains unparseable values")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        utc = parsed.dt.tz_convert("UTC")
        local = parsed.dt.tz_localize(None)
        known = pd.Series(True, index=raw_timestamp.index, dtype=bool)
        return utc, local, utc, known
    local = parsed
    timeline = pd.to_datetime(parsed, utc=True)
    utc = pd.Series(pd.NaT, index=raw_timestamp.index, dtype="datetime64[ns, UTC]")
    known = pd.Series(False, index=raw_timestamp.index, dtype=bool)
    return timeline, local, utc, known


def normalize_metabonet(
    raw: pd.DataFrame,
    *,
    source_map: SourceMap,
    cgm_unit: str = "mg/dL",
    interpolation_limit_minutes: int = 15,
    sample_minutes: int = 5,
    action_availability: Mapping[str, Mapping[str, bool]] | None = None,
    glucose_range_policy: Literal[
        "fail",
        "exclude_noninterpolable",
        "left_censor",
    ] = "fail",
    glucose_minimum_mg_dl: float = 20.0,
    glucose_maximum_mg_dl: float = 600.0,
    lower_censor_proxy_mg_dl: float = 39.0,
) -> pd.DataFrame:
    if source_map.source_column not in raw:
        raise ValueError(
            f"reviewed source column is absent: {source_map.source_column}"
        )
    resolved = {
        canonical: _resolve_column(raw, canonical)
        for canonical in METABONET_ALIASES
    }
    missing_core = [
        name
        for name in ("participant_id", "timestamp", "cgm_mg_dl")
        if resolved[name] is None
    ]
    if missing_core:
        raise ValueError(f"MetaboNet input is missing core columns: {missing_core}")

    source_id = raw[source_map.source_column].map(source_map.map_value)
    participant_column = str(resolved["participant_id"])
    timestamp_column = str(resolved["timestamp"])
    cgm_column = str(resolved["cgm_mg_dl"])
    timeline, timestamp_local, timestamp_utc, timezone_known = _normalize_timestamps(
        raw[timestamp_column]
    )
    cgm = pd.to_numeric(raw[cgm_column], errors="coerce")
    if cgm_unit == "mmol/L":
        cgm = cgm * 18.0182
    elif cgm_unit != "mg/dL":
        raise ValueError("cgm_unit must be explicitly mg/dL or mmol/L")
    if glucose_minimum_mg_dl >= glucose_maximum_mg_dl:
        raise ValueError("glucose range minimum must be below maximum")
    if not glucose_minimum_mg_dl <= lower_censor_proxy_mg_dl <= glucose_maximum_mg_dl:
        raise ValueError("lower-censor proxy must be inside the glucose range")
    if glucose_range_policy not in {
        "fail",
        "exclude_noninterpolable",
        "left_censor",
    }:
        raise ValueError(f"unsupported glucose range policy: {glucose_range_policy}")
    invalid_low = cgm.notna() & cgm.lt(glucose_minimum_mg_dl)
    invalid_high = cgm.notna() & cgm.gt(glucose_maximum_mg_dl)
    invalid_range = invalid_low | invalid_high
    if invalid_range.any() and glucose_range_policy == "fail":
        raise ValueError(
            "cgm_mg_dl contains "
            f"{int(invalid_range.sum())} values outside "
            f"{glucose_minimum_mg_dl:g}–{glucose_maximum_mg_dl:g}"
        )
    if invalid_high.any() and glucose_range_policy == "left_censor":
        raise ValueError("left-censor policy cannot represent high-range violations")
    lower_censored = pd.Series(False, index=raw.index, dtype=bool)
    blocked_from_interpolation = pd.Series(False, index=raw.index, dtype=bool)
    if glucose_range_policy == "exclude_noninterpolable":
        cgm = cgm.mask(invalid_range)
        blocked_from_interpolation = invalid_range.astype(bool)
    elif glucose_range_policy == "left_censor":
        cgm = cgm.mask(invalid_low, lower_censor_proxy_mg_dl)
        lower_censored = invalid_low.astype(bool)

    normalized = pd.DataFrame(
        {
            "source_id": source_id.astype(str),
            "participant_id": raw[participant_column].astype(str),
            "timestamp": timeline,
            "timestamp_local": timestamp_local,
            "timestamp_utc": timestamp_utc,
            "timezone_known": timezone_known,
            "cgm_mg_dl": cgm,
            "cgm_observed": cgm.notna(),
            "cgm_interpolated": False,
            "cgm_lower_censored": lower_censored,
            "_cgm_interpolation_blocked": blocked_from_interpolation,
        }
    )

    limit_samples = interpolation_limit_minutes // sample_minutes
    grouping = normalized.groupby(["source_id", "participant_id"], sort=False)
    for _, index in grouping.groups.items():
        ordered_index = normalized.loc[index].sort_values("timestamp").index
        filled, interpolated = _interpolate_short_runs(
            normalized.loc[ordered_index, "cgm_mg_dl"],
            limit_samples=limit_samples,
            blocked=normalized.loc[
                ordered_index,
                "_cgm_interpolation_blocked",
            ],
        )
        normalized.loc[ordered_index, "cgm_mg_dl"] = filled
        normalized.loc[ordered_index, "cgm_interpolated"] = interpolated

    for field in ACTION_FIELDS:
        raw_column = resolved[field]
        if raw_column is None:
            values = pd.Series(np.nan, index=raw.index, dtype=float)
            available = pd.Series(False, index=raw.index, dtype=bool)
        else:
            values = pd.to_numeric(raw[raw_column], errors="coerce")
            if action_availability is None:
                by_source = values.groupby(normalized["source_id"]).transform(
                    lambda column: bool(column.notna().any())
                )
                available = by_source.astype(bool)
            else:
                missing_sources = sorted(
                    set(normalized["source_id"])
                    .difference(action_availability)
                )
                if missing_sources:
                    raise ValueError(
                        "action availability is missing sources: "
                        f"{missing_sources}"
                    )
                available = normalized["source_id"].map(
                    lambda source, field=field: bool(
                        action_availability[str(source)].get(field, False)
                    )
                )
        observed = values.notna() & available
        values = values.where(observed)
        normalized[field] = values
        normalized[f"{field}_available"] = available
        normalized[f"{field}_observed"] = observed
        normalized[f"{field}_zero_ambiguous"] = observed & values.eq(0)

    normalized = normalized.loc[normalized["cgm_mg_dl"].notna()].copy()
    normalized = normalized.drop(columns="_cgm_interpolation_blocked")
    normalized = normalized.sort_values(
        ["source_id", "participant_id", "timestamp"]
    ).reset_index(drop=True)
    validate_frame(normalized)
    return normalized


def normalize_hupa(
    raw: pd.DataFrame,
    *,
    participant_id: str,
) -> pd.DataFrame:
    required = {
        "time",
        "glucose",
        "steps",
        "basal_rate",
        "bolus_volume_delivered",
        "carb_input",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"HUPA preprocessed CSV is missing columns: {missing}")
    prepared = pd.DataFrame(
        {
            "__hupa_source__": "HUPA-UCM",
            "id": str(participant_id),
            "date": raw["time"],
            "CGM": pd.to_numeric(raw["glucose"], errors="coerce"),
            "basal": pd.to_numeric(raw["basal_rate"], errors="coerce"),
            "bolus": pd.to_numeric(
                raw["bolus_volume_delivered"],
                errors="coerce",
            ),
            # The direct release records carbohydrate servings, where one
            # serving is ten grams.
            "carbs": (
                pd.to_numeric(raw["carb_input"], errors="coerce") * 10.0
            ),
            "steps": pd.to_numeric(raw["steps"], errors="coerce"),
        }
    )
    source_map = SourceMap(
        source_column="__hupa_source__",
        rules=(SourceRule(source_id="HUPA-UCM", pattern=".*"),),
    )
    normalized = normalize_metabonet(
        prepared,
        source_map=source_map,
        cgm_unit="mg/dL",
        interpolation_limit_minutes=0,
        sample_minutes=5,
        action_availability={
            "HUPA-UCM": {
                "basal_insulin_u": True,
                "bolus_insulin_u": True,
                "carbs_g": True,
                "activity_value": True,
            }
        },
    )
    # The source paper states that insulin gaps were filled with zero and
    # carbohydrate records were aggregated into zero-filled five-minute
    # intervals. Those zeros cannot be treated as verified no-action events.
    normalized["activity_value_zero_ambiguous"] = False
    normalized["cgm_preprocessed_by_source"] = True
    return normalized


def read_table(path: str | Path) -> pd.DataFrame:
    input_path = Path(path).expanduser().resolve()
    suffix = input_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(input_path)
    raise ValueError(f"unsupported input table: {input_path}")


def write_partitioned_frame(frame: pd.DataFrame, output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        output,
        index=False,
        partition_cols=["source_id"],
    )
    return output


def source_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    payload: dict[str, dict[str, int]] = {}
    for source, source_frame in frame.groupby("source_id", sort=True):
        payload[str(source)] = {
            "rows": int(len(source_frame)),
            "participants": int(source_frame["participant_id"].nunique()),
        }
    return payload


def manifest_payload(
    *,
    input_path: str | Path,
    frame: pd.DataFrame,
    release_id: str,
    source_map_path: str | Path,
) -> dict[str, Any]:
    return {
        "release_id": release_id,
        "input_file": str(Path(input_path).name),
        "sha256": sha256_file(input_path),
        "source_map": str(Path(source_map_path).name),
        "columns": list(frame.columns),
        "sources": source_counts(frame),
    }
