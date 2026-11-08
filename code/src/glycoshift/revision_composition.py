"""Fixed-pair step-ECDF composition and development-only mapping diagnostics.

Member references use every context-valid development calibration decision,
including retrospective unevaluable tails. Selection and external scores never
fit a reference or policy. The arithmetic mean of two ECDF ranks is a score,
not a calibrated event probability.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .calibration import apply_temperature
from .config import load_config
from .fixed_frame_evaluation import (
    FixedFrameEvaluation,
    FixedFrameSpec,
    _online_eligible,
    evaluate_fixed_frame,
    fit_fixed_frame_alarm_policy,
)
from .revision_inputs import ReplayInputs
from .revision_runner import load_replay_bundle
from .revision_training import RevisionTrainingContext
from .sequences import REVISION_CONTRACT

MEMBERS = ("lstm", "persistence")
HORIZONS = (30, 60)
MODEL_SEEDS = (17, 23, 42, 71, 101)
_KEYS = ["source_id", "participant_id", "context_end"]
_PARTITIONS = ("calibration", "selection")


@dataclass(frozen=True)
class FrozenStepECDF:
    """Lossless compressed form of sorted calibration scores with duplicates."""

    knots: tuple[float, ...]
    cumulative_counts: tuple[int, ...]
    sample_count: int

    def __post_init__(self) -> None:
        knots = np.asarray(self.knots, dtype=np.float64)
        counts = np.asarray(self.cumulative_counts)
        if (
            knots.ndim != 1
            or knots.size == 0
            or not np.isfinite(knots).all()
            or np.any(knots[1:] <= knots[:-1])
            or counts.ndim != 1
            or len(counts) != len(knots)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts <= 0)
            or np.any(counts[1:] <= counts[:-1])
            or isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or int(counts[-1]) != self.sample_count
            or self.sample_count > np.iinfo(np.int64).max
        ):
            raise ValueError("invalid frozen ECDF knots/counts/sample_count")
        object.__setattr__(self, "knots", tuple(map(float, knots)))
        object.__setattr__(self, "cumulative_counts", tuple(map(int, counts)))

    @classmethod
    def fit(cls, values: np.ndarray, *, partition: str) -> FrozenStepECDF:
        if partition != "calibration":
            raise ValueError("member ECDF must be fit on calibration only")
        scores = np.asarray(values, dtype=np.float64)
        if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
            raise ValueError("ECDF fitting requires complete finite calibration scores")
        knots, counts = np.unique(scores, return_counts=True)
        return cls(
            tuple(map(float, knots)), tuple(map(int, counts.cumsum())), len(scores)
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        scores = np.asarray(values, dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("ECDF scoring requires finite member scores")
        positions = np.searchsorted(self.knots, scores, side="right")
        cumulative = np.r_[0, np.asarray(self.cumulative_counts, dtype=np.int64)]
        return cumulative[positions] / self.sample_count

    def legacy_interpolated(self, values: np.ndarray) -> np.ndarray:
        """Execute the original np.interp formula exactly, diagnostic use only."""
        scores = np.asarray(values, dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("legacy mapping diagnostic requires finite scores")
        counts = np.diff(np.r_[0, self.cumulative_counts])
        reference = np.repeat(np.asarray(self.knots, dtype=np.float64), counts)
        return np.interp(scores, reference, np.linspace(0.0, 1.0, self.sample_count))

    def inline(self, *, max_knots: int = 10_000) -> dict[str, Any]:
        if len(self.knots) > max_knots:
            raise ValueError("large ECDF references require compressed remote sidecars")
        return {
            "knots": list(self.knots),
            "cumulative_counts": list(self.cumulative_counts),
            "sample_count": self.sample_count,
            "dtype": "float64",
            "count_dtype": "int64",
            "rule": "searchsorted_right_cumulative_count_divided_by_N",
        }


@dataclass(frozen=True)
class FrozenRevisionComposition:
    lstm_seed: int
    source_order: tuple[str, ...]
    mappings: Mapping[tuple[str, int], FrozenStepECDF]
    policies: Mapping[int, Mapping[str, Any]]
    member_metadata: Mapping[str, Mapping[str, Any]]

    def inline(self, *, max_knots: int = 10_000) -> dict[str, Any]:
        return _recipe(
            self,
            {
                str(horizon): {
                    member: self.mappings[member, horizon].inline(max_knots=max_knots)
                    for member in MEMBERS
                }
                for horizon in HORIZONS
            },
        )


@dataclass(frozen=True)
class RevisionCompositionResult:
    frozen: FrozenRevisionComposition
    predictions: pd.DataFrame
    evaluations: Mapping[tuple[str, int], FixedFrameEvaluation]
    mapping_audit: Mapping[str, Any]


def _recipe(
    frozen: FrozenRevisionComposition, references: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "revision_contract": REVISION_CONTRACT,
        "algebra": "arithmetic_mean_of_two_right_step_member_ECDF_ranks",
        "score_interpretation": "rank_mean_not_calibrated_event_probability",
        "members": list(MEMBERS),
        "comparison_lstm_seed": frozen.lstm_seed,
        "source_order": list(frozen.source_order),
        "horizons": list(HORIZONS),
        "reference_partition": "calibration",
        "reference_population": "all_context_valid_causal_calibration_decisions",
        "glucose_cut": 54,
        "nominal_fapd": 0.5,
        "member_metadata": copy.deepcopy(dict(frozen.member_metadata)),
        "references": references,
        "policies": {str(h): dict(frozen.policies[h]) for h in HORIZONS},
    }


def save_frozen_composition(
    frozen: FrozenRevisionComposition, directory: str | Path
) -> Path:
    """Write small JSON plus lossless compressed NPZ references in a new directory."""
    root = Path(directory)
    if root.is_symlink() or root.exists():
        raise FileExistsError(
            "frozen composition export requires a new private directory"
        )
    root.mkdir(mode=0o700)
    references = {}
    for horizon in HORIZONS:
        references[str(horizon)] = {}
        for member in MEMBERS:
            mapping = frozen.mappings[member, horizon]
            path = root / f"{member}-{horizon}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(
                    handle,
                    knots=np.asarray(mapping.knots, dtype=np.float64),
                    cumulative_counts=np.asarray(
                        mapping.cumulative_counts, dtype=np.int64
                    ),
                    sample_count=np.asarray([mapping.sample_count], dtype=np.int64),
                )
            path.chmod(0o600)
            stat = path.stat()
            references[str(horizon)][member] = {
                "format": "compressed_npz",
                "path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "dtype": "float64",
                "count_dtype": "int64",
                "knot_count": len(mapping.knots),
                "sample_count": mapping.sample_count,
                "rule": "searchsorted_right_cumulative_count_divided_by_N",
            }
    destination = root / "frozen-composition.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(_recipe(frozen, references), handle, indent=2, allow_nan=False)
        handle.write("\n")
    destination.chmod(0o600)
    return destination


def load_frozen_composition(path: str | Path) -> FrozenRevisionComposition:
    payload = json.loads(Path(path).read_text())
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("revision_contract") != REVISION_CONTRACT
        or payload.get("algebra")
        != "arithmetic_mean_of_two_right_step_member_ECDF_ranks"
        or tuple(payload.get("members", ())) != MEMBERS
        or tuple(payload.get("horizons", ())) != HORIZONS
        or payload.get("reference_partition") != "calibration"
        or payload.get("reference_population")
        != "all_context_valid_causal_calibration_decisions"
        or payload.get("glucose_cut") != 54
        or payload.get("nominal_fapd") != 0.5
    ):
        raise ValueError(
            "frozen composition recipe differs from the prospective contract"
        )
    mappings = {}
    for horizon in HORIZONS:
        for member in MEMBERS:
            reference = payload["references"][str(horizon)][member]
            if (
                reference.get("dtype") != "float64"
                or reference.get("count_dtype") != "int64"
                or reference.get("rule")
                != "searchsorted_right_cumulative_count_divided_by_N"
            ):
                raise ValueError("frozen member mapping type/rule differs")
            if "knots" in reference:
                mappings[member, horizon] = FrozenStepECDF(
                    tuple(reference["knots"]),
                    tuple(reference["cumulative_counts"]),
                    reference["sample_count"],
                )
                continue
            artifact = Path(reference["path"])
            if (
                reference.get("format") != "compressed_npz"
                or not artifact.is_absolute()
                or artifact.suffix != ".npz"
                or artifact.parent.resolve() != Path(path).resolve().parent
                or artifact.is_symlink()
                or not artifact.is_file()
            ):
                raise ValueError(
                    "frozen member reference is not a regular retained sidecar"
                )
            stat = artifact.stat()
            if (
                stat.st_size != reference["size_bytes"]
                or stat.st_mtime_ns != reference["mtime_ns"]
            ):
                raise ValueError("frozen member reference size/mtime binding differs")
            with np.load(artifact, allow_pickle=False) as arrays:
                if set(arrays.files) != {"knots", "cumulative_counts", "sample_count"}:
                    raise ValueError(
                        "frozen member reference has an unexpected array schema"
                    )
                knots, counts, total = (
                    arrays[name]
                    for name in ("knots", "cumulative_counts", "sample_count")
                )
                if (
                    knots.dtype != np.float64
                    or counts.dtype != np.int64
                    or (total.dtype != np.int64 or total.shape != (1,))
                ):
                    raise ValueError(
                        "frozen member reference precision/count dtype differs"
                    )
                mapping = FrozenStepECDF(
                    tuple(map(float, knots)), tuple(map(int, counts)), int(total[0])
                )
            if (
                len(mapping.knots) != reference["knot_count"]
                or mapping.sample_count != reference["sample_count"]
            ):
                raise ValueError("frozen member reference count binding differs")
            mappings[member, horizon] = mapping
    if len({mapping.sample_count for mapping in mappings.values()}) != 1:
        raise ValueError(
            "frozen member references do not share the complete calibration population"
        )
    seed = payload["comparison_lstm_seed"]
    if seed not in MODEL_SEEDS:
        raise ValueError("frozen composition uses an unregistered LSTM seed")
    sources = tuple(payload["source_order"])
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("frozen composition source order is invalid")
    policies = {h: payload["policies"][str(h)] for h in HORIZONS}
    for horizon, policy in policies.items():
        endpoint = policy["endpoint"]
        if (
            policy["partition"] != "calibration"
            or policy["glucose_cut"] != 54
            or policy["nominal_fapd"] != 0.5
            or endpoint["horizon_minutes"] != horizon
            or endpoint["score_column"] != f"composition_{horizon}_score"
            or tuple(endpoint["source_order"]) != sources
            or not np.isfinite(policy["threshold"])
        ):
            raise ValueError(
                "frozen composition policy differs from its declared endpoint"
            )
    return FrozenRevisionComposition(
        seed, sources, mappings, policies, payload["member_metadata"]
    )


def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    if not {*_KEYS, "partition"}.issubset(frame):
        raise ValueError("composition member lacks exact decision keys/partition")
    work = frame.copy(deep=False)
    for column in _KEYS[:2]:
        if work[column].isna().any():
            raise ValueError(
                "composition member has missing participant/source identity"
            )
        work[column] = work[column].astype(str)
    work["context_end"] = pd.to_datetime(work["context_end"], utc=True, errors="coerce")
    if work["context_end"].isna().any() or work.duplicated(_KEYS).any():
        raise ValueError("composition member has invalid or duplicate decision keys")
    if work["partition"].isna().any():
        raise ValueError("composition member has a missing partition")
    return work.set_index(_KEYS).sort_index()


def join_member_scores(members: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Exact complete join; never take an inner intersection or average positions."""
    if set(members) != set(MEMBERS):
        raise ValueError(
            "composition requires exactly the declared LSTM/persistence pair"
        )
    left, right = (_indexed(members[name]) for name in MEMBERS)
    if not left.index.equals(right.index):
        raise ValueError("composition member decision populations differ")
    bindings = {
        "partition",
        "patient_day",
        "cgm_last",
        "anchor_cgm_observed",
        "anchor_cgm_interpolated",
        "anchor_cgm_censored",
        *(f"base_evaluable_{h}" for h in HORIZONS),
        *(
            f"severe_hypo_{h}_{suffix}"
            for h in HORIZONS
            for suffix in ("label", "event_id", "event_onset")
        ),
    }
    for column in bindings & (set(left) | set(right)):
        if column not in left or column not in right:
            raise ValueError(f"composition member truth column is missing: {column}")
        left_values, right_values = left[column], right[column]
        if isinstance(left_values.dtype, pd.CategoricalDtype):
            left_values = left_values.astype(object)
        if isinstance(right_values.dtype, pd.CategoricalDtype):
            right_values = right_values.astype(object)
        equal = left_values.eq(right_values) | (
            left_values.isna() & right_values.isna()
        )
        if not equal.fillna(False).all():
            raise ValueError(f"composition member truth differs: {column}")
    # A rank mean has no inherited member probability or regression head. Keep
    # scores only under explicit member names so a legacy evaluator cannot read
    # one member's probability as the composed score by accident.
    inherited_heads = [
        column
        for column in left
        if column.endswith(
            ("_probability", "_logit", "_prediction", "_prediction_scaled")
        )
    ]
    combined = left.drop(columns=inherited_heads).reset_index()
    for member, frame in zip(MEMBERS, (left, right), strict=True):
        for horizon in HORIZONS:
            column = f"severe_hypo_{horizon}_probability"
            if column not in frame:
                raise ValueError(f"composition member is missing {column}")
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError("composition member scores must be finite")
            combined[f"member__{member}__{horizon}"] = values
    return combined


def compose_frozen_scores(
    frozen: FrozenRevisionComposition,
    members: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply only retained mappings; this function fits nothing on any role."""
    combined = join_member_scores(members)
    for horizon in HORIZONS:
        combined[f"composition_{horizon}_score"] = np.mean(
            [
                frozen.mappings[member, horizon].transform(
                    combined[f"member__{member}__{horizon}"].to_numpy(dtype=np.float64)
                )
                for member in MEMBERS
            ],
            axis=0,
        )
    return combined


def _development(members: Mapping[str, pd.DataFrame]) -> None:
    for frame in members.values():
        if "partition" not in frame or set(frame["partition"].astype(str)) != set(
            _PARTITIONS
        ):
            raise ValueError(
                "composition fitting/audit requires only both development roles"
            )


def _metadata(
    metadata: Mapping[str, Mapping[str, Any]],
    model_seed: int,
    members: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    if model_seed not in MODEL_SEEDS or set(metadata) != set(MEMBERS):
        raise ValueError(
            "composition requires a declared prespecified seed and fixed-pair metadata"
        )
    frozen_metadata = {}
    for member in MEMBERS:
        entry = metadata[member]
        summary = entry["summary"]
        if (
            summary.get("revision_contract") != REVISION_CONTRACT
            or summary.get("input_feature_mode") != "cgm_mask_clock"
            or summary.get("training_seed") not in MODEL_SEEDS
            or (
                member == "lstm"
                and (
                    summary.get("architecture") != "lstm"
                    or summary["training_seed"] != model_seed
                )
            )
            or (member == "persistence" and summary.get("baseline") != "persistence")
        ):
            raise ValueError(
                "composition member is historical, unmatched, "
                "or has a different information budget"
            )
        for horizon in HORIZONS:
            task = f"hypo_{horizon}"
            task_summary = summary["tasks"][task]
            training = (
                summary.get("training_targets", {}).get("tasks", {}).get(task, {})
            )
            target_column = training.get(
                "training_target_column", training.get("target_column")
            )
            if (
                task_summary.get("target_column") != f"severe_hypo_{horizon}_label"
                or task_summary.get("mask_column") != f"{task}_training_mask"
                or task_summary.get("temperature_fit_partition") != "calibration"
                or target_column != f"{task}_training_target"
                or training.get("mask_column") != f"{task}_training_mask"
            ):
                raise ValueError(
                    "composition member head is not target aligned/calibration fitted"
                )
            frame = members[member]
            expected = apply_temperature(
                frame[f"{task}_logit"].to_numpy(dtype=np.float64),
                float(task_summary["temperature"]),
            )
            probability = frame[f"severe_hypo_{horizon}_probability"].to_numpy(
                dtype=np.float64
            )
            if not np.array_equal(
                probability, frame[f"{task}_probability"].to_numpy()
            ) or not (np.allclose(probability, expected, rtol=0, atol=1e-12)):
                raise ValueError(
                    "composition member severe/hypo calibrated probability "
                    "binding differs"
                )
        run_dir = str(Path(entry["run_dir"]).resolve())
        frozen_metadata[member] = {
            "run_dir": run_dir,
            "training_seed": summary["training_seed"],
            "comparison_lstm_seed": model_seed,
            "shared_control_fit": member == "persistence",
            "revision_contract": REVISION_CONTRACT,
            "input_feature_mode": "cgm_mask_clock",
            "score_columns": {str(h): f"severe_hypo_{h}_probability" for h in HORIZONS},
        }
    return frozen_metadata


def fit_revision_composition(
    members: Mapping[str, pd.DataFrame],
    *,
    member_metadata: Mapping[str, Mapping[str, Any]],
    replays: Mapping[str, ReplayInputs],
    source_order: tuple[str, ...],
    model_seed: int = 17,
    candidate_count: int = 2001,
    diagnostic_thresholds: Mapping[int, Mapping[str, float]] | None = None,
) -> RevisionCompositionResult:
    _development(members)
    metadata = _metadata(member_metadata, model_seed, members)
    context = RevisionTrainingContext(replays, source_order=source_order)
    for frame in members.values():
        for partition in _PARTITIONS:
            context._bind(frame.loc[frame["partition"].eq(partition)], partition)
    combined = join_member_scores(members)
    calibration = combined["partition"].eq("calibration").to_numpy()
    mappings = {
        (member, horizon): FrozenStepECDF.fit(
            combined.loc[calibration, f"member__{member}__{horizon}"].to_numpy(
                dtype=np.float64
            ),
            partition="calibration",
        )
        for member in MEMBERS
        for horizon in HORIZONS
    }
    policies, evaluations = {}, {}
    for horizon in HORIZONS:
        column = f"composition_{horizon}_score"
        combined[column] = np.mean(
            [
                mappings[member, horizon].transform(
                    combined[f"member__{member}__{horizon}"].to_numpy()
                )
                for member in MEMBERS
            ],
            axis=0,
        )
        spec = FixedFrameSpec(column, horizon, source_order)
        events, exposure = (
            replays["calibration"].events[horizon],
            replays["calibration"].exposure[horizon],
        )
        timeline = context._timeline(combined.loc[calibration], horizon)
        policy = fit_fixed_frame_alarm_policy(
            timeline,
            events,
            exposure,
            spec=spec,
            partition="calibration",
            glucose_cut=54,
            nominal_fapd=0.5,
            candidate_count=candidate_count,
        )
        policies[horizon] = policy.as_dict()
        for partition in _PARTITIONS:
            evaluations[partition, horizon] = evaluate_fixed_frame(
                context._timeline(
                    combined.loc[combined["partition"].eq(partition)], horizon
                ),
                replays[partition].events[horizon],
                replays[partition].exposure[horizon],
                spec=spec,
                threshold=policy.threshold,
                glucose_cut=54,
            )
    frozen = FrozenRevisionComposition(
        model_seed, tuple(source_order), mappings, policies, metadata
    )
    audit = audit_development_mapping(
        frozen,
        members,
        replays=replays,
        thresholds=diagnostic_thresholds,
    )
    return RevisionCompositionResult(frozen, combined, evaluations, audit)


def _index(frame: pd.DataFrame, columns: list[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[columns])


def _alarm_differences(
    step: FixedFrameEvaluation, linear: FixedFrameEvaluation
) -> dict[str, int]:
    step_keys, linear_keys = _index(step.alarms, _KEYS), _index(linear.alarms, _KEYS)
    common = step_keys.intersection(linear_keys)
    match_columns = ["matched", "unmatched", "matched_event_id", "matched_event_onset"]
    a, b = step.alarms.set_index(_KEYS), linear.alarms.set_index(_KEYS)
    changed = np.zeros(len(common), dtype=bool)
    for column in match_columns:
        left, right = a.loc[common, column], b.loc[common, column]
        changed |= ~(left.eq(right) | (left.isna() & right.isna())).to_numpy(dtype=bool)
    event_keys = ["source_id", "participant_id", "event_id", "event_onset"]
    a, b = step.events.set_index(event_keys), linear.events.set_index(event_keys)
    if not a.index.equals(b.index):
        raise ValueError("mapping diagnostic event truth changed between maps")
    event_alarm_changed = ~(
        a["alarm_context_end"].eq(b["alarm_context_end"])
        | (a["alarm_context_end"].isna() & b["alarm_context_end"].isna())
    )
    return {
        "common_emitted_alarms": len(common),
        "step_only_emitted_alarms": len(step_keys.difference(linear_keys)),
        "linear_only_emitted_alarms": len(linear_keys.difference(step_keys)),
        "common_emissions_with_match_change": int(changed.sum()),
        "base_events_with_detection_change": int(a["detected"].ne(b["detected"]).sum()),
        "base_events_with_matched_alarm_change": int(event_alarm_changed.sum()),
    }


def audit_development_mapping(
    frozen: FrozenRevisionComposition,
    members: Mapping[str, pd.DataFrame],
    *,
    replays: Mapping[str, ReplayInputs],
    thresholds: Mapping[int, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Compare exact maps/alarms/matches without refitting a diagnostic policy."""
    _development(members)
    combined = compose_frozen_scores(frozen, members)
    output: dict[str, Any] = {
        "scope": "development_only_mapping_diagnostic_no_threshold_refit",
        "equivalence_claim": "not_inferred_from_aggregate_parity",
        "horizons": {},
    }
    for horizon in HORIZONS:
        linear = np.mean(
            [
                frozen.mappings[member, horizon].legacy_interpolated(
                    combined[f"member__{member}__{horizon}"].to_numpy(dtype=np.float64)
                )
                for member in MEMBERS
            ],
            axis=0,
        )
        step = combined[f"composition_{horizon}_score"].to_numpy(dtype=np.float64)
        counts = {frozen.mappings[member, horizon].sample_count for member in MEMBERS}
        if thresholds is None:
            if len(counts) != 1 or next(iter(counts)) < 2:
                raise ValueError(
                    "affine diagnostic thresholds require shared calibration N >=2"
                )
            n = next(iter(counts))
            step_threshold = float(frozen.policies[horizon]["threshold"])
            linear_threshold = (n * step_threshold - 1.0) / (n - 1)
            threshold_rule = "saved_step_threshold_and_calibration_affine_counterpart"
        else:
            if set(thresholds) != set(HORIZONS):
                raise ValueError(
                    "declared diagnostic thresholds must cover both horizons"
                )
            step_threshold = float(thresholds[horizon]["step"])
            linear_threshold = float(thresholds[horizon]["linear"])
            threshold_rule = "explicit_declared_threshold_pair_no_refit"
        if not np.isfinite((step_threshold, linear_threshold)).all():
            raise ValueError("declared diagnostic thresholds must be finite")
        partitions = {}
        for partition in _PARTITIONS:
            mask = combined["partition"].eq(partition).to_numpy()
            timeline = combined.loc[mask].copy(deep=False)
            timeline["base_evaluable"] = timeline[f"base_evaluable_{horizon}"]
            timeline["diagnostic_linear_score"] = linear[mask]
            events, exposure = (
                replays[partition].events[horizon],
                replays[partition].exposure[horizon],
            )
            spec = FixedFrameSpec(
                f"composition_{horizon}_score", horizon, frozen.source_order
            )
            a = evaluate_fixed_frame(
                timeline,
                events,
                exposure,
                spec=spec,
                threshold=step_threshold,
                glucose_cut=54,
            )
            b = evaluate_fixed_frame(
                timeline,
                events,
                exposure,
                spec=FixedFrameSpec(
                    "diagnostic_linear_score", horizon, frozen.source_order
                ),
                threshold=linear_threshold,
                glucose_cut=54,
            )
            disagreement = (step[mask] >= step_threshold) != (
                linear[mask] >= linear_threshold
            )
            partitions[partition] = {
                "rows": int(mask.sum()),
                "score_max_abs_difference": float(
                    np.max(np.abs(step[mask] - linear[mask]))
                ),
                "threshold_crossing_disagreements": int(disagreement.sum()),
                "online_eligible_crossing_disagreements": int(
                    (disagreement & _online_eligible(timeline, 54)).sum()
                ),
                **_alarm_differences(a, b),
            }
        output["horizons"][str(horizon)] = {
            "step_threshold": step_threshold,
            "linear_threshold": linear_threshold,
            "threshold_rule": threshold_rule,
            "partitions": partitions,
        }
    return output


def build_fit_alias_ledger(
    entries: Sequence[Mapping[str, Any]],
    *,
    pairings: Sequence[Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Count declared producer identities, never numerical coincidences as fits."""
    by_id, numerical, parents, owners = {}, {}, {}, {}

    def representative(name: str) -> str:
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    for raw in entries:
        entry = dict(raw)
        name = entry["entry_id"]
        if not isinstance(name, str) or not name or name in by_id:
            raise ValueError("fit entries require unique explicit entry_id values")
        if not entry.get("run_dir"):
            raise ValueError("fit entry lacks an explicit producer run directory")
        run = Path(entry["run_dir"]).resolve()
        artifact = (
            entry.get("producer_fit_path")
            or entry.get("checkpoint_path")
            or entry.get("fit_path")
        )
        artifact = str(Path(artifact).resolve()) if artifact else None
        identity = artifact or str(run)
        kind = (
            "canonical_producer_artifact_path"
            if artifact
            else "canonical_producer_run_path"
        )
        parents[name] = name
        for path_identity in (("run", str(run)), ("artifact", artifact)):
            if path_identity[1] is None:
                continue
            if path_identity in owners:
                parents[representative(name)] = representative(owners[path_identity])
            else:
                owners[path_identity] = name
        record = {
            "entry_id": name,
            "arm": entry.get("arm"),
            "model_seed": entry.get("model_seed"),
            "run_dir": str(run),
            "fit_identity": identity,
            "producer_artifact_path": artifact,
            "identity_basis": kind,
        }
        path = Path(identity)
        if path.exists():
            stat = path.stat()
            record["observed_path"] = {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "regular_file": path.is_file(),
                "directory": path.is_dir(),
            }
        else:
            record["observed_path"] = None
        record["claimed_alias_of"] = entry.get("alias_of")
        by_id[name] = record
        if entry.get("numerical_equivalence_group") is not None:
            numerical.setdefault(str(entry["numerical_equivalence_group"]), []).append(
                name
            )
    groups: dict[str, list[str]] = {}
    for name, record in by_id.items():
        producer = representative(name)
        groups.setdefault(producer, []).append(name)
        record["fit_identity"] = by_id[producer]["fit_identity"]
    alias_claims = []
    for name, record in by_id.items():
        alias = record["claimed_alias_of"]
        if alias is not None:
            if alias not in by_id or alias == name:
                raise ValueError(
                    "alias_of must reference a different provided fit entry"
                )
            alias_claims.append(
                {
                    "entry_id": name,
                    "alias_of": alias,
                    "confirmed_by_producer_path": record["fit_identity"]
                    == by_id[alias]["fit_identity"],
                }
            )
    unique_pairs = set()
    if pairings is not None:
        for pairing in pairings:
            if len(pairing) != 2 or any(name not in by_id for name in pairing):
                raise ValueError(
                    "pairing entries must name exactly two provided fit entries"
                )
            unique_pairs.add(
                tuple(sorted(by_id[name]["fit_identity"] for name in pairing))
            )
    return {
        "entry_count": len(by_id),
        "unique_fit_count": len(groups),
        "count_basis": "provided_canonical_producer_run_or_checkpoint_identity",
        "alias_group_interpretation": (
            "same_producer_fit_lineage_or_reused_artifact_not_numerical_equivalence"
        ),
        "numerical_equality_proves_alias": False,
        "entries": list(by_id.values()),
        "confirmed_alias_groups": [
            names for names in groups.values() if len(names) > 1
        ],
        "alias_claims": alias_claims,
        "unverified_numerical_coincidences": [
            {"entries": names, "status": "not_alias_evidence"}
            for names in numerical.values()
            if len({by_id[name]["fit_identity"] for name in names}) > 1
        ],
        "pairing_counts_status": "not_provided"
        if pairings is None
        else "from_explicit_configuration_entries",
        "pairing_entry_count": None if pairings is None else len(pairings),
        "unique_fit_pair_count": None if pairings is None else len(unique_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--lstm-run", type=Path, required=True)
    parser.add_argument("--persistence-run", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=MODEL_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("composition requires a new private output directory")
    config = load_config(args.config)
    candidate_count = int(config["evaluation"].get("threshold_candidate_count", 2001))
    if candidate_count != 2001 and not config["project"].get("synthetic", False):
        raise ValueError(
            "real composition requires the original 2001 threshold candidates"
        )
    _, replays = load_replay_bundle(args.prepared, config, sequences=False)
    members, metadata, ledger_entries = {}, {}, []
    for member, run in (("lstm", args.lstm_run), ("persistence", args.persistence_run)):
        path = run / "predictions.parquet"
        roles = set(
            pq.read_table(path, columns=["partition"]).column(0).unique().to_pylist()
        )
        if roles != set(_PARTITIONS):
            raise ValueError("composition CLI refuses non-development prediction roles")
        members[member] = pq.ParquetFile(path).read().to_pandas()
        metadata[member] = {
            "run_dir": str(run),
            "summary": json.loads((run / "model_summary.json").read_text()),
        }
        artifact = run / (
            "ckpts/best.pt" if member == "lstm" else "baseline_models.joblib"
        )
        ledger_entries.append(
            {
                "entry_id": member,
                "run_dir": str(run),
                "arm": member,
                "model_seed": metadata[member]["summary"]["training_seed"],
                "checkpoint_path": str(artifact),
            }
        )
    result = fit_revision_composition(
        members,
        member_metadata=metadata,
        replays=replays,
        source_order=tuple(sorted(map(str, config["data"]["development_sources"]))),
        model_seed=args.seed,
        candidate_count=candidate_count,
    )
    args.output.mkdir(mode=0o700)
    save_frozen_composition(result.frozen, args.output / "frozen")
    result.predictions.to_parquet(args.output / "predictions.parquet", index=False)
    for (partition, horizon), evaluation in result.evaluations.items():
        root = args.output / "fixed-frame" / partition / str(horizon)
        root.mkdir(parents=True, mode=0o700)
        for name in ("participants", "sources", "alarms", "events"):
            getattr(evaluation, name).to_parquet(root / f"{name}.parquet", index=False)
    for name, report in (
        ("development-mapping-audit", result.mapping_audit),
        (
            "fit-alias-ledger",
            build_fit_alias_ledger(ledger_entries, pairings=[MEMBERS]),
        ),
    ):
        (args.output / f"{name}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
