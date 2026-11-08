"""Prospective fixed-universe alarm evaluation, separate from legacy metrics.

The caller supplies the complete context-valid decision history, an independently
built event catalogue for one horizon, and the exposure before glucose gating.
Retrospective evaluability never changes an emitted alarm's refractory effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import _collapse_positions
from .operational_metrics import (
    _candidate_thresholds,
    _FenwickSet,
    _ParticipantOperationalSweep,
)

_KEYS = ["source_id", "participant_id"]
_EVENT_KEYS = [*_KEYS, "event_id"]
_EVENT_TRUTH = [*_EVENT_KEYS, "event_onset"]
_ANCHOR_FLAGS = (
    "anchor_cgm_observed",
    "anchor_cgm_interpolated",
    "anchor_cgm_censored",
)
_COVERAGE_MINUTES = (5, 10, 15, 20)
_MODEL_SEEDS = (17, 23, 42, 71, 101)


@dataclass(frozen=True)
class FixedFrameSpec:
    score_column: str
    horizon_minutes: int
    source_order: tuple[str, ...]

    def validate(self) -> None:
        if self.horizon_minutes not in (30, 60):
            raise ValueError("fixed-frame horizon must be 30 or 60 minutes")
        if not self.source_order or len(set(self.source_order)) != len(
            self.source_order
        ):
            raise ValueError("source_order must contain unique source names")
        reserved = {
            *_KEYS, "context_end", "patient_day", "cgm_last", "base_evaluable",
            *_ANCHOR_FLAGS,
        }
        if not self.score_column or self.score_column in reserved:
            raise ValueError("score_column must name a distinct score")


@dataclass(frozen=True)
class FixedFrameEvaluation:
    spec: FixedFrameSpec
    threshold: float
    glucose_cut: float
    participants: pd.DataFrame
    sources: pd.DataFrame
    alarms: pd.DataFrame
    events: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class FixedFrameAlarmPolicy:
    partition: str
    endpoint: dict[str, Any]
    glucose_cut: float
    nominal_fapd: float
    threshold: float
    threshold_count: int
    candidate_rule: str
    tie_rule: str
    achieved_equal_source_participant_macro_fapd: float
    achieved_equal_source_participant_macro_fixed_sensitivity: float
    participant_count: int
    base_event_count: int
    patient_days: int
    observation_minutes: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _identifiers(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"{name} has missing {column}")
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"{name} has empty {column}")


def _strict_boolean(frame: pd.DataFrame, column: str) -> None:
    if not frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{column} must contain complete boolean values")
    frame[column] = frame[column].astype(bool)


def _numeric(frame: pd.DataFrame, column: str, name: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} {column} must be finite")
    frame[column] = values
    return values


def _participant_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(frame[_KEYS].itertuples(index=False, name=None))


def _validated_inputs(
    timeline: pd.DataFrame,
    base_events: pd.DataFrame,
    base_exposure: pd.DataFrame,
    spec: FixedFrameSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec.validate()
    _require_columns(
        timeline,
        {
            *_KEYS, "context_end", "patient_day", "cgm_last", "base_evaluable",
            *_ANCHOR_FLAGS, spec.score_column,
        },
        "decision timeline",
    )
    _require_columns(base_events, set(_EVENT_TRUTH), "base events")
    _require_columns(
        base_exposure, {*_KEYS, "patient_days", "observation_minutes"},
        "base exposure",
    )
    timeline_columns = {
        *_KEYS, "context_end", "patient_day", "cgm_last", "base_evaluable",
        *_ANCHOR_FLAGS, spec.score_column,
    }
    frame, events, exposure = (
        table[sorted(columns | ({"horizon_minutes"} if "horizon_minutes" in table
                                else set()))].copy()
        for table, columns in (
            (timeline, timeline_columns), (base_events, set(_EVENT_TRUTH)),
            (base_exposure, {*_KEYS, "patient_days", "observation_minutes"}),
        )
    )
    if frame.empty or exposure.empty:
        raise ValueError("decision timeline and base exposure must be non-empty")
    for name, table in (
        ("decision timeline", frame), ("base events", events),
        ("base exposure", exposure),
    ):
        _identifiers(table, _KEYS, name)
        if "horizon_minutes" in table:
            if not table["horizon_minutes"].eq(spec.horizon_minutes).all():
                raise ValueError(f"{name} horizon does not match endpoint")
    for name, table in (("decision timeline", frame), ("base exposure", exposure)):
        if set(table["source_id"]) != set(spec.source_order):
            raise ValueError(f"{name} source set mismatch")
    if exposure.duplicated(_KEYS).any():
        raise ValueError("base exposure participant keys are not unique")
    if _participant_keys(frame) != _participant_keys(exposure):
        raise ValueError("decision timeline and base exposure participants differ")
    if not _participant_keys(events).issubset(_participant_keys(exposure)):
        raise ValueError("base events contain participants outside base exposure")

    frame["context_end"] = pd.to_datetime(
        frame["context_end"], utc=True, errors="coerce"
    )
    if frame["context_end"].isna().any():
        raise ValueError("decision timeline has invalid context_end")
    # A patient day is a calendar identifier supplied by the audited adapter,
    # not an instant whose date should change when converted to UTC.
    try:
        calendar_days = {}
        for value in frame["patient_day"].unique():
            day = pd.Timestamp(value)
            if pd.isna(day) or day != day.normalize():
                raise ValueError
            calendar_days[value] = pd.Timestamp(day.date(), tz="UTC")
        frame["patient_day"] = pd.to_datetime(
            frame["patient_day"].map(calendar_days), utc=True
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "patient_day must be a valid calendar-day identifier"
        ) from error
    if frame.duplicated([*_KEYS, "context_end"]).any():
        raise ValueError("decision keys are not unique")
    for column in (*_ANCHOR_FLAGS, "base_evaluable"):
        _strict_boolean(frame, column)
    if (frame["anchor_cgm_observed"] & frame["anchor_cgm_interpolated"]).any():
        raise ValueError("an observed anchor cannot also be interpolated")
    _numeric(frame, spec.score_column, "decision timeline")
    _numeric(frame, "cgm_last", "decision timeline")

    days = _numeric(exposure, "patient_days", "base exposure")
    minutes = _numeric(exposure, "observation_minutes", "base exposure")
    if (days <= 0).any() or not np.equal(days, np.floor(days)).all():
        raise ValueError("base exposure patient_days must be positive integers")
    if (minutes <= 0).any() or (minutes > days * 1440).any():
        raise ValueError("base exposure observation_minutes is outside calendar days")
    exposure["patient_days"] = days.astype(int)
    indexed_exposure = exposure.set_index(_KEYS)
    for key, part in frame.groupby(_KEYS, sort=False, observed=True):
        actual_days = part.loc[part["base_evaluable"], "patient_day"].nunique()
        expected = indexed_exposure.loc[key, "patient_days"]
        if actual_days != expected:
            raise ValueError("base exposure patient_days differs from evaluable days")

    _identifiers(events, ["event_id"], "base events")
    events["event_onset"] = pd.to_datetime(
        events["event_onset"], utc=True, errors="coerce"
    )
    if events["event_onset"].isna().any():
        raise ValueError("base events have invalid event_onset")
    if events.duplicated(_EVENT_KEYS).any():
        raise ValueError("base event identity is not unique or has conflicting onset")
    if events.duplicated([*_KEYS, "event_onset"]).any():
        raise ValueError("one event onset maps to multiple event IDs")
    frame = frame.sort_values([*_KEYS, "context_end"]).reset_index(drop=True)
    events = events.sort_values([*_KEYS, "event_onset"]).reset_index(drop=True)
    exposure = exposure.sort_values(_KEYS).reset_index(drop=True)
    base_times_by_key = {
        key: part.loc[part["base_evaluable"], "context_end"].to_numpy(
            dtype="datetime64[ns]"
        ).astype(np.int64)
        for key, part in frame.groupby(_KEYS, sort=False, observed=True)
    }
    horizon_ns = np.int64(spec.horizon_minutes) * 60_000_000_000
    for key, part in events.groupby(_KEYS, sort=False, observed=True):
        times = base_times_by_key[key]
        onsets = part["event_onset"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        starts = np.searchsorted(times, onsets - horizon_ns, side="left")
        stops = np.searchsorted(times, onsets, side="left")
        if np.any(starts == stops):
            raise ValueError("base event has no base-evaluable lead-in decision")
    return frame, events, exposure


def _online_eligible(frame: pd.DataFrame, glucose_cut: float) -> np.ndarray:
    return (
        frame["anchor_cgm_observed"]
        & ~frame["anchor_cgm_interpolated"]
        & ~frame["anchor_cgm_censored"]
        & frame["cgm_last"].ge(max(54.0, glucose_cut))
    ).to_numpy(dtype=bool)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _aggregate(table: pd.DataFrame) -> dict[str, Any]:
    positive = table.loc[table["base_event_count"] > 0]
    counts = (
        "base_event_count", "reachable_event_count", "detected_event_count",
        "patient_days", "observation_minutes", "eligible_decision_count",
        "base_evaluable_decision_count", "eligible_evaluable_decision_count",
        "emitted_alarm_count", "delivered_alarm_count", "unevaluable_alarm_count",
        "matched_alarm_count", "unmatched_alarm_count", "late_detection_count",
        "missed_event_count",
        *(f"coverage_{lead}_event_count" for lead in _COVERAGE_MINUTES),
    )
    output = {column: table[column].sum().item() for column in counts}
    output.update({
        "participant_count": len(table),
        "event_positive_participant_count": len(positive),
        "fixed_sensitivity": float(positive["fixed_sensitivity"].mean()),
        "conditional_sensitivity": float(positive["conditional_sensitivity"].mean()),
        "event_pooled_sensitivity": _ratio(
            output["detected_event_count"], output["base_event_count"]
        ),
        "participant_macro_unmatched_alarms_per_patient_day": float(
            table["unmatched_alarms_per_patient_day"].mean()
        ),
        "participant_macro_delivered_alarms_per_patient_day": float(
            table["delivered_alarms_per_patient_day"].mean()
        ),
        "pooled_unmatched_alarms_per_patient_day": _ratio(
            output["unmatched_alarm_count"], output["patient_days"]
        ),
        "pooled_unmatched_alarms_per_24h": _ratio(
            output["unmatched_alarm_count"], output["observation_minutes"] / 1440
        ),
        "pooled_delivered_alarms_per_patient_day": _ratio(
            output["delivered_alarm_count"], output["patient_days"]
        ),
        "ppv": _ratio(output["matched_alarm_count"], output["delivered_alarm_count"]),
        "mean_participant_ppv": float(table["ppv"].mean()),
        "effective_24h_days": float(output["observation_minutes"] / 1440),
    })
    leads = np.asarray([
        value for values in table["matched_lead_minutes"] for value in values
    ], dtype=float)
    for label, quantile in (("median", 0.5), ("q25", 0.25), ("q75", 0.75)):
        output[f"matched_only_lead_minutes_{label}"] = (
            float(np.quantile(leads, quantile)) if leads.size else float("nan")
        )
    output["matched_only_lead_minutes_mean"] = (
        float(leads.mean()) if leads.size else float("nan")
    )
    output["reachable_event_fraction"] = _ratio(
        output["reachable_event_count"], output["base_event_count"]
    )
    for minimum in _COVERAGE_MINUTES:
        output[f"coverage_{minimum}_event_fraction"] = _ratio(
            output[f"coverage_{minimum}_event_count"], output["base_event_count"]
        )
    return output


def evaluate_fixed_frame(
    timeline: pd.DataFrame,
    base_events: pd.DataFrame,
    base_exposure: pd.DataFrame,
    *,
    spec: FixedFrameSpec,
    threshold: float,
    glucose_cut: float = 54.0,
) -> FixedFrameEvaluation:
    """Evaluate causal emissions against fixed, independently supplied truth.

    ``delivered_alarm_count`` counts emissions inside base exposure; all other
    emissions remain in the alarm ledger and in ``emitted_alarm_count``. They
    affect refractory state but do not receive retrospective FA attribution.
    """
    if not np.isfinite(threshold) or not np.isfinite(glucose_cut):
        raise ValueError("threshold and glucose_cut must be finite")
    frame, events, exposure = _validated_inputs(
        timeline, base_events, base_exposure, spec
    )
    return _evaluate_validated(
        frame, events, exposure, spec=spec, threshold=float(threshold),
        glucose_cut=float(glucose_cut),
    )


def _evaluate_validated(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    spec: FixedFrameSpec,
    threshold: float,
    glucose_cut: float,
) -> FixedFrameEvaluation:
    online = _online_eligible(frame, glucose_cut)
    eligible = frame.assign(online_eligible=online)
    alarm_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    decisions_by_key = {
        key: part for key, part in eligible.groupby(_KEYS, sort=False, observed=True)
    }
    events_by_key = {
        key: part for key, part in events.groupby(_KEYS, sort=False, observed=True)
    }
    horizon = pd.Timedelta(minutes=spec.horizon_minutes)
    for item in exposure.to_dict("records"):
        key = (item["source_id"], item["participant_id"])
        part = decisions_by_key[key]
        part_events = events_by_key.get(key, events.iloc[:0])
        emitted: list[dict[str, Any]] = []
        candidates = part.loc[
            part["online_eligible"] & part[spec.score_column].ge(threshold)
        ]
        times = candidates["context_end"].to_numpy(dtype="datetime64[ns]").astype(
            np.int64
        )
        positions = _collapse_positions(
            times, np.array([0]), np.array([len(times)]), collapse_minutes=30
        )
        for row in candidates.iloc[positions].to_dict("records"):
            emitted.append({
                **{column: row[column] for column in (
                    *_KEYS, "context_end", "patient_day", "cgm_last",
                    "base_evaluable", spec.score_column,
                )},
                "matched": False, "unmatched": bool(row["base_evaluable"]),
                "matched_event_id": None, "matched_event_onset": pd.NaT,
                "lead_minutes": float("nan"),
            })
        matched: dict[str, dict[str, Any]] = {}
        event_records = part_events.to_dict("records")
        for alarm in emitted:
            if not alarm["base_evaluable"]:
                continue
            timestamp = alarm["context_end"]
            for event in event_records:
                if (
                    event["event_id"] not in matched
                    and timestamp < event["event_onset"] <= timestamp + horizon
                ):
                    lead = (event["event_onset"] - timestamp).total_seconds() / 60
                    alarm.update({
                        "matched": True, "unmatched": False,
                        "matched_event_id": event["event_id"],
                        "matched_event_onset": event["event_onset"],
                        "lead_minutes": float(lead),
                    })
                    matched[event["event_id"]] = alarm
                    break
        reachable_count = 0
        coverage = dict.fromkeys(_COVERAGE_MINUTES, 0)
        base_part = part.loc[part["base_evaluable"]]
        base_times = base_part["context_end"].to_numpy(dtype="datetime64[ns]")
        base_online = base_part["online_eligible"].to_numpy(dtype=bool)
        for event in event_records:
            start = int(np.searchsorted(
                base_times, (event["event_onset"] - horizon).to_datetime64(),
                side="left",
            ))
            stop = int(np.searchsorted(
                base_times, event["event_onset"].to_datetime64(), side="left"
            ))
            if start == stop:
                raise ValueError("base event has no base-evaluable lead-in decision")
            eligible_count = int(base_online[start:stop].sum())
            reachable = eligible_count > 0
            reachable_count += int(reachable)
            alarm = matched.get(event["event_id"])
            lead = float("nan") if alarm is None else alarm["lead_minutes"]
            event_row = {
                **{column: event[column] for column in _EVENT_TRUTH},
                "base_lead_in_decision_count": stop - start,
                "eligible_lead_in_decision_count": eligible_count,
                "reachable": reachable, "detected": alarm is not None,
                "alarm_context_end": pd.NaT if alarm is None else alarm["context_end"],
                "lead_minutes": lead,
                "late_detection": alarm is not None and lead < 5,
                "missed": alarm is None,
            }
            for minimum in _COVERAGE_MINUTES:
                covered = alarm is not None and lead >= minimum
                event_row[f"coverage_{minimum}"] = covered
                coverage[minimum] += int(covered)
            event_rows.append(event_row)
        alarm_rows.extend(emitted)
        delivered = sum(alarm["base_evaluable"] for alarm in emitted)
        detected = len(matched)
        base_count = len(event_records)
        participant_rows.append({
            **item,
            "base_patient_days": tuple(sorted(
                part.loc[part["base_evaluable"], "patient_day"].unique()
            )),
            "effective_24h_days": item["observation_minutes"] / 1440,
            "base_event_count": base_count,
            "reachable_event_count": reachable_count,
            "detected_event_count": detected,
            "fixed_sensitivity": _ratio(detected, base_count),
            "conditional_sensitivity": _ratio(detected, reachable_count),
            "eligible_decision_count": int(part["online_eligible"].sum()),
            "base_evaluable_decision_count": int(part["base_evaluable"].sum()),
            "eligible_evaluable_decision_count": int(
                (part["online_eligible"] & part["base_evaluable"]).sum()
            ),
            "emitted_alarm_count": len(emitted),
            "delivered_alarm_count": delivered,
            "unevaluable_alarm_count": len(emitted) - delivered,
            "matched_alarm_count": detected,
            "unmatched_alarm_count": delivered - detected,
            "ppv": _ratio(detected, delivered),
            "matched_lead_minutes": tuple(
                alarm["lead_minutes"] for alarm in matched.values()
            ),
            "unmatched_alarms_per_patient_day": (delivered - detected)
            / item["patient_days"],
            "delivered_alarms_per_patient_day": delivered / item["patient_days"],
            "late_detection_count": sum(
                alarm["lead_minutes"] < 5 for alarm in matched.values()
            ),
            "missed_event_count": base_count - detected,
            **{
                f"coverage_{lead}_event_count": count
                for lead, count in coverage.items()
            },
        })
    participants = pd.DataFrame(participant_rows)
    sources = pd.DataFrame([
        {"source_id": source, **_aggregate(
            participants.loc[participants["source_id"].eq(source)]
        )}
        for source in spec.source_order
    ])
    summary = _aggregate(participants)
    undefined = sources.loc[
        sources["event_positive_participant_count"].eq(0), "source_id"
    ].tolist()
    summary.update({
        "source_count": len(spec.source_order),
        "sensitivity_defined": not undefined,
        "undefined_sensitivity_sources": undefined,
        "participant_pooled_sensitivity": summary.pop("fixed_sensitivity"),
        "equal_source_participant_macro_fixed_sensitivity": (
            float("nan") if undefined else float(sources["fixed_sensitivity"].mean())
        ),
        "equal_source_participant_macro_unmatched_alarms_per_patient_day": float(
            sources["participant_macro_unmatched_alarms_per_patient_day"].mean()
        ),
        "equal_source_participant_macro_delivered_alarms_per_patient_day": float(
            sources["participant_macro_delivered_alarms_per_patient_day"].mean()
        ),
    })
    alarm_columns = [
        *_KEYS, "context_end", "patient_day", "cgm_last", "base_evaluable",
        spec.score_column, "matched", "unmatched", "matched_event_id",
        "matched_event_onset", "lead_minutes",
    ]
    event_columns = [
        *_EVENT_TRUTH, "base_lead_in_decision_count",
        "eligible_lead_in_decision_count", "reachable", "detected",
        "alarm_context_end", "lead_minutes", "late_detection", "missed",
        *(f"coverage_{lead}" for lead in _COVERAGE_MINUTES),
    ]
    return FixedFrameEvaluation(
        spec, threshold, glucose_cut, participants, sources,
        pd.DataFrame(alarm_rows, columns=alarm_columns),
        pd.DataFrame(event_rows, columns=event_columns), summary,
    )


class _ExposureAwareSelected(_FenwickSet):
    """Track all emissions and the subset attributable to fixed exposure."""

    def __init__(self, evaluable: np.ndarray) -> None:
        super().__init__(len(evaluable))
        self.evaluable = evaluable
        self.attributable = _FenwickSet(len(evaluable))

    def add(self, index: int) -> None:
        super().add(index)
        if self.evaluable[index]:
            self.attributable.add(index)

    def remove(self, index: int) -> None:
        super().remove(index)
        if self.evaluable[index]:
            self.attributable.remove(index)


class _FixedFrameParticipantSweep(_ParticipantOperationalSweep):
    """Reuse exact refractory insertion, with fixed events and masked matching.

    Only ``activate`` is inherited. Its selected set retains unevaluable alarms
    for causal suppression; the second Fenwick set serves retrospective counts.
    """

    def __init__(
        self, participant: pd.DataFrame, events: pd.DataFrame, *,
        spec: FixedFrameSpec, source_index: int, patient_days: int,
    ) -> None:
        self.times = participant["context_end"].to_numpy(
            dtype="datetime64[ns]"
        ).astype(np.int64)
        self.event_onsets = events["event_onset"].to_numpy(
            dtype="datetime64[ns]"
        ).astype(np.int64)
        self.event_count = len(self.event_onsets)
        self.patient_days = patient_days
        self.source_index = source_index
        self.collapse_ns = np.int64(30) * 60_000_000_000
        self.horizon_ns = np.int64(spec.horizon_minutes) * 60_000_000_000
        self.active = _FenwickSet(len(self.times))
        self.selected = _ExposureAwareSelected(
            participant["base_evaluable"].to_numpy(dtype=bool)
        )
        self.sensitivity_contribution = 0.0
        self.fapd_contribution = 0.0

    def recompute_detected_events(self) -> int:
        detected, last_alarm = 0, -1
        for onset in self.event_onsets:
            lower = int(np.searchsorted(
                self.times, onset - self.horizon_ns, side="left"
            ))
            alarm = self.selected.attributable.first_at_or_after(
                max(lower, last_alarm + 1)
            )
            if alarm is not None and self.times[alarm] < onset:
                detected += 1
                last_alarm = alarm
        return detected


def _fixed_frame_sweep_points(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    spec: FixedFrameSpec,
    glucose_cut: float,
    candidates: np.ndarray,
    retain_all: bool = False,
) -> list[dict[str, float]]:
    """Exact cached-array sweep; never builds per-threshold DataFrame ledgers."""
    participant_counts = exposure.groupby("source_id", observed=True).size().reindex(
        spec.source_order, fill_value=0
    ).to_numpy(dtype=int)
    positive_counts = events[_KEYS].drop_duplicates().groupby(
        "source_id", observed=True
    ).size().reindex(spec.source_order, fill_value=0).to_numpy(dtype=int)
    if np.any(positive_counts == 0):
        raise ValueError("base sensitivity is undefined for at least one source")
    eligible = frame.loc[_online_eligible(frame, glucose_cut)].reset_index(drop=True)
    source_indices = {source: index for index, source in enumerate(spec.source_order)}
    events_by_key = {
        key: part for key, part in events.groupby(_KEYS, sort=False, observed=True)
    }
    exposure_by_key = exposure.set_index(_KEYS)
    row_state = np.empty(len(eligible), dtype=int)
    row_local = np.empty(len(eligible), dtype=int)
    states: list[_FixedFrameParticipantSweep] = []
    offset = 0
    for key, participant in eligible.groupby(_KEYS, sort=False, observed=True):
        state = _FixedFrameParticipantSweep(
            participant, events_by_key.get(key, events.iloc[:0]), spec=spec,
            source_index=source_indices[key[0]],
            patient_days=int(exposure_by_key.loc[key, "patient_days"]),
        )
        count = len(participant)
        row_state[offset:offset + count] = len(states)
        row_local[offset:offset + count] = np.arange(count)
        states.append(state)
        offset += count
    sensitivity_sums = np.zeros(len(spec.source_order))
    fapd_sums = np.zeros(len(spec.source_order))
    scores = eligible[spec.score_column].to_numpy(dtype=float)
    activation_order = np.argsort(-scores, kind="stable")
    cursor = 0
    points: list[dict[str, float]] = []
    for threshold in np.sort(candidates)[::-1]:
        dirty: set[int] = set()
        while cursor < len(activation_order):
            index = int(activation_order[cursor])
            if scores[index] < threshold:
                break
            state_index = int(row_state[index])
            if states[state_index].activate(int(row_local[index])):
                dirty.add(state_index)
            cursor += 1
        for index in dirty:
            state = states[index]
            detected = state.recompute_detected_events()
            sensitivity = detected / state.event_count if state.event_count else 0.0
            fapd = (state.selected.attributable.count - detected) / state.patient_days
            sensitivity_sums[state.source_index] += (
                sensitivity - state.sensitivity_contribution
            )
            fapd_sums[state.source_index] += fapd - state.fapd_contribution
            state.sensitivity_contribution = sensitivity
            state.fapd_contribution = fapd
        point = {
            "threshold": float(threshold),
            "fixed_sensitivity": float(np.mean(sensitivity_sums / positive_counts)),
            "achieved_fapd": float(np.mean(fapd_sums / participant_counts)),
        }
        if not np.isfinite(list(point.values())).all():
            raise ValueError("fixed-frame sweep produced non-finite metrics")
        if (
            retain_all or not points
            or point["fixed_sensitivity"] != points[-1]["fixed_sensitivity"]
            or point["achieved_fapd"] != points[-1]["achieved_fapd"]
        ):
            points.append(point)
    return list(reversed(points))


def selection_fixed_frame_curve(
    timeline: pd.DataFrame,
    base_events: pd.DataFrame,
    base_exposure: pd.DataFrame,
    *,
    spec: FixedFrameSpec,
    glucose_cut: float = 54.0,
    fapd_min: float = 0.0,
    fapd_max: float = 2.0,
    candidate_count: int | None = None,
    include_raw_points: bool = False,
) -> dict[str, Any]:
    """Exact right-continuous feasible fixed-sensitivity envelope and AUC."""
    if (
        not np.isfinite((glucose_cut, fapd_min, fapd_max)).all()
        or fapd_min < 0 or fapd_max <= fapd_min
    ):
        raise ValueError("finite glucose cut and 0 <= FAPD min < max are required")
    frame, events, exposure = _validated_inputs(
        timeline, base_events, base_exposure, spec
    )
    scores = frame.loc[
        _online_eligible(frame, glucose_cut), spec.score_column
    ].to_numpy(dtype=float)
    candidates = _candidate_thresholds(
        scores if scores.size else frame[spec.score_column].to_numpy(dtype=float),
        candidate_count=candidate_count,
    )
    if not scores.size:
        candidates = candidates[-1:]
    points = _fixed_frame_sweep_points(
        frame, events, exposure, spec=spec, glucose_cut=glucose_cut,
        candidates=candidates, retain_all=include_raw_points,
    )
    by_fapd: dict[float, float] = {}
    for point in points:
        achieved = point["achieved_fapd"]
        by_fapd[achieved] = max(
            by_fapd.get(achieved, -np.inf), point["fixed_sensitivity"]
        )
    frontier = []
    running = 0.0
    for achieved, sensitivity in sorted(by_fapd.items()):
        running = max(running, sensitivity)
        frontier.append({
            "achieved_fapd": achieved, "best_feasible_fixed_sensitivity": running
        })

    def best_at(budget: float) -> float:
        return max(
            point["best_feasible_fixed_sensitivity"] for point in frontier
            if point["achieved_fapd"] <= budget
        )

    breakpoints = [fapd_min, *(
        point["achieved_fapd"] for point in frontier
        if fapd_min < point["achieved_fapd"] < fapd_max
    ), fapd_max]
    area = sum(
        (right - left) * best_at(left)
        for left, right in zip(breakpoints[:-1], breakpoints[1:], strict=True)
    )
    in_range = [
        point for point in points if fapd_min <= point["achieved_fapd"] <= fapd_max
    ]
    best = max(point["fixed_sensitivity"] for point in in_range)
    output: dict[str, Any] = {
        "raw_auc": float(area), "normalized_auc": float(area / (fapd_max - fapd_min)),
        "fapd_min": fapd_min, "fapd_max": fapd_max,
        "best_sensitivity_within_range": best,
        "fapd_at_best_sensitivity": min(
            point["achieved_fapd"] for point in in_range
            if point["fixed_sensitivity"] == best
        ),
        "threshold_count": len(candidates), "frontier": frontier,
        "integration_rule": "right_continuous_step_integral_of_feasible_envelope",
        "sweep_algorithm": "fixed_frame_exact_incremental_refractory",
        "candidate_rule": (
            "all_unique_online_eligible_scores_plus_no_alarm_sentinel"
            if candidate_count is None else
            f"{candidate_count}_online_eligible_quantiles_plus_no_alarm_sentinel"
        ) if scores.size else "no_eligible_decisions_no_alarm_sentinel",
    }
    if include_raw_points:
        output["raw_points"] = points
    return output


def fit_fixed_frame_alarm_policy(
    timeline: pd.DataFrame,
    base_events: pd.DataFrame,
    base_exposure: pd.DataFrame,
    *,
    spec: FixedFrameSpec,
    partition: str,
    glucose_cut: float = 54.0,
    nominal_fapd: float = 0.5,
    candidate_count: int | None = None,
) -> FixedFrameAlarmPolicy:
    """Calibration-only refit; maximize fixed sensitivity, then threshold.

    Candidates use the legacy unique/quantile rule on online-eligible scores,
    including decisions outside retrospective exposure because these affect
    refractory history. If there are no eligible decisions, retain only the
    finite no-alarm sentinel derived from the complete timeline.
    """
    if partition != "calibration":
        raise ValueError("fixed-frame policy must be fit on calibration only")
    for table in (timeline, base_events, base_exposure):
        if "partition" in table and not table["partition"].eq("calibration").all():
            raise ValueError(
                "policy input contains non-calibration or mixed partitions"
            )
    if not np.isfinite(nominal_fapd) or nominal_fapd < 0:
        raise ValueError("nominal_fapd must be finite and non-negative")
    if not np.isfinite(glucose_cut):
        raise ValueError("glucose_cut must be finite")
    frame, events, exposure = _validated_inputs(
        timeline, base_events, base_exposure, spec
    )
    eligible_scores = frame.loc[
        _online_eligible(frame, glucose_cut), spec.score_column
    ].to_numpy(dtype=float)
    scores = frame[spec.score_column].to_numpy(dtype=float)
    candidates = _candidate_thresholds(
        eligible_scores if eligible_scores.size else scores,
        candidate_count=candidate_count,
    )
    if not eligible_scores.size:
        candidates = candidates[-1:]
    points = _fixed_frame_sweep_points(
        frame, events, exposure, spec=spec, glucose_cut=glucose_cut,
        candidates=candidates,
    )
    feasible = [point for point in points if point["achieved_fapd"] <= nominal_fapd]
    if not feasible:
        raise AssertionError("no-alarm sentinel did not provide a feasible policy")
    best = max(feasible, key=lambda point: (
        point["fixed_sensitivity"], point["threshold"]
    ))
    selected = _evaluate_validated(
        frame, events, exposure, spec=spec, threshold=best["threshold"],
        glucose_cut=float(glucose_cut),
    )
    summary = selected.summary
    for observed, expected in (
        (summary["equal_source_participant_macro_fixed_sensitivity"],
         best["fixed_sensitivity"]),
        (summary["equal_source_participant_macro_unmatched_alarms_per_patient_day"],
         best["achieved_fapd"]),
    ):
        if not np.isclose(observed, expected, rtol=0, atol=1e-12):
            raise AssertionError("fixed-frame sweep disagrees with ledger evaluator")
    return FixedFrameAlarmPolicy(
        partition=partition, endpoint=asdict(spec), glucose_cut=float(glucose_cut),
        nominal_fapd=float(nominal_fapd), threshold=selected.threshold,
        threshold_count=len(candidates),
        candidate_rule=(
            "all_unique_online_eligible_scores_plus_no_alarm_sentinel"
            if candidate_count is None else
            f"{candidate_count}_online_eligible_quantiles_plus_no_alarm_sentinel"
        ) if eligible_scores.size else "no_eligible_decisions_no_alarm_sentinel",
        tie_rule="maximum_fixed_sensitivity_then_highest_threshold",
        achieved_equal_source_participant_macro_fapd=summary[
            "equal_source_participant_macro_unmatched_alarms_per_patient_day"
        ],
        achieved_equal_source_participant_macro_fixed_sensitivity=summary[
            "equal_source_participant_macro_fixed_sensitivity"
        ],
        participant_count=summary["participant_count"],
        base_event_count=summary["base_event_count"],
        patient_days=summary["patient_days"],
        observation_minutes=summary["observation_minutes"],
    )


@dataclass(frozen=True)
class FixedFrameKey:
    arm: str
    glucose_cut: float
    horizon_minutes: int
    model_seed: int


def joint_fixed_frame_bootstrap(
    evaluations: Mapping[FixedFrameKey, FixedFrameEvaluation],
    *,
    contrasts: Mapping[str, Mapping[tuple[str, float], float]],
    source_order: tuple[str, ...],
    horizon_order: tuple[int, ...] = (30, 60),
    model_seed_order: tuple[int, ...] = _MODEL_SEEDS,
    replicates: int = 1000,
    bootstrap_seed: int = 20260810,
    metric: str = "fixed_sensitivity",
    include_replicates: bool = False,
) -> dict[str, Any]:
    """Joint paired inference with one participant draw across all comparisons.

    Supply every arm/cut requested by the explicit contrasts for every horizon
    and seed. Smaller *complete* horizon/seed sets are intended for toy oracles.
    Sensitivity draws preserve source-by-horizon-positivity-pattern counts.
    Burden is a separate invocation/draw over all base participants.
    """
    allowed_metrics = {
        "fixed_sensitivity", "unmatched_alarms_per_patient_day",
        "delivered_alarms_per_patient_day",
    }
    if metric not in allowed_metrics:
        raise ValueError(f"unsupported joint bootstrap metric: {metric}")
    for name, sequence in (
        ("source_order", source_order), ("horizon_order", horizon_order),
        ("model_seed_order", model_seed_order),
    ):
        if not sequence or len(set(sequence)) != len(sequence):
            raise ValueError(f"{name} must contain unique values")
    if not set(model_seed_order).issubset(_MODEL_SEEDS):
        raise ValueError("model_seed_order must use the prespecified model seeds")
    if not set(horizon_order).issubset((30, 60)):
        raise ValueError("horizon_order must use the prespecified horizons")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not contrasts or any(not terms for terms in contrasts.values()):
        raise ValueError("explicit non-empty contrasts are required")
    for terms in contrasts.values():
        if not np.isfinite(list(terms.values())).all():
            raise ValueError("contrast coefficients must be finite")
    conditions = set().union(*(set(terms) for terms in contrasts.values()))
    expected = {
        FixedFrameKey(arm, cut, horizon, seed)
        for arm, cut in conditions for horizon in horizon_order
        for seed in model_seed_order
    }
    if set(evaluations) != expected:
        raise ValueError(
            "evaluations do not exactly cover arms/cuts/horizons/model seeds"
        )
    truth_by_horizon: dict[int, pd.DataFrame] = {}
    events_by_horizon: dict[int, pd.DataFrame] = {}
    arrays: dict[FixedFrameKey, np.ndarray] = {}
    common_keys: pd.DataFrame | None = None
    for key in sorted(
        expected, key=lambda value: (
            value.horizon_minutes, value.model_seed, value.arm, value.glucose_cut
        ),
    ):
        evaluation = evaluations[key]
        if (
            evaluation.spec.horizon_minutes != key.horizon_minutes
            or evaluation.glucose_cut != key.glucose_cut
            or evaluation.spec.source_order != source_order
        ):
            raise ValueError("evaluation key/spec/source order mismatch")
        table = evaluation.participants.copy()
        truth_columns = [
            *_KEYS, "base_event_count", "patient_days", "observation_minutes",
            "base_patient_days",
        ]
        _require_columns(table, {*truth_columns, metric}, "bootstrap participant table")
        _identifiers(table, _KEYS, "bootstrap participant table")
        if table.duplicated(_KEYS).any():
            raise ValueError("bootstrap participant keys are not unique")
        if set(table["source_id"]) != set(source_order):
            raise ValueError("bootstrap source set is incomplete")
        table = table.sort_values(_KEYS).reset_index(drop=True)
        if common_keys is None:
            common_keys = table[_KEYS]
        elif not common_keys.equals(table[_KEYS]):
            raise ValueError("joint bootstrap requires identical participant rows")
        event_truth = evaluation.events[_EVENT_TRUTH].sort_values(
            _EVENT_KEYS
        ).reset_index(drop=True)
        counts = event_truth.groupby(_KEYS, observed=True).size()
        expected_counts = np.array([
            counts.get(pair, 0)
            for pair in table[_KEYS].itertuples(index=False, name=None)
        ])
        if not np.array_equal(table["base_event_count"], expected_counts):
            raise ValueError(
                "participant base event truth differs from event catalogue"
            )
        current_truth = table[truth_columns]
        horizon = key.horizon_minutes
        if horizon not in truth_by_horizon:
            truth_by_horizon[horizon] = current_truth
            events_by_horizon[horizon] = event_truth
        elif (
            not current_truth.equals(truth_by_horizon[horizon])
            or not event_truth.equals(events_by_horizon[horizon])
        ):
            raise ValueError(
                "same-horizon event/exposure truth differs across comparisons"
            )
        values = pd.to_numeric(table[metric], errors="coerce").to_numpy(dtype=float)
        positive = expected_counts > 0
        if metric == "fixed_sensitivity":
            if not np.isfinite(values[positive]).all() or (
                (values[positive] < 0) | (values[positive] > 1)
            ).any():
                raise ValueError("positive-participant fixed sensitivity is invalid")
            _require_columns(
                table, {"detected_event_count"}, "bootstrap participant table"
            )
            detected = pd.to_numeric(table["detected_event_count"], errors="coerce")
            if not np.isfinite(detected).all() or (
                (detected < 0) | (detected > expected_counts)
            ).any() or not np.equal(detected, np.floor(detected)).all():
                raise ValueError("detected event counts are invalid")
            if not np.allclose(
                values[positive],
                detected.to_numpy()[positive] / expected_counts[positive],
                rtol=0, atol=1e-12,
            ):
                raise ValueError("fixed sensitivity disagrees with event counts")
        elif not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("participant burden is invalid")
        arrays[key] = values
    if common_keys is None:
        raise AssertionError("validated evaluations unexpectedly empty")

    positivity = np.column_stack([
        truth_by_horizon[horizon]["base_event_count"].to_numpy() > 0
        for horizon in horizon_order
    ])
    source_values = common_keys["source_id"].to_numpy()
    groups: list[np.ndarray] = []
    stratum_counts: list[dict[str, Any]] = []
    for source in source_order:
        source_indices = np.flatnonzero(source_values == source)
        if metric == "fixed_sensitivity":
            for column, horizon in enumerate(horizon_order):
                if not positivity[source_indices, column].any():
                    raise ValueError(
                        f"base sensitivity undefined: source={source}, "
                        f"horizon={horizon}"
                    )
            patterns = sorted(set(map(tuple, positivity[source_indices])))
            for pattern in patterns:
                indices = source_indices[
                    np.all(positivity[source_indices] == pattern, axis=1)
                ]
                if not any(pattern):
                    continue
                groups.append(indices)
                stratum_counts.append({
                    "source_id": source,
                    "positivity_pattern": [bool(value) for value in pattern],
                    "participant_count": len(indices),
                })
        else:
            groups.append(source_indices)
            stratum_counts.append({
                "source_id": source, "participant_count": len(source_indices)
            })

    def estimate(
        indices: np.ndarray, terms: Mapping[tuple[str, float], float]
    ) -> np.ndarray:
        per_horizon_seed = np.empty((len(horizon_order), len(model_seed_order)))
        for h_index, horizon in enumerate(horizon_order):
            for s_index, seed in enumerate(model_seed_order):
                source_values_for_contrast = []
                for source in source_order:
                    selected = indices[source_values[indices] == source]
                    if metric == "fixed_sensitivity":
                        selected = selected[positivity[selected, h_index]]
                    value = sum(
                        coefficient * arrays[
                            FixedFrameKey(arm, cut, horizon, seed)
                        ][selected].mean()
                        for (arm, cut), coefficient in terms.items()
                    )
                    source_values_for_contrast.append(value)
                per_horizon_seed[h_index, s_index] = np.mean(source_values_for_contrast)
        return per_horizon_seed

    identity = np.concatenate(groups)
    point = {name: estimate(identity, terms) for name, terms in contrasts.items()}
    draws = {
        name: np.empty((replicates, len(horizon_order), len(model_seed_order)))
        for name in contrasts
    }
    rng = np.random.default_rng(bootstrap_seed)
    for replicate in range(replicates):
        indices = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])
        for name, terms in contrasts.items():
            draws[name][replicate] = estimate(indices, terms)
    output: dict[str, Any] = {
        "metric": metric, "replicates": replicates,
        "bootstrap_seed": bootstrap_seed, "model_seeds": list(model_seed_order),
        "horizons": list(horizon_order), "source_order": list(source_order),
        "resampling_unit": "source_scoped_participant",
        "population": (
            "event_positive_in_at_least_one_horizon" if metric == "fixed_sensitivity"
            else "all_base_participants"
        ),
        "stratification": (
            "source_by_horizon_positivity_pattern" if metric == "fixed_sensitivity"
            else "source"
        ),
        "stratum_counts": stratum_counts,
        "aggregation": "equal_source_then_equal_horizon_and_model_seed",
        "contrasts": {},
    }
    for name in contrasts:
        values = draws[name].mean(axis=(1, 2))
        if not np.isfinite(values).all():
            raise ValueError("joint bootstrap produced a non-finite draw")
        result: dict[str, Any] = {
            "estimate": float(point[name].mean()),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "coefficients": [
                {"arm": arm, "glucose_cut": cut, "coefficient": coefficient}
                for (arm, cut), coefficient in contrasts[name].items()
            ],
            "per_horizon": {},
            "per_model_seed": {
                seed: float(point[name][:, index].mean())
                for index, seed in enumerate(model_seed_order)
            },
        }
        for index, horizon in enumerate(horizon_order):
            horizon_values = draws[name][:, index, :].mean(axis=1)
            result["per_horizon"][horizon] = {
                "estimate": float(point[name][index].mean()),
                "ci_low": float(np.quantile(horizon_values, 0.025)),
                "ci_high": float(np.quantile(horizon_values, 0.975)),
            }
        if include_replicates:
            result["replicate_estimates"] = values.tolist()
            result["replicate_horizon_seed_estimates"] = draws[name].tolist()
        output["contrasts"][name] = result
    return output
