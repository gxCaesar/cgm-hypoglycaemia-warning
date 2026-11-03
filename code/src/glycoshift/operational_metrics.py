from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import _collapsed_alarm_rows, _event_table, _match_alarms


@dataclass(frozen=True)
class OperationalEndpointSpec:
    """Columns and fixed event semantics for one operational endpoint."""

    score_column: str
    label_column: str
    event_id_column: str
    event_onset_column: str
    active_column: str
    horizon_minutes: int
    source_order: tuple[str, ...]
    collapse_minutes: int = 30

    def validate(self) -> None:
        if self.horizon_minutes <= 0 or self.collapse_minutes <= 0:
            raise ValueError("horizon and collapse minutes must be positive")
        if not self.source_order or len(set(self.source_order)) != len(
            self.source_order
        ):
            raise ValueError("source_order must contain unique source names")
        columns = (
            self.score_column,
            self.label_column,
            self.event_id_column,
            self.event_onset_column,
            self.active_column,
        )
        if len(set(columns)) != len(columns):
            raise ValueError("operational endpoint columns must be distinct")


@dataclass(frozen=True)
class FrozenAlarmPolicy:
    """A calibration-only alarm policy with a content-addressed receipt."""

    schema_version: str
    partition: str
    endpoint: dict[str, Any]
    nominal_fapd: float
    threshold: float
    achieved_equal_source_participant_macro_fapd: float
    achieved_equal_source_participant_macro_event_sensitivity: float
    calibration_prediction_sha256: str
    score_transform: str
    participant_count: int
    event_positive_participant_count: int
    event_count: int
    patient_days: int
    policy_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required_columns(spec: OperationalEndpointSpec) -> set[str]:
    return {
        "source_id",
        "participant_id",
        "context_end",
        "patient_day",
        spec.score_column,
        spec.label_column,
        spec.event_id_column,
        spec.event_onset_column,
        spec.active_column,
    }


def _eligible_operational_frame(
    predictions: pd.DataFrame,
    *,
    spec: OperationalEndpointSpec,
) -> pd.DataFrame:
    spec.validate()
    missing = _required_columns(spec).difference(predictions.columns)
    if missing:
        raise ValueError(f"operational frame is missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("operational frame is empty")

    frame = predictions[list(_required_columns(spec))].copy()
    frame["source_id"] = frame["source_id"].astype(str)
    frame["participant_id"] = frame["participant_id"].astype(str)
    observed_sources = set(frame["source_id"])
    expected_sources = set(spec.source_order)
    if observed_sources != expected_sources:
        missing_sources = sorted(expected_sources.difference(observed_sources))
        unexpected_sources = sorted(observed_sources.difference(expected_sources))
        raise ValueError(
            "operational source set mismatch: "
            f"missing={missing_sources}, unexpected={unexpected_sources}"
        )
    if frame[spec.active_column].isna().any():
        raise ValueError("active-event status contains missing values")
    frame = frame.loc[~frame[spec.active_column].astype(bool)].copy()
    if frame.empty:
        raise ValueError("operational endpoint has no eligible decision rows")
    if set(frame["source_id"]) != expected_sources:
        raise ValueError("a source has no rows after current-event exclusion")

    frame["context_end"] = pd.to_datetime(frame["context_end"], utc=True)
    frame["patient_day"] = pd.to_datetime(frame["patient_day"], errors="coerce")
    if frame["context_end"].isna().any() or frame["patient_day"].isna().any():
        raise ValueError("operational timestamps or patient days are invalid")
    decision_columns = ["source_id", "participant_id", "context_end"]
    if frame.duplicated(decision_columns).any():
        raise ValueError("operational decision keys are not unique")

    scores = pd.to_numeric(frame[spec.score_column], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(scores).all():
        raise ValueError("operational scores must be finite")
    frame[spec.score_column] = scores

    labels = pd.to_numeric(frame[spec.label_column], errors="coerce")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError("operational labels must be binary and complete")
    frame[spec.label_column] = labels.astype(int)
    has_event = frame[spec.event_id_column].notna()
    has_onset = frame[spec.event_onset_column].notna()
    if not (has_event == labels.astype(bool)).all():
        raise ValueError("operational label and event ID are inconsistent")
    if not (has_onset == has_event).all():
        raise ValueError("operational event ID and onset are inconsistent")

    positives = frame.loc[has_event].copy()
    if not positives.empty:
        positives[spec.event_onset_column] = pd.to_datetime(
            positives[spec.event_onset_column],
            utc=True,
            errors="coerce",
        )
        if positives[spec.event_onset_column].isna().any():
            raise ValueError("operational event onset is invalid")
        # `observed=True` is load-bearing, not tidiness. These three keys arrive
        # as pandas categoricals whenever the frame descends from window shards,
        # because `_compact_window_table` dictionary-encodes source, participant
        # and every event ID column. A grouped *aggregation* over categoricals
        # reindexes its result to the full cartesian product of categories, so
        # absent combinations appear with `nunique() == 0` and this check
        # rejects clean data. Measured on one calibration slice: 545 real events
        # became 95,375 groups, 94,830 of them empty, none with more than one
        # onset. Iterating a groupby does not do this -- only aggregating does,
        # which is why the participant loops below are unaffected.
        onset_counts = positives.groupby(
            ["source_id", "participant_id", spec.event_id_column],
            dropna=False,
            observed=True,
        )[spec.event_onset_column].nunique()
        if (onset_counts != 1).any():
            raise ValueError("one event ID maps to multiple event onsets")

    return frame.sort_values(decision_columns).reset_index(drop=True)


def _participant_alarm_table_from_eligible(
    frame: pd.DataFrame,
    *,
    spec: OperationalEndpointSpec,
    threshold: float,
) -> pd.DataFrame:
    if not np.isfinite(threshold):
        raise ValueError("alarm threshold must be finite")
    rows: list[dict[str, Any]] = []
    for (source_id, participant_id), participant in frame.groupby(
        ["source_id", "participant_id"],
        sort=False,
    ):
        alarms = _collapsed_alarm_rows(
            participant,
            probability_column=spec.score_column,
            threshold=float(threshold),
            collapse_minutes=spec.collapse_minutes,
        )
        events = _event_table(
            participant,
            event_id_column=spec.event_id_column,
            event_onset_column=spec.event_onset_column,
        )
        matched, false_alarms, _ = _match_alarms(
            alarms,
            events,
            horizon_minutes=spec.horizon_minutes,
        )
        patient_days = int(participant["patient_day"].nunique())
        if patient_days <= 0:
            raise ValueError("eligible participant has zero patient-days")
        event_count = int(len(events))
        detected = int(len(matched))
        rows.append(
            {
                "source_id": str(source_id),
                "participant_id": str(participant_id),
                "event_count": event_count,
                "detected_event_count": detected,
                "false_alarm_count": int(false_alarms),
                "alarm_count": int(len(alarms)),
                "patient_days": patient_days,
                "event_sensitivity": (
                    float(detected / event_count)
                    if event_count > 0
                    else float("nan")
                ),
                "false_alarms_per_patient_day": float(
                    false_alarms / patient_days
                ),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("operational endpoint has no eligible participants")
    return table


def participant_alarm_table(
    predictions: pd.DataFrame,
    *,
    spec: OperationalEndpointSpec,
    threshold: float,
) -> pd.DataFrame:
    """Return fixed-threshold event and burden counts for every participant."""

    frame = _eligible_operational_frame(predictions, spec=spec)
    return _participant_alarm_table_from_eligible(
        frame,
        spec=spec,
        threshold=threshold,
    )


def _summarize_participant_alarm_table(
    table: pd.DataFrame,
    *,
    source_order: tuple[str, ...],
) -> dict[str, Any]:
    required = {
        "source_id",
        "participant_id",
        "event_count",
        "detected_event_count",
        "false_alarm_count",
        "alarm_count",
        "patient_days",
        "event_sensitivity",
        "false_alarms_per_patient_day",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"participant alarm table is missing: {sorted(missing)}")
    if table.duplicated(["source_id", "participant_id"]).any():
        raise ValueError("participant alarm table contains duplicate participants")
    if set(table["source_id"].astype(str)) != set(source_order):
        raise ValueError("participant alarm table source set is incomplete")

    per_source: dict[str, dict[str, Any]] = {}
    source_sensitivity: list[float] = []
    source_fapd: list[float] = []
    for source in source_order:
        part = table.loc[table["source_id"].astype(str) == source]
        if part.empty:
            raise ValueError(f"source has no participant rows: {source}")
        event_positive = part.loc[part["event_count"] > 0]
        if event_positive.empty:
            raise ValueError(
                f"source has no event-positive participant: {source}"
            )
        sensitivity_values = event_positive["event_sensitivity"].to_numpy(
            dtype=float
        )
        fapd_values = part["false_alarms_per_patient_day"].to_numpy(dtype=float)
        if not np.isfinite(sensitivity_values).all() or not np.isfinite(
            fapd_values
        ).all():
            raise ValueError(f"source has a non-finite operational metric: {source}")
        sensitivity = float(sensitivity_values.mean())
        fapd = float(fapd_values.mean())
        source_sensitivity.append(sensitivity)
        source_fapd.append(fapd)
        per_source[source] = {
            "participant_macro_event_sensitivity": sensitivity,
            "participant_macro_false_alarms_per_patient_day": fapd,
            "participant_count": int(len(part)),
            "event_positive_participant_count": int(len(event_positive)),
            "event_count": int(part["event_count"].sum()),
            "detected_event_count": int(part["detected_event_count"].sum()),
            "alarm_count": int(part["alarm_count"].sum()),
            "false_alarm_count": int(part["false_alarm_count"].sum()),
            "patient_days": int(part["patient_days"].sum()),
        }
    return {
        "equal_source_participant_macro_event_sensitivity": float(
            np.mean(source_sensitivity)
        ),
        "equal_source_participant_macro_false_alarms_per_patient_day": float(
            np.mean(source_fapd)
        ),
        "source_count": int(len(source_order)),
        "participant_count": int(len(table)),
        "event_positive_participant_count": int((table["event_count"] > 0).sum()),
        "event_count": int(table["event_count"].sum()),
        "detected_event_count": int(table["detected_event_count"].sum()),
        "alarm_count": int(table["alarm_count"].sum()),
        "false_alarm_count": int(table["false_alarm_count"].sum()),
        "patient_days": int(table["patient_days"].sum()),
        "per_source": per_source,
    }


def operational_endpoint_metrics(
    predictions: pd.DataFrame,
    *,
    spec: OperationalEndpointSpec,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate the frozen source-then-participant macro endpoint."""

    table = participant_alarm_table(
        predictions,
        spec=spec,
        threshold=threshold,
    )
    output = _summarize_participant_alarm_table(
        table,
        source_order=spec.source_order,
    )
    output["threshold"] = float(threshold)
    return output


def _candidate_thresholds(
    scores: np.ndarray,
    *,
    candidate_count: int | None,
) -> np.ndarray:
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("threshold fitting requires finite non-empty scores")
    no_alarm = np.nextafter(float(scores.max()), np.inf)
    if not np.isfinite(no_alarm):
        raise ValueError("cannot construct a finite no-alarm threshold")
    if candidate_count is None:
        candidates = np.unique(scores)
    else:
        if candidate_count < 2:
            raise ValueError("candidate_count must be at least two")
        candidates = np.unique(
            np.quantile(scores, np.linspace(0.0, 1.0, candidate_count))
        )
    return np.unique(np.concatenate((candidates, np.array([no_alarm]))))


class _FenwickSet:
    """Dynamic ordered integer set with logarithmic predecessor/successor."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("Fenwick set size must be positive")
        self.size = int(size)
        self.tree = np.zeros(self.size + 1, dtype=np.int64)
        self.present = np.zeros(self.size, dtype=bool)

    def add(self, index: int) -> None:
        if self.present[index]:
            raise ValueError("ordered-set index is already present")
        self.present[index] = True
        self._increment(index, 1)

    def remove(self, index: int) -> None:
        if not self.present[index]:
            raise ValueError("ordered-set index is not present")
        self.present[index] = False
        self._increment(index, -1)

    def _increment(self, index: int, delta: int) -> None:
        cursor = int(index) + 1
        while cursor <= self.size:
            self.tree[cursor] += delta
            cursor += cursor & -cursor

    def prefix_count(self, stop: int) -> int:
        """Number of present indices in ``[0, stop)``."""

        cursor = min(max(int(stop), 0), self.size)
        total = 0
        while cursor > 0:
            total += int(self.tree[cursor])
            cursor -= cursor & -cursor
        return total

    @property
    def count(self) -> int:
        return self.prefix_count(self.size)

    def _at_rank(self, rank: int) -> int:
        """Return the zero-based index at one-based occupied rank."""

        if rank <= 0 or rank > self.count:
            raise IndexError("ordered-set rank is outside the occupied set")
        cursor = 0
        step = 1 << (self.size.bit_length() - 1)
        remaining = int(rank)
        while step:
            candidate = cursor + step
            if candidate <= self.size and int(self.tree[candidate]) < remaining:
                remaining -= int(self.tree[candidate])
                cursor = candidate
            step >>= 1
        return cursor

    def predecessor(self, index: int) -> int | None:
        rank = self.prefix_count(index)
        return None if rank == 0 else self._at_rank(rank)

    def first_at_or_after(self, index: int) -> int | None:
        before = self.prefix_count(index)
        return None if before == self.count else self._at_rank(before + 1)


class _ParticipantOperationalSweep:
    """Maintain one participant's exact chronological refractory alarm set.

    Rows are activated in descending score order. Inserting a newly eligible
    decision can only change the greedy alarm sequence from that decision
    forward. The update walks active successors until it rejoins an unchanged
    selected alarm. This evaluates every unique threshold without rescanning
    every decision row at every threshold.
    """

    def __init__(
        self,
        participant: pd.DataFrame,
        *,
        spec: OperationalEndpointSpec,
        source_index: int,
    ) -> None:
        ordered = participant.sort_values("context_end")
        self.times = (
            pd.to_datetime(ordered["context_end"], utc=True)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        if len(self.times) == 0 or np.any(self.times[1:] <= self.times[:-1]):
            raise ValueError("participant decision times must be strictly increasing")
        events = _event_table(
            ordered,
            event_id_column=spec.event_id_column,
            event_onset_column=spec.event_onset_column,
        )
        self.event_onsets = (
            pd.to_datetime(events["event_onset"], utc=True)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        self.event_count = int(len(self.event_onsets))
        self.patient_days = int(ordered["patient_day"].nunique())
        if self.patient_days <= 0:
            raise ValueError("eligible participant has zero patient-days")
        self.source_index = int(source_index)
        self.collapse_ns = np.int64(spec.collapse_minutes) * 60 * 1_000_000_000
        self.horizon_ns = np.int64(spec.horizon_minutes) * 60 * 1_000_000_000
        self.active = _FenwickSet(len(self.times))
        self.selected = _FenwickSet(len(self.times))
        self.detected_event_count = 0
        self.sensitivity_contribution = 0.0
        self.fapd_contribution = 0.0

    @property
    def alarm_count(self) -> int:
        return self.selected.count

    def activate(self, local_index: int) -> bool:
        """Activate one row and return whether the collapsed alarm set changed."""

        index = int(local_index)
        self.active.add(index)
        predecessor = self.selected.predecessor(index)
        if (
            predecessor is not None
            and self.times[index] - self.times[predecessor] <= self.collapse_ns
        ):
            return False

        changed = False
        current = index
        while True:
            if not self.selected.present[current]:
                self.selected.add(current)
                changed = True
            successor = self.selected.first_at_or_after(current + 1)
            while (
                successor is not None
                and self.times[successor] - self.times[current] <= self.collapse_ns
            ):
                self.selected.remove(successor)
                changed = True
                successor = self.selected.first_at_or_after(current + 1)

            next_start = int(
                np.searchsorted(
                    self.times,
                    self.times[current] + self.collapse_ns,
                    side="right",
                )
            )
            next_active = self.active.first_at_or_after(next_start)
            if next_active is None or self.selected.present[next_active]:
                break
            current = next_active
        return changed

    def recompute_detected_events(self) -> int:
        """Count the same chronological one-to-one matches as the evaluator."""

        detected = 0
        last_alarm = -1
        for onset in self.event_onsets:
            lower = int(
                np.searchsorted(
                    self.times,
                    onset - self.horizon_ns,
                    side="left",
                )
            )
            alarm = self.selected.first_at_or_after(max(lower, last_alarm + 1))
            if alarm is not None and self.times[alarm] < onset:
                detected += 1
                last_alarm = alarm
        self.detected_event_count = detected
        return detected


def _operational_sweep_points(
    frame: pd.DataFrame,
    *,
    spec: OperationalEndpointSpec,
    candidates: np.ndarray,
    retain_all: bool = False,
) -> list[dict[str, float]]:
    """Evaluate candidate thresholds with an exact incremental participant sweep."""

    source_to_index = {
        source: index for index, source in enumerate(spec.source_order)
    }
    states: list[_ParticipantOperationalSweep] = []
    row_state = np.empty(len(frame), dtype=np.int64)
    row_local = np.empty(len(frame), dtype=np.int64)
    participant_counts = np.zeros(len(spec.source_order), dtype=np.int64)
    event_positive_counts = np.zeros(len(spec.source_order), dtype=np.int64)
    offset = 0
    for (source, _participant), participant in frame.groupby(
        ["source_id", "participant_id"],
        sort=False,
    ):
        source_index = source_to_index[str(source)]
        state_index = len(states)
        state = _ParticipantOperationalSweep(
            participant,
            spec=spec,
            source_index=source_index,
        )
        states.append(state)
        count = len(participant)
        row_state[offset : offset + count] = state_index
        row_local[offset : offset + count] = np.arange(count, dtype=np.int64)
        participant_counts[source_index] += 1
        event_positive_counts[source_index] += int(state.event_count > 0)
        offset += count
    if offset != len(frame):
        raise AssertionError("participant sweep did not map every eligible row")
    if np.any(participant_counts == 0):
        raise ValueError("operational participant sweep is missing a source")
    missing_event_sources = np.flatnonzero(event_positive_counts == 0)
    if len(missing_event_sources):
        missing = [spec.source_order[index] for index in missing_event_sources]
        raise ValueError(f"source has no event-positive participant: {missing}")

    source_sensitivity_sums = np.zeros(len(spec.source_order), dtype=float)
    source_fapd_sums = np.zeros(len(spec.source_order), dtype=float)
    scores = frame[spec.score_column].to_numpy(dtype=float)
    activation_order = np.argsort(-scores, kind="stable")
    descending_candidates = np.sort(np.asarray(candidates, dtype=float))[::-1]
    cursor = 0
    points: list[dict[str, float]] = []
    for threshold in descending_candidates:
        dirty: set[int] = set()
        while cursor < len(activation_order):
            global_index = int(activation_order[cursor])
            if scores[global_index] < threshold:
                break
            state_index = int(row_state[global_index])
            state = states[state_index]
            if state.activate(int(row_local[global_index])):
                dirty.add(state_index)
            cursor += 1
        for state_index in dirty:
            state = states[state_index]
            source_index = state.source_index
            old_sensitivity = state.sensitivity_contribution
            old_fapd = state.fapd_contribution
            detected = state.recompute_detected_events()
            sensitivity = (
                detected / state.event_count if state.event_count > 0 else 0.0
            )
            false_alarms = state.alarm_count - detected
            fapd = false_alarms / state.patient_days
            source_sensitivity_sums[source_index] += sensitivity - old_sensitivity
            source_fapd_sums[source_index] += fapd - old_fapd
            state.sensitivity_contribution = sensitivity
            state.fapd_contribution = fapd
        sensitivity = float(
            np.mean(source_sensitivity_sums / event_positive_counts)
        )
        fapd = float(np.mean(source_fapd_sums / participant_counts))
        if not np.isfinite(sensitivity) or not np.isfinite(fapd):
            raise ValueError("incremental operational sweep became non-finite")
        point = {
            "threshold": float(threshold),
            "achieved_fapd": fapd,
            "event_sensitivity": sensitivity,
        }
        if (
            retain_all
            or not points
            or point["achieved_fapd"] != points[-1]["achieved_fapd"]
            or point["event_sensitivity"] != points[-1]["event_sensitivity"]
        ):
            points.append(point)
    # Preserve the public sweep's historical ascending-threshold order even
    # though activation necessarily proceeds from the no-alarm sentinel down.
    return list(reversed(points))


def _prediction_content_sha256(
    frame: pd.DataFrame,
    *,
    spec: OperationalEndpointSpec,
) -> str:
    columns = sorted(_required_columns(spec))
    ordered = frame[columns].copy().sort_values(
        ["source_id", "participant_id", "context_end"]
    )
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy(
        dtype=np.uint64
    )
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def fit_operational_alarm_policy(
    calibration_predictions: pd.DataFrame,
    *,
    spec: OperationalEndpointSpec,
    nominal_fapd: float,
    candidate_count: int | None = 2001,
    score_transform: str = "identity",
    partition: str = "calibration",
) -> FrozenAlarmPolicy:
    """Fit one feasible severe-event alarm threshold on calibration only."""

    if partition != "calibration":
        raise ValueError("operational alarm policy must be fit on calibration")
    if not np.isfinite(nominal_fapd) or nominal_fapd < 0:
        raise ValueError("nominal_fapd must be finite and non-negative")
    frame = _eligible_operational_frame(calibration_predictions, spec=spec)
    candidates = _candidate_thresholds(
        frame[spec.score_column].to_numpy(dtype=float),
        candidate_count=candidate_count,
    )
    points = _operational_sweep_points(frame, spec=spec, candidates=candidates)
    feasible = [
        (point["event_sensitivity"], point["threshold"])
        for point in points
        if point["achieved_fapd"] <= nominal_fapd
    ]
    if not feasible:
        raise AssertionError("no-alarm sentinel did not produce a feasible policy")
    best_sensitivity = max(item[0] for item in feasible)
    sensitivity, threshold = max(
        (item for item in feasible if item[0] == best_sensitivity),
        key=lambda item: item[1],
    )
    table = _participant_alarm_table_from_eligible(
        frame,
        spec=spec,
        threshold=float(threshold),
    )
    metrics = _summarize_participant_alarm_table(
        table,
        source_order=spec.source_order,
    )
    if not np.isclose(
        sensitivity,
        metrics["equal_source_participant_macro_event_sensitivity"],
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("incremental policy sensitivity disagrees with evaluator")
    selected_point = next(
        point for point in points if point["threshold"] == threshold
    )
    if not np.isclose(
        selected_point["achieved_fapd"],
        metrics["equal_source_participant_macro_false_alarms_per_patient_day"],
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("incremental policy FAPD disagrees with evaluator")
    endpoint = asdict(spec)
    endpoint["source_order"] = list(spec.source_order)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "partition": partition,
        "endpoint": endpoint,
        "nominal_fapd": float(nominal_fapd),
        "threshold": threshold,
        "achieved_equal_source_participant_macro_fapd": float(
            metrics[
                "equal_source_participant_macro_false_alarms_per_patient_day"
            ]
        ),
        "achieved_equal_source_participant_macro_event_sensitivity": sensitivity,
        "calibration_prediction_sha256": _prediction_content_sha256(
            frame,
            spec=spec,
        ),
        "score_transform": str(score_transform),
        "participant_count": int(metrics["participant_count"]),
        "event_positive_participant_count": int(
            metrics["event_positive_participant_count"]
        ),
        "event_count": int(metrics["event_count"]),
        "patient_days": int(metrics["patient_days"]),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["policy_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
    return FrozenAlarmPolicy(**payload)


def selection_operational_curve(
    selection_predictions: pd.DataFrame,
    *,
    spec: OperationalEndpointSpec,
    fapd_min: float = 0.0,
    fapd_max: float = 2.0,
    candidate_count: int | None = None,
    include_raw_points: bool = False,
) -> dict[str, Any]:
    """Compute the exact-logit operational budget frontier and its raw AUC.

    The frontier at budget ``b`` is the best sensitivity among thresholds with
    achieved FAPD no greater than ``b``. The AUC is the right-continuous step
    integral of that feasibility envelope. This defines duplicate and
    non-monotone achieved-FAPD points without optimistic interpolation.
    """

    if not (np.isfinite(fapd_min) and np.isfinite(fapd_max)):
        raise ValueError("FAPD bounds must be finite")
    if fapd_min < 0 or fapd_max <= fapd_min:
        raise ValueError("FAPD bounds must satisfy 0 <= min < max")
    frame = _eligible_operational_frame(selection_predictions, spec=spec)
    candidates = _candidate_thresholds(
        frame[spec.score_column].to_numpy(dtype=float),
        candidate_count=candidate_count,
    )
    raw_points = _operational_sweep_points(
        frame,
        spec=spec,
        candidates=candidates,
        retain_all=include_raw_points,
    )

    by_fapd: dict[float, float] = {}
    for point in raw_points:
        achieved = point["achieved_fapd"]
        sensitivity = point["event_sensitivity"]
        by_fapd[achieved] = max(by_fapd.get(achieved, -np.inf), sensitivity)
    ordered = sorted(by_fapd.items())
    running = -np.inf
    frontier: list[dict[str, float]] = []
    for achieved, sensitivity in ordered:
        running = max(running, sensitivity)
        frontier.append(
            {
                "achieved_fapd": float(achieved),
                "best_feasible_event_sensitivity": float(running),
            }
        )
    if not frontier or frontier[0]["achieved_fapd"] > fapd_min:
        raise AssertionError("operational curve is missing the no-alarm point")

    breakpoints = [fapd_min]
    breakpoints.extend(
        point["achieved_fapd"]
        for point in frontier
        if fapd_min < point["achieved_fapd"] < fapd_max
    )
    breakpoints.append(fapd_max)

    def best_at(budget: float) -> float:
        feasible = [
            point["best_feasible_event_sensitivity"]
            for point in frontier
            if point["achieved_fapd"] <= budget
        ]
        return max(feasible) if feasible else 0.0

    area = 0.0
    for left, right in zip(breakpoints[:-1], breakpoints[1:]):  # noqa: B905
        area += (right - left) * best_at(left)
    width = fapd_max - fapd_min
    in_range = [
        point
        for point in raw_points
        if fapd_min <= point["achieved_fapd"] <= fapd_max
    ]
    if not in_range:
        raise AssertionError("operational curve has no point in the frozen range")
    best_sensitivity = max(point["event_sensitivity"] for point in in_range)
    fapd_at_best = min(
        point["achieved_fapd"]
        for point in in_range
        if point["event_sensitivity"] == best_sensitivity
    )
    output = {
        "raw_auc": float(area),
        "normalized_auc": float(area / width),
        "best_sensitivity_within_range": float(best_sensitivity),
        "fapd_at_best_sensitivity": float(fapd_at_best),
        "fapd_min": float(fapd_min),
        "fapd_max": float(fapd_max),
        "threshold_count": int(len(candidates)),
        "integration_rule": (
            "right_continuous_step_integral_of_best_feasible_sensitivity"
        ),
        "sweep_algorithm": "incremental_exact_greedy_refractory_v1",
        "candidate_rule": (
            "all_unique_scores_plus_no_alarm_sentinel"
            if candidate_count is None
            else f"{candidate_count}_quantiles_plus_no_alarm_sentinel"
        ),
        "frontier": frontier,
    }
    if include_raw_points:
        output["raw_points"] = raw_points
    return output


def source_stratified_paired_operational_bootstrap_difference(
    candidate_table: pd.DataFrame,
    reference_table: pd.DataFrame,
    *,
    source_order: tuple[str, ...],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Paired participant bootstrap for the frozen sensitivity contrast.

    Alarm matching and fixed-threshold counts are computed before this call.
    Each source is resampled independently, and candidate/reference receive the
    same participant multiplicities. Because sensitivity is defined only among
    event-positive participants, each source's event-positive participant set
    is the bootstrap population. Zero-event participants remain load-bearing
    for alarm-burden analyses but are not pseudo-replicates for sensitivity.
    """

    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    keys = ["source_id", "participant_id"]
    alignment = [*keys, "event_count", "patient_days"]
    for name, table in (("candidate", candidate_table), ("reference", reference_table)):
        missing = set(alignment + ["event_sensitivity"]).difference(table.columns)
        if missing:
            raise ValueError(f"{name} participant table is missing: {sorted(missing)}")
        if table.duplicated(keys).any():
            raise ValueError(f"{name} participant table has duplicate keys")
    aligned = candidate_table.merge(
        reference_table,
        on=keys,
        how="outer",
        suffixes=("_candidate", "_reference"),
        indicator=True,
        validate="one_to_one",
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("paired bootstrap requires identical participants")
    for column in ("event_count", "patient_days"):
        if not aligned[f"{column}_candidate"].eq(
            aligned[f"{column}_reference"]
        ).all():
            raise ValueError(f"paired bootstrap outcome mismatch: {column}")
    if set(aligned["source_id"].astype(str)) != set(source_order):
        raise ValueError("paired bootstrap source set is incomplete")

    def contrast(frame: pd.DataFrame) -> float:
        source_differences: list[float] = []
        for source in source_order:
            part = frame.loc[frame["source_id"].astype(str) == source]
            positive = part.loc[part["event_count_candidate"] > 0]
            if positive.empty:
                raise ValueError(
                    "bootstrap replicate has no event-positive participant "
                    f"for source: {source}"
                )
            candidate = positive["event_sensitivity_candidate"].to_numpy(
                dtype=float
            )
            reference = positive["event_sensitivity_reference"].to_numpy(
                dtype=float
            )
            if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
                raise ValueError("bootstrap sensitivity contains non-finite values")
            source_differences.append(float(candidate.mean() - reference.mean()))
        return float(np.mean(source_differences))

    estimate = contrast(aligned)
    rng = np.random.default_rng(seed)
    draws_by_source: dict[str, pd.DataFrame] = {}
    for source in source_order:
        source_frame = aligned.loc[
            (aligned["source_id"].astype(str) == source)
            & (aligned["event_count_candidate"] > 0)
        ].reset_index(drop=True)
        if source_frame.empty:
            raise ValueError(
                "paired bootstrap has no event-positive participant "
                f"for source: {source}"
            )
        draws_by_source[source] = source_frame
    differences: list[float] = []
    for _ in range(replicates):
        sampled_sources: list[pd.DataFrame] = []
        for source in source_order:
            source_frame = draws_by_source[source]
            sampled = rng.choice(
                len(source_frame),
                size=len(source_frame),
                replace=True,
            )
            sampled_sources.append(source_frame.iloc[sampled])
        difference = contrast(pd.concat(sampled_sources, ignore_index=True))
        if not np.isfinite(difference):
            raise ValueError("bootstrap replicate produced a non-finite contrast")
        differences.append(difference)
    values = np.asarray(differences, dtype=float)
    lower_tail = (float(np.sum(values <= 0)) + 1) / (len(values) + 1)
    upper_tail = (float(np.sum(values >= 0)) + 1) / (len(values) + 1)
    return {
        "estimate": float(estimate),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_value_two_sided": float(min(1.0, 2 * min(lower_tail, upper_tail))),
        "replicates": float(replicates),
        "seed": float(seed),
    }


def source_stratified_paired_multiseed_operational_bootstrap_difference(
    candidate_tables: Mapping[int, pd.DataFrame],
    reference_tables: Mapping[int, pd.DataFrame],
    *,
    source_order: tuple[str, ...],
    model_seed_order: tuple[int, ...],
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Bootstrap the frozen mean contrast across a complete model-seed set.

    One event-positive participant draw is sampled within each source and is
    reused for candidate and reference under every model seed. Each draw first
    computes the equal-source contrast for each model seed, then takes the
    arithmetic mean over ``model_seed_order``. Model seeds are fixed design
    replicates and are never concatenated into the participant population.
    """

    if not source_order or len(set(source_order)) != len(source_order):
        raise ValueError("source_order must contain unique source names")
    if not model_seed_order or len(set(model_seed_order)) != len(
        model_seed_order
    ):
        raise ValueError("model_seed_order must contain unique model seeds")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    expected_seeds = set(model_seed_order)
    if set(candidate_tables) != expected_seeds:
        raise ValueError("candidate tables do not exactly cover model_seed_order")
    if set(reference_tables) != expected_seeds:
        raise ValueError("reference tables do not exactly cover model_seed_order")

    keys = ["source_id", "participant_id"]
    alignment = [*keys, "event_count", "patient_days"]
    aligned_by_seed: dict[int, pd.DataFrame] = {}
    truth_by_key: pd.DataFrame | None = None
    for model_seed in model_seed_order:
        candidate = candidate_tables[model_seed].copy()
        reference = reference_tables[model_seed].copy()
        for name, table in (("candidate", candidate), ("reference", reference)):
            missing = set(alignment + ["event_sensitivity"]).difference(
                table.columns
            )
            if missing:
                raise ValueError(
                    f"{name} seed {model_seed} participant table is missing: "
                    f"{sorted(missing)}"
                )
            table["source_id"] = table["source_id"].astype(str)
            table["participant_id"] = table["participant_id"].astype(str)
            if table.duplicated(keys).any():
                raise ValueError(
                    f"{name} seed {model_seed} participant table has duplicate keys"
                )
        aligned = candidate.merge(
            reference,
            on=keys,
            how="outer",
            suffixes=("_candidate", "_reference"),
            indicator=True,
            validate="one_to_one",
        )
        if not aligned["_merge"].eq("both").all():
            raise ValueError(
                "multiseed paired bootstrap requires identical participants "
                f"for model seed {model_seed}"
            )
        for column in ("event_count", "patient_days"):
            if not aligned[f"{column}_candidate"].eq(
                aligned[f"{column}_reference"]
            ).all():
                raise ValueError(
                    "multiseed paired bootstrap outcome mismatch: "
                    f"{column} at model seed {model_seed}"
                )
        if set(aligned["source_id"].astype(str)) != set(source_order):
            raise ValueError(
                "multiseed paired bootstrap source set is incomplete at "
                f"model seed {model_seed}"
            )
        aligned = aligned.sort_values(keys).reset_index(drop=True)
        current_truth = aligned[
            [*keys, "event_count_candidate", "patient_days_candidate"]
        ].rename(
            columns={
                "event_count_candidate": "event_count",
                "patient_days_candidate": "patient_days",
            }
        )
        if truth_by_key is None:
            truth_by_key = current_truth
        elif not current_truth.equals(truth_by_key):
            raise ValueError(
                "multiseed paired bootstrap truth or participant order differs "
                "across model seeds"
            )
        for side in ("candidate", "reference"):
            positives = aligned.loc[aligned["event_count_candidate"] > 0]
            values = positives[f"event_sensitivity_{side}"].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    "multiseed paired bootstrap sensitivity contains non-finite "
                    f"values for {side} seed {model_seed}"
                )
        aligned_by_seed[model_seed] = aligned

    if truth_by_key is None:
        raise AssertionError("validated model seed set is unexpectedly empty")
    positive_indices_by_source: dict[str, np.ndarray] = {}
    base = aligned_by_seed[model_seed_order[0]]
    for source in source_order:
        indices = base.index[
            (base["source_id"] == source) & (base["event_count_candidate"] > 0)
        ].to_numpy(dtype=int)
        if len(indices) == 0:
            raise ValueError(
                "multiseed paired bootstrap has no event-positive participant "
                f"for source: {source}"
            )
        positive_indices_by_source[source] = indices

    def contrast(sampled_indices: Mapping[str, np.ndarray]) -> float:
        seed_contrasts: list[float] = []
        for model_seed in model_seed_order:
            aligned = aligned_by_seed[model_seed]
            source_contrasts: list[float] = []
            for source in source_order:
                sampled = aligned.iloc[sampled_indices[source]]
                candidate = sampled["event_sensitivity_candidate"].to_numpy(
                    dtype=float
                )
                reference = sampled["event_sensitivity_reference"].to_numpy(
                    dtype=float
                )
                source_contrasts.append(float(candidate.mean() - reference.mean()))
            seed_contrasts.append(float(np.mean(source_contrasts)))
        return float(np.mean(seed_contrasts))

    identity_draw = {
        source: indices.copy()
        for source, indices in positive_indices_by_source.items()
    }
    estimate = contrast(identity_draw)
    rng = np.random.default_rng(bootstrap_seed)
    differences: list[float] = []
    for _ in range(replicates):
        draw = {
            source: rng.choice(indices, size=len(indices), replace=True)
            for source, indices in positive_indices_by_source.items()
        }
        difference = contrast(draw)
        if not np.isfinite(difference):
            raise ValueError("multiseed bootstrap produced a non-finite contrast")
        differences.append(difference)
    values = np.asarray(differences, dtype=float)
    lower_tail = (float(np.sum(values <= 0)) + 1) / (len(values) + 1)
    upper_tail = (float(np.sum(values >= 0)) + 1) / (len(values) + 1)
    return {
        "estimate": float(estimate),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_value_two_sided": float(min(1.0, 2 * min(lower_tail, upper_tail))),
        "replicates": int(replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "model_seeds": list(model_seed_order),
        "model_seed_count": len(model_seed_order),
        "resampling_unit": "event_positive_source_scoped_participant",
        "seed_aggregation": "arithmetic_mean_within_each_bootstrap_draw",
    }
