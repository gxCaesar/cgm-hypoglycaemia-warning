from __future__ import annotations

import multiprocessing
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .calibration import conformal_prediction_set


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p-values while preserving the input keys."""
    finite = [
        (name, float(value))
        for name, value in p_values.items()
        if np.isfinite(value)
    ]
    finite.sort(key=lambda item: item[1])
    adjusted: dict[str, float] = {
        name: float("nan")
        for name, value in p_values.items()
        if not np.isfinite(value)
    }
    running_maximum = 0.0
    comparison_count = len(finite)
    for rank, (name, value) in enumerate(finite):
        candidate = min(1.0, (comparison_count - rank) * value)
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = running_maximum
    return {name: adjusted[name] for name in p_values}


def conformal_classification_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(list(labels), dtype=int)
    y_score = np.asarray(list(probabilities), dtype=float)
    sets = conformal_prediction_set(y_score, threshold)
    sizes = sets.sum(axis=1)
    included = sets[np.arange(len(y_true)), y_true]
    singleton = sizes == 1
    predicted = sets[:, 1].astype(int)
    singleton_risk = (
        float(np.mean(predicted[singleton] != y_true[singleton]))
        if singleton.any()
        else float("nan")
    )
    return {
        "nominal_coverage": 0.90,
        "empirical_coverage": float(included.mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_fraction": float(singleton.mean()),
        "singleton_risk": singleton_risk,
        "empty_set_fraction": float((sizes == 0).mean()),
    }


def selective_risk_curve(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    coverage_levels: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 1.0),
) -> list[dict[str, float]]:
    y_true = np.asarray(list(labels), dtype=int)
    y_score = np.asarray(list(probabilities), dtype=float)
    if len(y_true) == 0:
        return []
    certainty = np.abs(y_score - 0.5)
    order = np.argsort(-certainty, kind="stable")
    predicted = (y_score >= 0.5).astype(int)
    output: list[dict[str, float]] = []
    for requested in coverage_levels:
        count = min(len(y_true), max(1, int(np.ceil(requested * len(y_true)))))
        retained = order[:count]
        output.append(
            {
                "coverage": float(count / len(y_true)),
                "classification_risk": float(
                    np.mean(predicted[retained] != y_true[retained])
                ),
            }
        )
    return output


def regression_metrics(
    targets: Iterable[float],
    predictions: Iterable[float],
) -> dict[str, float]:
    y_true = np.asarray(list(targets), dtype=float)
    y_pred = np.asarray(list(predictions), dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0.0}
    error = y_pred[valid] - y_true[valid]
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "n": float(valid.sum()),
    }


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 0:
        return float("nan")
    quantiles = np.unique(np.quantile(probabilities, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 2:
        return float(abs(labels.mean() - probabilities.mean()))
    assignments = np.digitize(probabilities, quantiles[1:-1], right=True)
    ece = 0.0
    for bin_index in range(len(quantiles) - 1):
        mask = assignments == bin_index
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(ece)


def calibration_slope_intercept(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    coefficients = np.zeros(2, dtype=float)
    for _ in range(100):
        linear = np.clip(
            coefficients[0] + coefficients[1] * logits,
            -50.0,
            50.0,
        )
        fitted = np.empty_like(linear)
        positive = linear >= 0
        fitted[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
        exponential = np.exp(linear[~positive])
        fitted[~positive] = exponential / (1.0 + exponential)
        residual = labels - fitted
        weights = np.maximum(fitted * (1.0 - fitted), 1e-10)
        score = np.array(
            [
                residual.sum(),
                np.sum(logits * residual),
            ]
        )
        information = np.array(
            [
                [weights.sum(), np.sum(weights * logits)],
                [
                    np.sum(weights * logits),
                    np.sum(weights * logits * logits),
                ],
            ]
        )
        information.flat[::3] += 1e-8
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        if not np.isfinite(step).all():
            return float("nan"), float("nan")
        step = np.clip(step, -5.0, 5.0)
        coefficients += step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    intercept, slope = coefficients
    return float(slope), float(intercept)


def _collapsed_alarm_rows_oracle(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    event_id_column: str | None = None,
    threshold: float,
    collapse_minutes: int = 30,
) -> pd.DataFrame:
    """Reference implementation. Retained as the equivalence oracle.

    Correct but quadratic in practice: DataFrame.iterrows() rebuilds a Series
    per row, so a threshold near zero over a multi-million-row partition costs
    minutes. Kept unchanged so the vectorised path can be proved against it.
    """
    del event_id_column
    alarms = predictions.loc[predictions[probability_column] >= threshold].copy()
    if alarms.empty:
        return alarms
    alarms["context_end"] = pd.to_datetime(alarms["context_end"], utc=True)
    alarms = alarms.sort_values(["source_id", "participant_id", "context_end"])
    keep: list[int] = []
    last_kept_time: dict[tuple[str, str], pd.Timestamp] = {}
    collapse_interval = pd.Timedelta(minutes=collapse_minutes)
    for index, row in alarms.iterrows():
        key = (str(row["source_id"]), str(row["participant_id"]))
        current = pd.Timestamp(row["context_end"])
        previous = last_kept_time.get(key)
        if previous is None or current - previous > collapse_interval:
            keep.append(index)
            last_kept_time[key] = current
    return alarms.loc[keep].copy()


def _collapsed_alarm_rows(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    event_id_column: str | None = None,
    threshold: float,
    collapse_minutes: int = 30,
) -> pd.DataFrame:
    """First-alarm refractory collapse, identical in output to the oracle.

    The rule is greedy and order-dependent — keep an alarm only if it is more
    than ``collapse_minutes`` after the last *kept* alarm for that participant
    — so it cannot be expressed as a groupby. It can, however, be walked with
    ``searchsorted`` instead of row by row: from each kept alarm, jump straight
    to the first alarm beyond the refractory window. That makes the cost
    proportional to the number of alarms *kept* rather than the number
    scanned, which is the whole point at a low threshold.
    """
    del event_id_column
    alarms = predictions.loc[predictions[probability_column] >= threshold].copy()
    if alarms.empty:
        return alarms
    alarms["context_end"] = pd.to_datetime(alarms["context_end"], utc=True)
    alarms = alarms.sort_values(["source_id", "participant_id", "context_end"])

    source = alarms["source_id"].astype(str).to_numpy()
    participant = alarms["participant_id"].astype(str).to_numpy()
    times = alarms["context_end"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    starts, stops = _contiguous_blocks(source, participant)
    positions = _collapse_positions(
        times, starts, stops, collapse_minutes=collapse_minutes
    )
    return alarms.iloc[positions].copy()


def _contiguous_blocks(
    source: np.ndarray,
    participant: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Start/stop offsets of each participant block in a sorted array pair."""
    count = len(source)
    if count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    boundary = np.empty(count, dtype=bool)
    boundary[0] = True
    if count > 1:
        boundary[1:] = (source[1:] != source[:-1]) | (
            participant[1:] != participant[:-1]
        )
    starts = np.flatnonzero(boundary)
    stops = np.append(starts[1:], count)
    return starts, stops


def _collapse_positions(
    times: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    *,
    collapse_minutes: int,
) -> list[int]:
    """Greedy refractory collapse, returning positions into ``times``.

    ``times`` must be int64 nanoseconds, sorted within each block. Shared by
    the single-threshold DataFrame path and the sweep path so the two cannot
    drift apart.
    """
    interval_ns = np.int64(collapse_minutes) * 60 * 1_000_000_000
    positions: list[int] = []
    for start, stop in zip(starts, stops):  # noqa: B905
        block = times[start:stop]
        # One vectorised searchsorted for the whole block instead of one call
        # per kept alarm: jumps[i] is where the walk goes after keeping i.
        # Strictly greater than last_kept + interval, matching the oracle's
        # `current - previous > collapse_interval`.
        jumps = np.searchsorted(block, block + interval_ns, side="right")
        block_size = len(block)
        cursor = 0
        while cursor < block_size:
            positions.append(int(start + cursor))
            cursor = int(jumps[cursor])
    return positions


def _event_table(
    predictions: pd.DataFrame,
    *,
    event_id_column: str,
    event_onset_column: str | None,
) -> pd.DataFrame:
    positive = predictions.loc[predictions[event_id_column].notna()].copy()
    if positive.empty:
        return pd.DataFrame(
            columns=("source_id", "participant_id", "event_id", "event_onset")
        )
    positive["event_id"] = positive[event_id_column].astype(str)
    positive["context_end"] = pd.to_datetime(positive["context_end"], utc=True)
    if event_onset_column is not None and event_onset_column in positive:
        positive["event_onset"] = pd.to_datetime(
            positive[event_onset_column],
            utc=True,
        )
        if positive["event_onset"].isna().any():
            raise ValueError("event onset is missing for a labeled event window")
        aggregation = "first"
    else:
        positive["event_onset"] = positive["context_end"] + pd.Timedelta(minutes=5)
        aggregation = "max"
    events = (
        positive.groupby(
            ["source_id", "participant_id", "event_id"],
            as_index=False,
            # Without this, categorical keys make the aggregation reindex to the
            # full cartesian product of categories, and the event table gains
            # one fabricated row -- with a NaT onset -- for every absent
            # combination. Measured on one calibration slice: 545 real events
            # became 95,375 rows, 94,830 fabricated. Every caller reaching here
            # today passes stringified keys, so no reported number was affected,
            # but this table is the denominator of every event sensitivity and
            # the failure mode is silent inflation rather than an error.
            observed=True,
        )["event_onset"]
        .agg(aggregation)
        .sort_values(["source_id", "participant_id", "event_onset"])
    )
    return events


def _match_alarms(
    alarms: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_minutes: int,
) -> tuple[list[str], int, list[float]]:
    """One-to-one alarm/event matching, identical in output to the oracle.

    Same greedy rule: walk alarms in global ``context_end`` order and take,
    for each, the earliest not-yet-used event of the same participant whose
    onset falls in ``(alarm_time, alarm_time + horizon]``.

    The oracle rescans the entire event table per alarm, which is
    O(alarms x events). Here the event table is split into per-participant
    blocks once, and each alarm binary-searches its own block. The candidate
    span is bounded by the horizon, so in practice only a couple of events are
    examined per alarm.

    ``used`` stays a single global set of event ids, exactly as in the oracle.
    Event ids are only unique within a participant, so a repeated id across
    participants can block a match; that behaviour is preserved deliberately
    rather than silently corrected.
    """
    matched_event_ids: list[str] = []
    false_alarms = 0
    lead_minutes: list[float] = []
    if alarms.empty:
        return matched_event_ids, false_alarms, lead_minutes
    if events.empty:
        return matched_event_ids, int(len(alarms)), lead_minutes

    ordered = alarms.sort_values("context_end")
    alarm_source = ordered["source_id"].astype(str).to_numpy()
    alarm_participant = ordered["participant_id"].astype(str).to_numpy()
    alarm_times = (
        pd.to_datetime(ordered["context_end"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )

    # _event_table already sorts by (source_id, participant_id, event_onset),
    # so each participant is one contiguous block and each block is ordered.
    event_source = events["source_id"].astype(str).to_numpy()
    event_participant = events["participant_id"].astype(str).to_numpy()
    event_ids = events["event_id"].astype(str).to_numpy()
    event_onsets = (
        pd.to_datetime(events["event_onset"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )

    boundary = np.empty(len(events), dtype=bool)
    boundary[0] = True
    if len(events) > 1:
        boundary[1:] = (event_source[1:] != event_source[:-1]) | (
            event_participant[1:] != event_participant[:-1]
        )
    starts = np.flatnonzero(boundary)
    stops = np.append(starts[1:], len(events))
    blocks = {
        (event_source[start], event_participant[start]): (int(start), int(stop))
        for start, stop in zip(starts, stops)  # noqa: B905
    }

    horizon_ns = np.int64(horizon_minutes) * 60 * 1_000_000_000
    used: set[str] = set()

    for position in range(len(ordered)):
        block = blocks.get(
            (alarm_source[position], alarm_participant[position])
        )
        if block is None:
            false_alarms += 1
            continue
        start, stop = block
        onsets = event_onsets[start:stop]
        alarm_time = alarm_times[position]
        # onset > alarm_time, and onset <= alarm_time + horizon
        low = start + int(np.searchsorted(onsets, alarm_time, side="right"))
        high = start + int(
            np.searchsorted(onsets, alarm_time + horizon_ns, side="right")
        )
        chosen = -1
        for candidate in range(low, high):
            if event_ids[candidate] not in used:
                chosen = candidate
                break
        if chosen < 0:
            false_alarms += 1
            continue
        event_id = str(event_ids[chosen])
        used.add(event_id)
        matched_event_ids.append(event_id)
        lead_minutes.append(
            float((event_onsets[chosen] - alarm_time) / 60_000_000_000)
        )
    return matched_event_ids, false_alarms, lead_minutes


def _match_alarms_oracle(
    alarms: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_minutes: int,
) -> tuple[list[str], int, list[float]]:
    """Reference implementation. Retained as the equivalence oracle."""
    matched_event_ids: list[str] = []
    false_alarms = 0
    lead_minutes: list[float] = []
    used: set[str] = set()
    horizon = pd.Timedelta(minutes=horizon_minutes)
    for _, alarm in alarms.sort_values("context_end").iterrows():
        alarm_time = pd.Timestamp(alarm["context_end"])
        same_participant = events.loc[
            (events["source_id"].astype(str) == str(alarm["source_id"]))
            & (
                events["participant_id"].astype(str)
                == str(alarm["participant_id"])
            )
        ]
        candidates = same_participant.loc[
            (same_participant["event_onset"] > alarm_time)
            & (same_participant["event_onset"] <= alarm_time + horizon)
            & (~same_participant["event_id"].isin(used))
        ].sort_values("event_onset")
        if candidates.empty:
            false_alarms += 1
            continue
        event = candidates.iloc[0]
        event_id = str(event["event_id"])
        used.add(event_id)
        matched_event_ids.append(event_id)
        lead_minutes.append(
            float(
                (pd.Timestamp(event["event_onset"]) - alarm_time)
                / pd.Timedelta(minutes=1)
            )
        )
    return matched_event_ids, false_alarms, lead_minutes


def alarm_metrics(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    label_column: str,
    event_id_column: str,
    threshold: float,
    horizon_minutes: int | None = None,
    event_onset_column: str | None = None,
    collapse_minutes: int = 30,
    events: pd.DataFrame | None = None,
    patient_days: int | None = None,
) -> dict[str, float]:
    """Alarm-burden metrics at one probability threshold.

    ``events`` and ``patient_days`` do not depend on ``threshold``. Callers
    that sweep thresholds (select_alarm_threshold, event_average_precision)
    should compute them once with :func:`alarm_sweep_context` and pass them in;
    recomputing a groupby and a drop_duplicates over millions of rows per
    threshold dominated the entire evaluation path. Left as ``None`` the
    behaviour is unchanged.
    """
    del label_column
    if horizon_minutes is None:
        try:
            horizon_minutes = int(event_id_column.split("_")[1])
        except (IndexError, ValueError) as error:
            raise ValueError("horizon_minutes is required") from error
    alarms = _collapsed_alarm_rows(
        predictions,
        probability_column=probability_column,
        threshold=threshold,
        collapse_minutes=collapse_minutes,
    )
    if events is None:
        events = _event_table(
            predictions,
            event_id_column=event_id_column,
            event_onset_column=event_onset_column,
        )
    matched, false_alarms, lead_minutes = _match_alarms(
        alarms,
        events,
        horizon_minutes=horizon_minutes,
    )
    if patient_days is None:
        patient_days = max(
            1,
            len(
                predictions[
                    ["source_id", "participant_id", "patient_day"]
                ].drop_duplicates()
            ),
        )
    return _alarm_result(
        alarm_count=len(alarms),
        event_count=len(events),
        matched=len(matched),
        false_alarms=false_alarms,
        lead_minutes=lead_minutes,
        patient_days=patient_days,
    )


class AlarmThresholdSweep:
    """Evaluate alarm metrics at many thresholds over one prediction frame.

    ``alarm_metrics`` is correct but rebuilds everything per call: it masks and
    copies the frame, parses timestamps, and sorts millions of rows, none of
    which depends on the threshold. A 2001-candidate sweep therefore pays that
    cost 2001 times and does not finish at production scale.

    This precomputes the sorted int64 view once. Each threshold then costs a
    numpy mask plus a refractory walk over the rows that survive it, and no
    DataFrame is constructed at all.

    Output is identical to ``alarm_metrics`` by construction: the same
    ``_collapse_positions`` core, the same event blocks, and the same greedy
    global-time matching with one shared ``used`` set.
    """

    def __init__(
        self,
        predictions: pd.DataFrame,
        *,
        probability_column: str,
        event_id_column: str,
        event_onset_column: str | None,
        collapse_minutes: int = 30,
    ) -> None:
        self.collapse_minutes = int(collapse_minutes)

        source = predictions["source_id"].astype(str).to_numpy()
        participant = predictions["participant_id"].astype(str).to_numpy()
        times = (
            pd.to_datetime(predictions["context_end"], utc=True)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        probability = predictions[probability_column].to_numpy(dtype=float)

        # Lexicographic codes, so ordering matches the string sort the
        # single-threshold path performs.
        source_code = pd.factorize(source, sort=True)[0]
        participant_code = pd.factorize(participant, sort=True)[0]
        order = np.lexsort((times, participant_code, source_code))

        self._source_code = source_code[order]
        self._participant_code = participant_code[order]
        self._times = times[order]
        self._probability = probability[order]

        events, patient_days = alarm_sweep_context(
            predictions,
            event_id_column=event_id_column,
            event_onset_column=event_onset_column,
        )
        self.events = events
        self.patient_days = patient_days
        self.event_count = int(len(events))

        self._participant_stride = int(participant_code.max()) + 2 if len(
            participant_code
        ) else 2
        if events.empty:
            self._block_keys = np.empty(0, dtype=np.int64)
            self._block_starts = np.empty(0, dtype=np.int64)
            self._block_stops = np.empty(0, dtype=np.int64)
            self._event_onsets = np.empty(0, dtype=np.int64)
            self._event_ids = np.empty(0, dtype=object)
            return

        # Map the event table onto the same integer codes. Events whose
        # participant never appears in the predictions cannot be matched, and
        # get no block, exactly as the oracle's participant filter would.
        source_lookup = {value: code for code, value in enumerate(
            pd.factorize(source, sort=True)[1]
        )}
        participant_lookup = {value: code for code, value in enumerate(
            pd.factorize(participant, sort=True)[1]
        )}
        event_source = np.array(
            [source_lookup.get(value, -1) for value in events["source_id"].astype(str)],
            dtype=np.int64,
        )
        event_participant = np.array(
            [
                participant_lookup.get(value, -1)
                for value in events["participant_id"].astype(str)
            ],
            dtype=np.int64,
        )
        self._event_onsets = (
            pd.to_datetime(events["event_onset"], utc=True)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        self._event_ids = events["event_id"].astype(str).to_numpy()

        starts, stops = _contiguous_blocks(event_source, event_participant)
        keys = (
            event_source[starts] * self._participant_stride
            + event_participant[starts]
        )
        # Sorted keys so alarms can be resolved to blocks with one batched
        # searchsorted. Participants absent from the predictions get code -1
        # and simply never match an alarm key.
        key_order = np.argsort(keys, kind="stable")
        self._block_keys = keys[key_order]
        self._block_starts = starts[key_order].astype(np.int64)
        self._block_stops = stops[key_order].astype(np.int64)

    def metrics_at(
        self,
        threshold: float,
        *,
        horizon_minutes: int,
    ) -> dict[str, float]:
        selected = np.flatnonzero(self._probability >= threshold)
        if selected.size == 0:
            return _alarm_result(
                alarm_count=0,
                event_count=self.event_count,
                matched=0,
                false_alarms=0,
                lead_minutes=[],
                patient_days=self.patient_days,
            )

        source_code = self._source_code[selected]
        participant_code = self._participant_code[selected]
        times = self._times[selected]
        starts, stops = _contiguous_blocks(source_code, participant_code)
        keep = _collapse_positions(
            times, starts, stops, collapse_minutes=self.collapse_minutes
        )
        keep_index = np.asarray(keep, dtype=np.int64)
        alarm_source = source_code[keep_index]
        alarm_participant = participant_code[keep_index]
        alarm_times = times[keep_index]

        # The oracle matches alarms in global context_end order; a stable sort
        # keeps ties in participant order so the walk is deterministic.
        walk = np.argsort(alarm_times, kind="stable")
        matched, false_alarms, lead_minutes = self._match(
            alarm_source[walk],
            alarm_participant[walk],
            alarm_times[walk],
            horizon_minutes=horizon_minutes,
        )
        return _alarm_result(
            alarm_count=len(keep_index),
            event_count=self.event_count,
            matched=matched,
            false_alarms=false_alarms,
            lead_minutes=lead_minutes,
            patient_days=self.patient_days,
        )

    def _match(
        self,
        alarm_source: np.ndarray,
        alarm_participant: np.ndarray,
        alarm_times: np.ndarray,
        *,
        horizon_minutes: int,
    ) -> tuple[int, int, list[float]]:
        count = int(len(alarm_times))
        if self._block_keys.size == 0 or count == 0:
            return 0, count, []
        horizon_ns = np.int64(horizon_minutes) * 60 * 1_000_000_000

        # Resolve every alarm to its event block in one vectorised lookup.
        alarm_keys = alarm_source * self._participant_stride + alarm_participant
        slot = np.searchsorted(self._block_keys, alarm_keys)
        slot_clipped = np.minimum(slot, len(self._block_keys) - 1)
        has_block = (slot < len(self._block_keys)) & (
            self._block_keys[slot_clipped] == alarm_keys
        )

        # Candidate span per alarm, computed one BLOCK at a time rather than
        # one alarm at a time. There are ~1e3 blocks and ~1e6 alarms, so this
        # replaces two million scalar searchsorted calls with a few thousand
        # vectorised ones.
        low = np.zeros(count, dtype=np.int64)
        high = np.zeros(count, dtype=np.int64)
        candidates = np.flatnonzero(has_block)
        if candidates.size:
            block_of = slot_clipped[candidates]
            order = np.argsort(block_of, kind="stable")
            grouped = candidates[order]
            grouped_blocks = block_of[order]
            edges = np.flatnonzero(
                np.concatenate(
                    ([True], grouped_blocks[1:] != grouped_blocks[:-1])
                )
            )
            edges = np.append(edges, len(grouped))
            for edge in range(len(edges) - 1):
                members = grouped[edges[edge] : edges[edge + 1]]
                block = int(grouped_blocks[edges[edge]])
                start = int(self._block_starts[block])
                stop = int(self._block_stops[block])
                onsets = self._event_onsets[start:stop]
                times = alarm_times[members]
                low[members] = start + np.searchsorted(onsets, times, side="right")
                high[members] = start + np.searchsorted(
                    onsets, times + horizon_ns, side="right"
                )

        # An alarm with an empty candidate span is a false alarm no matter what
        # the `used` set contains, so it never needs to enter the greedy walk.
        # Events are sparse relative to alarms, so this removes the large
        # majority of iterations.
        pending = np.flatnonzero(high > low)
        false_alarms = count - int(pending.size)

        used: set[str] = set()
        matched = 0
        lead_minutes: list[float] = []
        event_ids = self._event_ids
        onsets_all = self._event_onsets
        for position in pending.tolist():
            chosen = -1
            for candidate in range(int(low[position]), int(high[position])):
                if event_ids[candidate] not in used:
                    chosen = candidate
                    break
            if chosen < 0:
                false_alarms += 1
                continue
            used.add(str(event_ids[chosen]))
            matched += 1
            lead_minutes.append(
                float(
                    (onsets_all[chosen] - alarm_times[position]) / 60_000_000_000
                )
            )
        return matched, false_alarms, lead_minutes


def _alarm_result(
    *,
    alarm_count: int,
    event_count: int,
    matched: int,
    false_alarms: int,
    lead_minutes: list[float],
    patient_days: int,
) -> dict[str, float]:
    sensitivity = matched / event_count if event_count else float("nan")
    precision = matched / alarm_count if alarm_count else float("nan")
    return {
        "event_sensitivity": float(sensitivity),
        "alarm_precision": float(precision),
        "false_alarms_per_patient_day": float(false_alarms / patient_days),
        "alarm_count": float(alarm_count),
        "event_count": float(event_count),
        "detected_event_count": float(matched),
        "median_lead_minutes": (
            float(np.median(lead_minutes)) if lead_minutes else float("nan")
        ),
    }


def alarm_sweep_context(
    predictions: pd.DataFrame,
    *,
    event_id_column: str,
    event_onset_column: str | None,
) -> tuple[pd.DataFrame, int]:
    """Precompute the threshold-independent inputs of :func:`alarm_metrics`.

    Both the event table and the patient-day count are functions of the
    prediction frame alone. A threshold sweep recomputes them once per
    candidate otherwise, which is thousands of groupby and drop_duplicates
    passes over the same millions of rows.
    """
    events = _event_table(
        predictions,
        event_id_column=event_id_column,
        event_onset_column=event_onset_column,
    )
    patient_days = max(
        1,
        len(
            predictions[
                ["source_id", "participant_id", "patient_day"]
            ].drop_duplicates()
        ),
    )
    return events, patient_days


def event_average_precision(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    label_column: str,
    event_id_column: str,
    event_onset_column: str | None,
    horizon_minutes: int,
    candidate_count: int = 201,
    workers: int = 1,
) -> float:
    """Event-level average precision over a candidate threshold sweep.

    ``workers`` routes the candidate loop through the same process pool as
    ``select_alarm_threshold``. The candidates are evaluated independently and
    ``AlarmThresholdSweep.metrics_at`` is a pure function of immutable
    precomputed arrays plus one threshold, so the parallel path returns the
    serial path's values bit for bit and in the same order.

    This matters for more than throughput. This function runs inside
    ``evaluate_predictions``, which is the CPU-only tail of every deep arm, and
    the tail is measured at 12 and 27 minutes on the two arms of the 2026-08-19
    campaign. During it the arm has released its GPU and the approved card sits
    empty, which on a contended host is the window a co-tenant takes it in.
    Shortening the tail shortens that window.
    """

    probabilities = predictions[probability_column].to_numpy(dtype=float)
    if len(probabilities) == 0:
        return float("nan")
    thresholds = np.unique(
        np.concatenate(
            (
                np.array([np.nextafter(probabilities.max(), np.inf), 0.0]),
                np.quantile(
                    probabilities,
                    np.linspace(0, 1, candidate_count),
                ),
            )
        )
    )
    del label_column
    sweep = AlarmThresholdSweep(
        predictions,
        probability_column=probability_column,
        event_id_column=event_id_column,
        event_onset_column=event_onset_column,
    )
    points: list[tuple[float, float]] = [(0.0, 1.0)]
    descending = thresholds[::-1]
    swept = _sweep_all_candidates(
        sweep,
        descending,
        horizon_minutes=horizon_minutes,
        workers=workers,
    )
    for metrics in swept:
        recall = metrics["event_sensitivity"]
        precision = metrics["alarm_precision"]
        if np.isfinite(recall):
            points.append(
                (
                    float(recall),
                    0.0 if not np.isfinite(precision) else float(precision),
                )
            )
    ordered = np.array(sorted(points, key=lambda point: point[0]), dtype=float)
    recall = ordered[:, 0]
    precision = ordered[:, 1]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    unique_recall, unique_indices = np.unique(recall, return_index=True)
    unique_precision = precision[unique_indices]
    if len(unique_recall) < 2:
        return float("nan")
    return float(
        np.sum(np.diff(unique_recall) * unique_precision[1:])
    )



# ---------------------------------------------------------------------------
# Parallel candidate evaluation
# ---------------------------------------------------------------------------
#
# `select_alarm_threshold` evaluates ~2001 candidates per (task, false-alarm
# target), and a run does four tasks times two targets, so about 16,812
# `metrics_at` calls -- measured at roughly 4 core-hours on the real 5.78M-row
# calibration partition. Fifteen of the seventeen campaign arms emit continuous
# scores and pay it in full.
#
# The calls are independent by construction: `AlarmThresholdSweep.__init__`
# precomputes immutable numpy arrays and `metrics_at` is a pure function of
# those plus the threshold. Measured, not assumed -- eight threads reading one
# sweep object concurrently produce output identical to the serial loop on all
# 101 sampled candidates under a NaN-aware comparison. (Naive `==` reports two
# differences, which are `median_lead_minutes` being NaN on both sides.)
#
# Threads do not help: the same experiment gives 1.43x at two and 1.35x at
# eight, because `_collapse_positions` and `_match` hold the GIL. Processes do,
# and `fork` lets the workers inherit the precomputed arrays instead of pickling
# them -- only `(lo, hi)` index pairs cross the boundary in, and small dicts
# out.
#
# Default stays serial. A parallel run must be asked for explicitly, and the
# result is bit-identical either way.

_SWEEP_CHUNKS_PER_WORKER = 8
_SWEEP_MINIMUM_CANDIDATES_PER_WORKER = 25
_SWEEP_WORKER_STATE: dict[str, object] = {}


def _sweep_worker_init(
    sweep: AlarmThresholdSweep,
    horizon_minutes: int,
    candidates: np.ndarray,
) -> None:
    _SWEEP_WORKER_STATE["sweep"] = sweep
    _SWEEP_WORKER_STATE["horizon"] = int(horizon_minutes)
    _SWEEP_WORKER_STATE["candidates"] = candidates


def _sweep_chunk(bounds: tuple[int, int]) -> list[dict[str, float]]:
    low, high = bounds
    sweep = _SWEEP_WORKER_STATE["sweep"]
    horizon = _SWEEP_WORKER_STATE["horizon"]
    candidates = _SWEEP_WORKER_STATE["candidates"]
    return [
        sweep.metrics_at(float(candidates[index]), horizon_minutes=horizon)
        for index in range(low, high)
    ]


def _sweep_start_method() -> str | None:
    """The safest available start method, or None if none is.

    Deliberately **not** `fork`. `select_alarm_threshold` runs after training
    in `deep_training` and `forecast_training`, where torch's thread pools are
    still alive, and forking a multi-threaded process can hand the child a
    mutex that is locked with no owner. Python 3.12 warns about exactly this.

    `forkserver` forks from a helper started before those threads exist, so the
    child is clean. The cost is that the sweep is pickled instead of inherited
    -- about 185 MiB for the 5.78M-row calibration partition, once per worker
    through `initializer`, not once per chunk. Against a sweep measured in
    core-hours that is not a consideration.
    """

    available = multiprocessing.get_all_start_methods()
    for method in ("forkserver", "spawn"):
        if method in available:
            return method
    return None


def _sweep_all_candidates(
    sweep: AlarmThresholdSweep,
    candidates: np.ndarray,
    *,
    horizon_minutes: int,
    workers: int,
) -> list[dict[str, float]]:
    """Evaluate every candidate, serially or across worker processes."""

    count = len(candidates)
    # Cap the pool by how much work there is to spread. Every worker pays a fixed
    # cost to receive the precomputed sweep state, and under `forkserver` that is
    # a pickle of arrays over the whole partition, so past some point another
    # worker costs more than it saves.
    #
    # Measured on l20 against `tcn_erm_seed17`'s real predictions, one selection
    # cell of 5,842,352 rows and 201 candidates, all three runs bit-identical:
    #
    #     workers  1   55.5 s   1.00x
    #     workers  8   16.6 s   3.35x
    #     workers 16   19.6 s   2.83x     <- slower than eight
    #
    # Sixteen is what `v13_launch_bundle` derives for a lane, and it is right for
    # the 2001-candidate alarm sweep and wrong for this 201-candidate one. The
    # divisor puts 201 candidates on eight workers and leaves 2001 at the caller's
    # figure.
    workers = max(1, min(workers, count // _SWEEP_MINIMUM_CANDIDATES_PER_WORKER))
    method = _sweep_start_method()
    if workers <= 1 or count < 2 * workers or method is None:
        return [
            sweep.metrics_at(float(threshold), horizon_minutes=horizon_minutes)
            for threshold in candidates
        ]

    # Contiguous chunks in candidate order, so the returned list is already in
    # the order the serial loop would have produced and no sort is needed.
    #
    # Several chunks per worker rather than one: cost per candidate falls
    # steeply with the threshold, because a low threshold admits far more
    # alarms into the refractory walk. One chunk each hands the whole expensive
    # head to a single worker and measured 4.07x on eight cores; letting the
    # pool refill from a finer queue took the same case to 4.64x.
    chunk = max(1, -(-count // (workers * _SWEEP_CHUNKS_PER_WORKER)))
    bounds = [
        (low, min(low + chunk, count)) for low in range(0, count, chunk)
    ]
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context(method),
            initializer=_sweep_worker_init,
            initargs=(sweep, int(horizon_minutes), candidates),
        ) as pool:
            chunks = list(pool.map(_sweep_chunk, bounds))
    except Exception as error:  # noqa: BLE001 - any pool failure degrades safely
        # This path is a pure optimisation over a loop that is already correct,
        # so a pool that cannot start must cost time, never the run. `spawn`
        # and `forkserver` re-import the parent's `__main__` in the child,
        # which fails outright when the entry point is not importable -- an
        # interactive session, or `python - <<EOF`. The campaign enters through
        # `python -m glycoshift.train`, which is importable, but a caller that
        # is not must still get its numbers.
        warnings.warn(
            "parallel alarm sweep unavailable, falling back to the serial "
            f"loop: {type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return [
            sweep.metrics_at(float(threshold), horizon_minutes=horizon_minutes)
            for threshold in candidates
        ]
    return [result for block in chunks for result in block]


def select_alarm_threshold(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    label_column: str,
    event_id_column: str,
    target_false_alarms_per_day: float,
    event_onset_column: str | None = None,
    horizon_minutes: int | None = None,
    candidate_count: int = 2001,
    workers: int = 1,
) -> float:
    probabilities = predictions[probability_column].to_numpy(dtype=float)
    if len(probabilities) == 0 or not np.isfinite(probabilities).all():
        raise ValueError("alarm threshold selection requires finite scores")
    no_alarm = np.nextafter(float(probabilities.max()), np.inf)
    if not np.isfinite(no_alarm):
        raise ValueError("cannot construct a finite no-alarm threshold")
    candidates = np.unique(
        np.concatenate(
            (
                np.array([0.0, no_alarm]),
                np.quantile(probabilities, np.linspace(0, 1, candidate_count)),
            )
        )
    )
    del label_column
    if horizon_minutes is None:
        raise ValueError("horizon_minutes is required")
    sweep = AlarmThresholdSweep(
        predictions,
        probability_column=probability_column,
        event_id_column=event_id_column,
        event_onset_column=event_onset_column,
    )
    results = _sweep_all_candidates(
        sweep,
        candidates,
        horizon_minutes=horizon_minutes,
        workers=workers,
    )
    feasible: list[tuple[float, float]] = []
    for threshold, metrics in zip(candidates, results, strict=True):
        if metrics["false_alarms_per_patient_day"] <= target_false_alarms_per_day:
            sensitivity = metrics["event_sensitivity"]
            feasible.append(
                (
                    -1.0 if np.isnan(sensitivity) else sensitivity,
                    float(threshold),
                )
            )
    if not feasible:
        raise AssertionError("no-alarm sentinel did not produce a feasible threshold")
    best_sensitivity = max(item[0] for item in feasible)
    return max(
        threshold
        for sensitivity, threshold in feasible
        if sensitivity == best_sensitivity
    )


def probability_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    ece_bins: int,
) -> dict[str, float]:
    y_true = np.asarray(list(labels), dtype=int)
    y_score = np.asarray(list(probabilities), dtype=float)
    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
        auprc = float("nan")
        slope = float("nan")
        intercept = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
        slope, intercept = calibration_slope_intercept(y_true, y_score)
    return {
        "window_auprc": auprc,
        "auprc": auprc,
        "auroc": auroc,
        "brier": float(brier_score_loss(y_true, y_score)),
        "ece": expected_calibration_error(y_true, y_score, bins=ece_bins),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "prevalence": float(y_true.mean()),
        "n": float(len(y_true)),
    }


def participant_bootstrap(
    predictions: pd.DataFrame,
    *,
    metric: Callable[[pd.DataFrame], float],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    result = participant_bootstrap_metrics(
        predictions,
        metric=lambda frame: {"value": float(metric(frame))},
        replicates=replicates,
        seed=seed,
    )
    return result["value"]


def _bootstrap_draw_copy(
    frame: pd.DataFrame,
    *,
    draw: int,
) -> pd.DataFrame:
    replicate = frame.copy()
    suffix = f"__bootstrap_{draw}"
    replicate["participant_id"] = (
        replicate["participant_id"].astype(str) + suffix
    )
    for event_column in replicate.filter(regex=r"_event_id$"):
        observed = replicate[event_column].notna()
        replicate.loc[observed, event_column] = (
            replicate.loc[observed, event_column].astype(str) + suffix
        )
    return replicate


def participant_bootstrap_metrics(
    predictions: pd.DataFrame,
    *,
    metric: Callable[[pd.DataFrame], dict[str, float]],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    participants = predictions[
        ["source_id", "participant_id"]
    ].drop_duplicates()
    if participants.empty:
        return {}
    keys = list(
        zip(  # noqa: B905
            participants["source_id"].astype(str),
            participants["participant_id"].astype(str),
        )
    )
    keyed = predictions.assign(
        _source_key=predictions["source_id"].astype(str),
        _participant_key=predictions["participant_id"].astype(str),
    )
    by_key = {
        (str(key[0]), str(key[1])): group.drop(
            columns=["_source_key", "_participant_key"]
        )
        for key, group in keyed.groupby(
            ["_source_key", "_participant_key"],
            sort=False,
        )
    }
    estimate = {
        name: float(value)
        for name, value in metric(predictions).items()
    }
    if replicates <= 0:
        return {
            name: {
                "estimate": value,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            }
            for name, value in estimate.items()
        }
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        name: []
        for name in estimate
    }
    for _ in range(replicates):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        replicate_frames: list[pd.DataFrame] = []
        for draw, index in enumerate(sampled):
            replicate_frames.append(
                _bootstrap_draw_copy(
                    by_key[keys[int(index)]],
                    draw=draw,
                )
            )
        replicate_metrics = metric(
            pd.concat(replicate_frames, ignore_index=True)
        )
        for name, value in replicate_metrics.items():
            numeric = float(value)
            if np.isfinite(numeric):
                values.setdefault(name, []).append(numeric)
    output: dict[str, dict[str, float]] = {}
    for name, point in estimate.items():
        samples = values.get(name, [])
        output[name] = {
            "estimate": point,
            "ci_low": (
                float(np.quantile(samples, 0.025))
                if samples
                else float("nan")
            ),
            "ci_high": (
                float(np.quantile(samples, 0.975))
                if samples
                else float("nan")
            ),
        }
    return output


def paired_participant_bootstrap_difference(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    metric: Callable[[pd.DataFrame], float],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    candidate_keys = set(
        zip(  # noqa: B905
            candidate["source_id"].astype(str),
            candidate["participant_id"].astype(str),
        )
    )
    reference_keys = set(
        zip(  # noqa: B905
            reference["source_id"].astype(str),
            reference["participant_id"].astype(str),
        )
    )
    if candidate_keys != reference_keys:
        raise ValueError("paired bootstrap requires identical participants")
    alignment_columns = ["source_id", "participant_id"]
    if "context_end" in candidate and "context_end" in reference:
        alignment_columns.append("context_end")
    candidate_rows = candidate[alignment_columns].astype(str)
    reference_rows = reference[alignment_columns].astype(str)
    if set(map(tuple, candidate_rows.to_numpy())) != set(
        map(tuple, reference_rows.to_numpy())
    ):
        raise ValueError("paired bootstrap requires identical prediction rows")
    common_labels = sorted(
        {
            column
            for column in candidate
            if column.endswith("_label") and column in reference
        }
    )
    if common_labels:
        aligned = candidate[
            alignment_columns + common_labels
        ].merge(
            reference[alignment_columns + common_labels],
            on=alignment_columns,
            how="inner",
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        for column in common_labels:
            left = aligned[f"{column}_candidate"]
            right = aligned[f"{column}_reference"]
            equal = left.eq(right) | (left.isna() & right.isna())
            if not equal.all():
                raise ValueError(
                    f"paired bootstrap outcome mismatch: {column}"
                )

    def group_by_key(frame: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
        keyed = frame.assign(
            _source_key=frame["source_id"].astype(str),
            _participant_key=frame["participant_id"].astype(str),
        )
        return {
            (str(key[0]), str(key[1])): group.drop(
                columns=["_source_key", "_participant_key"]
            )
            for key, group in keyed.groupby(
                ["_source_key", "_participant_key"],
                sort=False,
            )
        }

    candidate_groups = group_by_key(candidate)
    reference_groups = group_by_key(reference)
    keys = sorted(candidate_keys)
    estimate = float(metric(candidate) - metric(reference))
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        candidate_draws = [
            _bootstrap_draw_copy(
                candidate_groups[keys[int(index)]],
                draw=draw,
            )
            for draw, index in enumerate(sampled)
        ]
        reference_draws = [
            _bootstrap_draw_copy(
                reference_groups[keys[int(index)]],
                draw=draw,
            )
            for draw, index in enumerate(sampled)
        ]
        difference = float(
            metric(pd.concat(candidate_draws, ignore_index=True))
            - metric(pd.concat(reference_draws, ignore_index=True))
        )
        if np.isfinite(difference):
            differences.append(difference)
    if not differences:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value_two_sided": float("nan"),
        }
    values = np.asarray(differences, dtype=float)
    lower_tail = (float(np.sum(values <= 0)) + 1) / (len(values) + 1)
    upper_tail = (float(np.sum(values >= 0)) + 1) / (len(values) + 1)
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_value_two_sided": float(min(1.0, 2 * min(lower_tail, upper_tail))),
    }
