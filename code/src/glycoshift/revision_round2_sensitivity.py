"""Common-frame, fixed-emission and common-source round2 sensitivities.

Catalogue construction is intentionally separate from retrospective matching.
No helper here fits a policy, changes a score, or regenerates refractory emissions.
Common-source comparisons retain each catalogue's own participant/event population.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .events import Event
from .fixed_frame_evaluation import _online_eligible
from .revision_event_sensitivity import (
    _MATCH,
    _boolean,
    _equal,
    _event_frame,
    _normalized,
    base_catalogue,
    rematch_emitted_alarms,
)

_IDS = ["source_id", "participant_id"]
_CONDITIONS = ["arm", "model_seed", "horizon_minutes", "glucose_cut"]
_HORIZONS = (30, 60)
_CUTS = (54, 70)


@dataclass
class CommonFrame:
    timeline: pd.DataFrame
    exposure: pd.DataFrame
    audit: dict


@dataclass
class CatalogueRematch:
    alarms: pd.DataFrame
    events: pd.DataFrame
    participants: pd.DataFrame


@dataclass
class CommonSupportSummary:
    source_support: pd.DataFrame
    population_audit: pd.DataFrame
    points: pd.DataFrame
    contrasts: pd.DataFrame
    audit: dict


@dataclass(frozen=True)
class ElapsedSpanEvent(Event):
    """Event confirmed over 15 minutes, with explicit gap/terminal censoring.

    A noncensored ``end`` is the fourth consecutive recovery reading. A censored
    ``end`` is the last available reading before the gap or record boundary.
    """

    onset_confirmed_at: pd.Timestamp
    right_censored: bool
    censor_reason: str | None


def build_elapsed_span_catalogue(normalized: pd.DataFrame) -> list[ElapsedSpanEvent]:
    """Use four <54 readings for onset and four >=70 readings for recovery.

    Consecutive readings must be exactly five minutes apart. Onset is the first
    of the four low readings. A gap retains an already confirmed onset with right
    censoring and resets all detection state. An unfinished confirmed event at
    record end is likewise censored. No additional merge rule is applied.
    """
    frame = _normalized(normalized)
    source, participant = _one_identity(frame, "elapsed-span input")
    times = frame.timestamp
    glucose = frame.cgm_mg_dl.to_numpy(float)
    events = []
    onset = confirmed = None
    low_start = None
    low_count = recovery_count = 0

    def finish(end: int, reason: str | None) -> None:
        events.append(ElapsedSpanEvent(
            event_id=f"{source}:{participant}:severe_hypo_elapsed_span:{len(events):04d}",
            kind="severe_hypo_elapsed_span", onset=times.iloc[onset],
            end=times.iloc[end], onset_confirmed_at=times.iloc[confirmed],
            right_censored=reason is not None, censor_reason=reason,
        ))

    for index, value in enumerate(glucose):
        if index and (
            times.iloc[index] - times.iloc[index - 1] != pd.Timedelta(minutes=5)
        ):
            if onset is not None:
                finish(index - 1, "time_gap")
            onset = confirmed = low_start = None
            low_count = recovery_count = 0
        if onset is None:
            if value < 54:
                if low_count == 0:
                    low_start = index
                low_count += 1
                if low_count == 4:
                    onset, confirmed = low_start, index
                    recovery_count = 0
            else:
                low_start, low_count = None, 0
        elif value >= 70:
            recovery_count += 1
            if recovery_count == 4:
                finish(index, None)
                onset = confirmed = low_start = None
                low_count = recovery_count = 0
        else:
            recovery_count = 0
    if onset is not None:
        finish(len(frame) - 1, "record_end")
    return events


def _one_identity(frame: pd.DataFrame, name: str) -> tuple[str, str]:
    if not set(_IDS).issubset(frame) or frame.empty:
        raise ValueError(f"{name} requires one source-scoped participant")
    if frame[_IDS].isna().any().any():
        raise ValueError(f"{name} participant identities must be complete")
    identities = frame[_IDS].astype(str).drop_duplicates()
    if len(identities) != 1 or identities.iloc[0].str.strip().eq("").any():
        raise ValueError(f"{name} requires one source-scoped participant")
    return tuple(identities.iloc[0])


def common_completeness_frame(
    normalized_time_axis: pd.DataFrame, timeline: pd.DataFrame, *, horizon: int
) -> CommonFrame:
    """Require t+60/t+75 continuity for 30/60 minutes on the historical frame.

    Only normalized identities and timestamps define added completeness. No
    normalized glucose values, outcomes or provenance flags are read. All online
    timeline rows remain available, including rows excluded from attribution.
    """
    if horizon not in _HORIZONS:
        raise ValueError("common-frame horizon must be 30 or 60")
    key = _one_identity(normalized_time_axis, "normalized time axis")
    if _one_identity(timeline, "online timeline") != key:
        raise ValueError("common-frame participant identities differ")
    axis = pd.to_datetime(normalized_time_axis["timestamp"], utc=True, errors="coerce")
    times = pd.to_datetime(timeline["context_end"], utc=True, errors="coerce")
    if (
        axis.isna().any() or axis.duplicated().any()
        or times.isna().any() or times.duplicated().any()
        or not times.is_monotonic_increasing
    ):
        raise ValueError("common-frame timestamps must be complete, ordered and unique")
    axis = axis.sort_values().reset_index(drop=True)
    ns = axis.array.asi8
    at = np.searchsorted(ns, times.array.asi8)
    if np.any(at == len(ns)) or not np.array_equal(ns[at], times.array.asi8):
        raise ValueError("an online anchor is absent from the normalized time axis")
    width = 60 if horizon == 30 else 75
    step = pd.Timedelta(minutes=5).value
    segments = np.r_[0, np.diff(ns) != step].cumsum()
    end = at + width // 5
    safe = np.minimum(end, len(ns) - 1)
    complete = (end < len(ns)) & (segments[at] == segments[safe])
    historical = _boolean(timeline, "base_evaluable")
    base = historical & complete
    frame = timeline.copy(deep=True)
    frame["historical_base_evaluable"] = historical
    frame["accepted_primary_base_evaluable"] = _boolean(
        timeline, f"base_evaluable_{horizon}"
    )
    frame["common_confirmation_complete"] = complete
    frame["base_evaluable"] = base
    frame[f"base_evaluable_{horizon}"] = base
    exposure = pd.DataFrame([{
        "source_id": key[0], "participant_id": key[1],
        "patient_days": int(frame.loc[base, "patient_day"].nunique()),
        "observation_minutes": int(base.sum()) * 5,
    }])
    return CommonFrame(frame, exposure, {
        "horizon_minutes": horizon, "required_future_minutes": width,
        "online_decisions": len(frame), "common_evaluable_decisions": int(base.sum()),
        "evaluation_only": True, "completeness_uses_glucose": False,
    })


def _no_exposure_rematch(
    frame: CommonFrame, emitted: pd.DataFrame, *, glucose_cut: float
) -> CatalogueRematch:
    alarms = emitted.copy(deep=True)
    for column in _MATCH:
        alarms[f"primary_{column}"] = alarms[column]
    alarms["matched"] = False
    alarms["unmatched"] = False
    alarms["matched_event_id"] = None
    alarms["matched_event_onset"] = pd.to_datetime(pd.NaT, utc=True)
    alarms["lead_minutes"] = np.nan
    counts = (
        "base_event_count", "reachable_event_count", "detected_event_count",
        "eligible_evaluable_decision_count", "base_evaluable_decision_count",
        "delivered_alarm_count", "matched_alarm_count", "unmatched_alarm_count",
        "late_detection_count", "missed_event_count",
        *(f"coverage_{lead}_event_count" for lead in (5, 10, 15, 20)),
    )
    row = {
        **frame.exposure.iloc[0].to_dict(), **dict.fromkeys(counts, 0),
        "base_patient_days": (), "effective_24h_days": 0.0,
        "emitted_alarm_count": len(alarms), "unevaluable_alarm_count": len(alarms),
        "matched_lead_minutes": (),
        **dict.fromkeys(("fixed_sensitivity", "conditional_sensitivity", "ppv",
                         "unmatched_alarms_per_patient_day",
                         "delivered_alarms_per_patient_day"), float("nan")),
    }
    # Eligibility remains online even where attribution exposure is absent.
    row["eligible_decision_count"] = int(
        _online_eligible(frame.timeline, glucose_cut).sum()
    )
    key = _one_identity(frame.timeline, "common timeline")
    return CatalogueRematch(alarms, _event_frame([], key), pd.DataFrame([row]))


def rematch_catalogues_on_common_frame(
    normalized_time_axis: pd.DataFrame,
    timeline: pd.DataFrame,
    catalogues: Mapping[str, list[Event]],
    emitted_alarms: pd.DataFrame,
    *,
    horizon: int,
    glucose_cut: float,
) -> tuple[CommonFrame, dict[str, CatalogueRematch]]:
    """Rematch supplied catalogues using one common frame and fixed emissions."""
    if len(catalogues) < 2:
        raise ValueError("common-frame comparison requires at least two catalogues")
    if not np.isfinite(glucose_cut) or glucose_cut < 54:
        raise ValueError("invalid fixed-emission glucose cut")
    common = common_completeness_frame(normalized_time_axis, timeline, horizon=horizon)
    accepted = emitted_alarms.copy(deep=True)
    if not set(_MATCH).issubset(accepted):
        raise ValueError("fixed emissions are missing their original match ledger")
    accepted["context_end"] = pd.to_datetime(accepted.context_end, utc=True)
    if (
        accepted.context_end.isna().any() or accepted.context_end.duplicated().any()
        or not accepted.context_end.is_monotonic_increasing
    ):
        raise ValueError("fixed emission times must be ordered, unique and complete")
    anchor = common.timeline.set_index("context_end")
    if not accepted.context_end.isin(anchor.index).all():
        raise ValueError("a fixed emission is absent from the common online timeline")
    expected = anchor.loc[accepted.context_end].reset_index()
    _equal(
        accepted,
        expected.assign(base_evaluable=expected.accepted_primary_base_evaluable),
        [*_IDS, "context_end", "patient_day", "cgm_last", "base_evaluable"],
        "accepted primary emission anchor",
    )
    accepted["accepted_primary_base_evaluable"] = _boolean(accepted, "base_evaluable")
    accepted["base_evaluable"] = expected.base_evaluable.to_numpy(bool)
    online = _online_eligible(common.timeline, glucose_cut)
    if not online[common.timeline.context_end.isin(accepted.context_end)].all():
        raise ValueError("a fixed emission violates its frozen online eligibility")
    key = _one_identity(common.timeline, "common timeline")
    results = {}
    for name, catalogue in catalogues.items():
        retained = base_catalogue(catalogue, common.timeline, horizon)
        truth = _event_frame(retained, key)
        if catalogue and all(
            isinstance(event, ElapsedSpanEvent) for event in catalogue
        ):
            truth["onset_confirmed_at"] = pd.to_datetime(
                [event.onset_confirmed_at for event in retained], utc=True
            )
            truth["right_censored"] = pd.Series(
                [event.right_censored for event in retained], dtype=bool
            )
            truth["censor_reason"] = [event.censor_reason for event in retained]
        if not common.exposure.observation_minutes.iloc[0]:
            result = _no_exposure_rematch(common, accepted, glucose_cut=glucose_cut)
        else:
            alarms, events, participants = rematch_emitted_alarms(
                common.timeline, truth, common.exposure, accepted,
                horizon=horizon, glucose_cut=glucose_cut,
            )
            result = CatalogueRematch(alarms, events, participants)
        invariant = [column for column in emitted_alarms
                     if column not in (*_MATCH, "base_evaluable")]
        _equal(
            result.alarms, emitted_alarms, invariant, "fixed online emission sequence"
        )
        results[name] = result
    return common, results


def _validated_metrics(
    table: pd.DataFrame, sources: tuple[str, ...], candidate: str, control: str
) -> tuple[pd.DataFrame, list[int]]:
    required = {*_IDS, *_CONDITIONS, "base_event_count", "detected_event_count"}
    if table.empty or not required.issubset(table):
        raise ValueError("common-support participant metric columns/rows are missing")
    out = table.copy(deep=True)
    if out[_IDS + ["arm"]].isna().any().any():
        raise ValueError("common-support identities must be complete")
    out[_IDS] = out[_IDS].astype(str)
    if any(out[column].str.strip().eq("").any() for column in _IDS):
        raise ValueError("common-support identities must be complete")
    if set(out.source_id) != set(sources) or set(out.arm) != {candidate, control}:
        raise ValueError("common-support sources or arms are missing/foreign")
    if out.duplicated([*_CONDITIONS, *_IDS]).any():
        raise ValueError("common-support participant condition keys are duplicated")
    for column in (
        "horizon_minutes", "glucose_cut", "base_event_count", "detected_event_count"
    ):
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(float)
        if (
            not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values != np.floor(values))
        ):
            raise ValueError(
                f"common-support {column} must contain nonnegative integers"
            )
        out[column] = values.astype(int)
    if set(out.horizon_minutes) != set(_HORIZONS) or set(out.glucose_cut) != set(_CUTS):
        raise ValueError("common-support requires both horizons and cuts 54/70")
    if np.any(out.detected_event_count > out.base_event_count):
        raise ValueError("detected events exceed the catalogue denominator")
    candidate_seeds = pd.to_numeric(
        out.loc[out.arm.eq(candidate), "model_seed"], errors="coerce"
    ).to_numpy(float)
    if (
        not np.isfinite(candidate_seeds).all()
        or any(seed not in (17, 23, 42, 71, 101) for seed in candidate_seeds)
        or out.loc[out.arm.eq(control), "model_seed"].notna().any()
    ):
        raise ValueError(
            "common-support model seeds or deterministic control are invalid"
        )
    out.loc[out.arm.eq(candidate), "model_seed"] = candidate_seeds
    if out.duplicated([*_CONDITIONS, *_IDS]).any():
        raise ValueError("common-support participant condition keys are duplicated")
    seeds = sorted(set(candidate_seeds.astype(int)))
    expected = {(candidate, seed, horizon, cut) for seed in seeds
                for horizon in _HORIZONS for cut in _CUTS}
    expected.update(
        (control, None, horizon, cut) for horizon in _HORIZONS for cut in _CUTS
    )
    actual = {(str(row.arm), None if pd.isna(row.model_seed) else int(row.model_seed),
               int(row.horizon_minutes), int(row.glucose_cut))
              for row in out[_CONDITIONS].drop_duplicates().itertuples(index=False)}
    if actual != expected:
        raise ValueError("common-support comparison condition grid is incomplete")
    for _, horizon_table in out.groupby("horizon_minutes", observed=True):
        reference = None
        for _, condition in horizon_table.groupby(
            _CONDITIONS, dropna=False, observed=True
        ):
            truth = condition.set_index(_IDS).base_event_count.sort_index()
            if set(condition.source_id) != set(sources):
                raise ValueError("a common-support condition is missing a source")
            if reference is None:
                reference = truth
            elif not truth.equals(reference):
                raise ValueError(
                    "catalogue support/population changes with method, cut or seed"
                )
    return out, seeds


def summarize_common_source_support(
    primary: pd.DataFrame,
    observed_only: pd.DataFrame,
    *,
    source_order: tuple[str, ...],
    candidate_arm: str = "lstm",
    control_arm: str = "persistence",
) -> CommonSupportSummary:
    """Compare the fixed source intersection across both catalogues and horizons.

    Each input is a participant count ledger across both arms, horizons and cuts.
    Control has a null seed; its unchanged estimate is shared across model seeds.
    No participant or onset intersection is applied between catalogue definitions.
    """
    if not source_order or len(set(source_order)) != len(source_order):
        raise ValueError("source_order must contain unique sources")
    tables = {}
    seeds_by_catalogue = []
    for name, table in (("primary", primary), ("observed_only", observed_only)):
        tables[name], seeds = _validated_metrics(
            table, source_order, candidate_arm, control_arm
        )
        seeds_by_catalogue.append(seeds)
    if seeds_by_catalogue[0] != seeds_by_catalogue[1]:
        raise ValueError("catalogue comparison model seed grids differ")
    seeds = seeds_by_catalogue[0]
    support, population = [], []
    positive = {}
    for horizon in _HORIZONS:
        inventories = {}
        for name, table in tables.items():
            reference = table.loc[
                table.arm.eq(control_arm) & table.horizon_minutes.eq(horizon)
                & table.glucose_cut.eq(54)
            ]
            inventories[name] = reference
            positive[name, horizon] = set(reference.loc[
                reference.base_event_count.gt(0), "source_id"
            ])
        for source in source_order:
            row = {"horizon_minutes": horizon, "source_id": source}
            identities, positive_identities = {}, {}
            for name, inventory in inventories.items():
                selected = inventory.loc[inventory.source_id.eq(source)]
                identities[name] = set(selected.participant_id)
                positive_identities[name] = set(selected.loc[
                    selected.base_event_count.gt(0), "participant_id"
                ])
                row[f"{name}_base_event_count"] = int(selected.base_event_count.sum())
                row[f"{name}_event_positive_participants"] = len(
                    positive_identities[name]
                )
            row["common_positive_support"] = all(
                source in positive[name, horizon] for name in tables
            )
            support.append(row)
            a, b = identities["primary"], identities["observed_only"]
            population.append({
                "horizon_minutes": horizon, "source_id": source,
                "primary_participants": len(a), "observed_only_participants": len(b),
                "shared_participants": len(a & b),
                "primary_only_participants": len(a - b),
                "observed_only_only_participants": len(b - a),
                "same_participant_population": a == b,
                "same_event_positive_participant_population": (
                    positive_identities["primary"]
                    == positive_identities["observed_only"]
                ),
            })
    common = tuple(source for source in source_order if all(
        source in positive[name, horizon] for name in tables for horizon in _HORIZONS
    ))
    points, contrasts = [], []
    for scope, sources in (("full_sources", source_order), ("common_support", common)):
        for name, table in tables.items():
            estimates = {}
            for (arm, seed, horizon, cut), condition in table.groupby(
                _CONDITIONS, dropna=False, observed=True
            ):
                undefined, source_estimates = [], []
                for source in sources:
                    selected = condition.loc[
                        condition.source_id.eq(source)
                        & condition.base_event_count.gt(0)
                    ]
                    if selected.empty:
                        undefined.append(source)
                    else:
                        source_estimates.append(float((
                            selected.detected_event_count / selected.base_event_count
                        ).mean()))
                value = (
                    float(np.mean(source_estimates))
                    if sources and not undefined else float("nan")
                )
                seed = None if pd.isna(seed) else int(seed)
                estimates[arm, seed, int(horizon), int(cut)] = value
                points.append({
                    "scope": scope, "catalogue": name, "arm": arm, "model_seed": seed,
                    "horizon_minutes": horizon, "glucose_cut": cut,
                    "aggregation": "individual_condition", "source_count": len(sources),
                    "undefined_sources": tuple(undefined), "fixed_sensitivity": value,
                })
            interactions = []
            for horizon in _HORIZONS:
                for seed in seeds:
                    effect = {
                        cut: estimates[candidate_arm, seed, horizon, cut]
                        - estimates[control_arm, None, horizon, cut] for cut in _CUTS
                    }
                    interaction = effect[70] - effect[54]
                    interactions.append(interaction)
                    contrasts.append({
                        "scope": scope, "catalogue": name, "model_seed": seed,
                        "horizon_minutes": horizon,
                        "aggregation": "individual_condition",
                        "interaction_70_minus_54": interaction,
                    })
            contrasts.append({
                "scope": scope, "catalogue": name, "model_seed": None,
                "horizon_minutes": None,
                "aggregation": "equal_horizon_equal_model_seed",
                "interaction_70_minus_54": float(np.mean(interactions)),
            })
            for arm in (candidate_arm, control_arm):
                for cut in _CUTS:
                    values = [value for (a, _, _, c), value in estimates.items()
                              if a == arm and c == cut]
                    points.append({
                        "scope": scope, "catalogue": name, "arm": arm,
                        "model_seed": None,
                        "horizon_minutes": None, "glucose_cut": cut,
                        "aggregation": "equal_horizon_equal_model_seed",
                        "source_count": len(sources), "undefined_sources": tuple(
                            source for source in sources if any(
                                source not in positive[name, horizon]
                                for horizon in _HORIZONS
                            )
                        ),
                        "fixed_sensitivity": float(np.mean(values)),
                    })
    return CommonSupportSummary(
        pd.DataFrame(support), pd.DataFrame(population), pd.DataFrame(points),
        pd.DataFrame(contrasts), {
            "source_order": source_order, "common_source_order": common,
            "support_rule": "positive_in_both_catalogues_and_both_horizons",
            "model_seeds": tuple(seeds), "participant_intersection_applied": False,
            "onset_intersection_applied": False,
            "population_interpretation": (
                "Common sources retain each catalogue's participants and event truth; "
                "this is not a paired same-event or common-participant comparison."
            ),
        },
    )
