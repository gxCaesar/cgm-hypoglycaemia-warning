from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import torch


@dataclass(frozen=True)
class ParticipantEventSurrogate:
    """Differentiable alarm and one-to-one event accounting for one participant."""

    refractory_alarm_mass: torch.Tensor
    matched_event_mass: torch.Tensor
    false_alarm_mass: torch.Tensor
    event_miss: torch.Tensor
    event_credit: torch.Tensor
    patient_days: int

    @property
    def event_count(self) -> int:
        return int(self.event_credit.numel())

    @property
    def false_alarms_per_patient_day(self) -> torch.Tensor:
        return self.false_alarm_mass / self.patient_days


@dataclass(frozen=True)
class SourceEventSurrogate:
    """Participant-macro event miss and burden in lexical source order."""

    event_miss: torch.Tensor
    false_alarms_per_patient_day: torch.Tensor
    participant_counts: torch.Tensor
    event_positive_participant_counts: torch.Tensor


@dataclass(frozen=True)
class SourceEventRegretResult:
    """One source-normalized base-plus-event regret objective."""

    objective: torch.Tensor
    current_source_risks: torch.Tensor
    reference_source_risks: torch.Tensor
    excess_source_risks: torch.Tensor
    active_source_weights: torch.Tensor


def _finite_scalar(value: Real, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    if positive and resolved <= 0:
        raise ValueError(f"{field} must be positive")
    return resolved


def _validate_probability_vector(probabilities: torch.Tensor) -> None:
    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if not probabilities.is_floating_point():
        raise TypeError("probabilities must use a floating-point dtype")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities must be finite")
    if bool(((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("probabilities must lie in [0, 1]")


def _validate_decision_times(
    decision_minutes: torch.Tensor,
    *,
    expected_count: int,
) -> torch.Tensor:
    if decision_minutes.ndim != 1 or decision_minutes.numel() != expected_count:
        raise ValueError("decision_minutes must align one-to-one with probabilities")
    decision_minutes = decision_minutes.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(decision_minutes).all()):
        raise ValueError("decision_minutes must be finite")
    if expected_count > 1 and not bool(
        (decision_minutes[1:] > decision_minutes[:-1]).all()
    ):
        raise ValueError("decision_minutes must be strictly increasing")
    return decision_minutes


def soft_refractory_alarm_mass(
    probabilities: torch.Tensor,
    decision_minutes: torch.Tensor,
    *,
    collapse_minutes: Real = 30.0,
) -> torch.Tensor:
    """Apply a differentiable first-alarm refractory recurrence.

    In the binary-probability limit this is identical to the frozen greedy
    collapse: an alarm is retained only when it is strictly more than the
    collapse interval after the last retained alarm.  The state is local to
    this call, so callers must invoke it independently per participant.
    """

    _validate_probability_vector(probabilities)
    times = _validate_decision_times(
        decision_minutes,
        expected_count=int(probabilities.numel()),
    )
    interval = _finite_scalar(
        collapse_minutes,
        field="collapse_minutes",
        positive=True,
    )

    retained: list[torch.Tensor] = []
    refractory_start = 0
    for index, probability in enumerate(probabilities.unbind()):
        while (
            refractory_start < index
            and float(times[index] - times[refractory_start]) > interval
        ):
            refractory_start += 1
        suppressors = retained[refractory_start:index]
        if suppressors:
            available = torch.prod(1.0 - torch.stack(suppressors))
            retained.append(probability * available)
        else:
            retained.append(probability)
    return torch.stack(retained)


def participant_event_surrogate(
    probabilities: torch.Tensor,
    decision_minutes: torch.Tensor,
    event_onset_minutes: torch.Tensor,
    *,
    horizon_minutes: Real,
    patient_days: Integral,
    collapse_minutes: Real = 30.0,
    score_from: Integral = 0,
) -> ParticipantEventSurrogate:
    """Relax the frozen chronological earliest-unused-event alarm evaluator.

    Event availability is consumed softly in onset order.  With probabilities
    restricted to zero or one, the returned alarm, match, and false-alarm
    masses equal the exact greedy refractory and one-to-one counts.

    ``score_from`` restricts the returned alarm, match and false-alarm **masses**
    to decisions at or after that index while still running every earlier decision
    for state.  It exists so a warmed segment can report only the region it scores;
    ``0`` is the default and reproduces the original behaviour exactly.  Event
    credit is never restricted, because an earlier decision consuming an event is
    part of the state the later decisions must see.
    """

    _validate_probability_vector(probabilities)
    times = _validate_decision_times(
        decision_minutes,
        expected_count=int(probabilities.numel()),
    )
    if event_onset_minutes.ndim != 1:
        raise ValueError("event_onset_minutes must be a vector")
    event_onset_minutes = event_onset_minutes.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(event_onset_minutes).all()):
        raise ValueError("event_onset_minutes must be finite")
    if event_onset_minutes.numel() > 1 and not bool(
        (event_onset_minutes[1:] > event_onset_minutes[:-1]).all()
    ):
        raise ValueError("event_onset_minutes must be strictly increasing")
    horizon = _finite_scalar(
        horizon_minutes,
        field="horizon_minutes",
        positive=True,
    )
    if isinstance(patient_days, bool) or not isinstance(patient_days, Integral):
        raise TypeError("patient_days must be an integer")
    days = int(patient_days)
    if days <= 0:
        raise ValueError("patient_days must be positive")
    if isinstance(score_from, bool) or not isinstance(score_from, Integral):
        raise TypeError("score_from must be an integer")
    first_scored = int(score_from)
    if first_scored < 0 or first_scored >= int(probabilities.numel()):
        raise ValueError("score_from must index a scored decision")

    alarm_mass = soft_refractory_alarm_mass(
        probabilities,
        times,
        collapse_minutes=collapse_minutes,
    )
    event_count = int(event_onset_minutes.numel())
    zero = probabilities.new_zeros(())
    remaining = [probabilities.new_ones(()) for _ in range(event_count)]
    credits = [probabilities.new_zeros(()) for _ in range(event_count)]
    matched_terms: list[torch.Tensor] = []
    false_terms: list[torch.Tensor] = []
    eligible_starts = torch.searchsorted(
        event_onset_minutes,
        times,
        right=True,
    ).tolist()
    eligible_stops = torch.searchsorted(
        event_onset_minutes,
        times + horizon,
        right=True,
    ).tolist()

    for decision_index, (retained_mass, start, stop) in enumerate(
        zip(
            alarm_mass.unbind(),
            eligible_starts,
            eligible_stops,
            strict=True,
        )
    ):
        scored = decision_index >= first_scored
        earlier_unavailable = probabilities.new_ones(())
        choices: list[tuple[int, torch.Tensor]] = []
        for event_index in range(int(start), int(stop)):
            choice = remaining[event_index] * earlier_unavailable
            earlier_unavailable = earlier_unavailable * (
                1.0 - remaining[event_index]
            )
            choices.append((event_index, choice))

        total_choice = (
            torch.stack([choice for _, choice in choices]).sum()
            if choices
            else zero
        )
        matches = [
            (event_index, retained_mass * choice)
            for event_index, choice in choices
        ]
        matched = (
            torch.stack([match for _, match in matches]).sum()
            if matches
            else zero
        )
        if scored:
            matched_terms.append(matched)
            false_terms.append(retained_mass * (1.0 - total_choice))
        for event_index, match in matches:
            remaining[event_index] = remaining[event_index] - match
            credits[event_index] = credits[event_index] + match

    event_credit = torch.stack(credits) if credits else probabilities.new_empty(0)
    matched_mass = torch.stack(matched_terms).sum() if matched_terms else zero
    false_alarm_mass = torch.stack(false_terms).sum() if false_terms else zero
    event_miss = (
        (1.0 - event_credit).mean() if event_count else probabilities.new_zeros(())
    )
    values = (alarm_mass.sum(), matched_mass, false_alarm_mass, event_miss)
    if not all(bool(torch.isfinite(value)) for value in values):
        raise FloatingPointError("participant event surrogate became non-finite")
    return ParticipantEventSurrogate(
        refractory_alarm_mass=alarm_mass[first_scored:].sum(),
        matched_event_mass=matched_mass,
        false_alarm_mass=false_alarm_mass,
        event_miss=event_miss,
        event_credit=event_credit,
        patient_days=days,
    )


def aggregate_source_event_surrogates(
    participants: Sequence[ParticipantEventSurrogate],
    source_indices: torch.Tensor,
    *,
    source_count: Integral,
) -> SourceEventSurrogate:
    """Macro-average participants within each source without dropping burden rows."""

    if not participants:
        raise ValueError("at least one participant surrogate is required")
    positive_indices = tuple(
        index
        for index, participant in enumerate(participants)
        if participant.event_count > 0
    )
    return aggregate_source_event_components(
        miss_participants=tuple(participants[index] for index in positive_indices),
        miss_source_indices=source_indices[
            torch.tensor(
                positive_indices,
                device=source_indices.device,
                dtype=torch.long,
            )
        ],
        burden_participants=participants,
        burden_source_indices=source_indices,
        source_count=source_count,
    )


def _validate_source_indices(
    source_indices: torch.Tensor,
    *,
    participant_count: int,
    source_count: int,
    device: torch.device,
    field: str,
) -> None:
    if source_indices.ndim != 1 or source_indices.numel() != participant_count:
        raise ValueError(f"{field} must align with participants")
    if source_indices.dtype != torch.long:
        raise TypeError(f"{field} must use torch.long")
    if source_indices.device != device:
        raise ValueError(f"{field} and participant surrogates must share a device")
    if bool(((source_indices < 0) | (source_indices >= source_count)).any()):
        raise ValueError(f"{field} contains an index outside source_count")


def _validate_surrogate_scalars(
    participants: Sequence[ParticipantEventSurrogate],
    *,
    device: torch.device,
    dtype: torch.dtype,
    field: str,
) -> None:
    for participant in participants:
        for value in (
            participant.event_miss,
            participant.refractory_alarm_mass,
            participant.matched_event_mass,
            participant.false_alarm_mass,
        ):
            if (
                value.ndim != 0
                or value.device != device
                or value.dtype != dtype
                or not bool(torch.isfinite(value))
            ):
                raise ValueError(
                    f"{field} surrogate scalars must be finite with shared "
                    "shape, dtype, and device"
                )


def aggregate_source_event_components(
    *,
    miss_participants: Sequence[ParticipantEventSurrogate],
    miss_source_indices: torch.Tensor,
    burden_participants: Sequence[ParticipantEventSurrogate],
    burden_source_indices: torch.Tensor,
    source_count: Integral,
) -> SourceEventSurrogate:
    """Aggregate separate unbiased participant samples for miss and burden."""

    if not miss_participants or not burden_participants:
        raise ValueError("miss and burden participant samples must be non-empty")
    if isinstance(source_count, bool) or not isinstance(source_count, Integral):
        raise TypeError("source_count must be an integer")
    count = int(source_count)
    if count <= 0:
        raise ValueError("source_count must be positive")
    device = miss_participants[0].event_miss.device
    dtype = miss_participants[0].event_miss.dtype
    _validate_source_indices(
        miss_source_indices,
        participant_count=len(miss_participants),
        source_count=count,
        device=device,
        field="miss_source_indices",
    )
    _validate_source_indices(
        burden_source_indices,
        participant_count=len(burden_participants),
        source_count=count,
        device=device,
        field="burden_source_indices",
    )
    _validate_surrogate_scalars(
        miss_participants,
        device=device,
        dtype=dtype,
        field="miss",
    )
    _validate_surrogate_scalars(
        burden_participants,
        device=device,
        dtype=dtype,
        field="burden",
    )
    if any(participant.event_count <= 0 for participant in miss_participants):
        raise ValueError("miss participants must be event-positive")

    miss = torch.stack(
        [participant.event_miss for participant in miss_participants]
    )
    burden = torch.stack(
        [
            participant.false_alarms_per_patient_day
            for participant in burden_participants
        ]
    )
    participant_counts = torch.bincount(
        burden_source_indices,
        minlength=count,
    )
    positive_counts = torch.bincount(
        miss_source_indices,
        minlength=count,
    )
    if bool((participant_counts == 0).any()):
        raise ValueError("every source requires at least one participant")
    if bool((positive_counts == 0).any()):
        raise ValueError("every source requires an event-positive participant")

    burden_sums = torch.zeros(count, device=device, dtype=dtype).scatter_add(
        0,
        burden_source_indices,
        burden,
    )
    miss_sums = torch.zeros(count, device=device, dtype=dtype).scatter_add(
        0,
        miss_source_indices,
        miss,
    )
    return SourceEventSurrogate(
        event_miss=miss_sums / positive_counts.to(dtype=dtype),
        false_alarms_per_patient_day=(
            burden_sums / participant_counts.to(dtype=dtype)
        ),
        participant_counts=participant_counts,
        event_positive_participant_counts=positive_counts,
    )


def equal_weight_horizon_source_event_regret(
    *,
    base_source_risks: torch.Tensor,
    reference_base_risks: torch.Tensor,
    event_miss_by_horizon: Mapping[int, torch.Tensor],
    reference_event_miss_by_horizon: Mapping[int, torch.Tensor],
    false_alarms_by_horizon: Mapping[int, torch.Tensor],
    reference_false_alarms_by_horizon: Mapping[int, torch.Tensor],
    event_scale: Real,
    dual_value: Real,
    target_fapd: Real = 0.5,
    horizon_order: Sequence[int] = (30, 60),
) -> SourceEventRegretResult:
    """Combine the frozen 30/60-minute event components with equal weight."""

    horizons = tuple(horizon_order)
    if horizons != (30, 60):
        raise ValueError("horizon_order must be exactly (30, 60)")
    mappings = (
        event_miss_by_horizon,
        reference_event_miss_by_horizon,
        false_alarms_by_horizon,
        reference_false_alarms_by_horizon,
    )
    if any(set(mapping) != set(horizons) for mapping in mappings):
        raise ValueError("every event component must cover both frozen horizons")

    def mean_component(mapping: Mapping[int, torch.Tensor]) -> torch.Tensor:
        return torch.stack([mapping[horizon] for horizon in horizons]).mean(dim=0)

    return source_event_regret_objective(
        base_source_risks=base_source_risks,
        reference_base_risks=reference_base_risks,
        event_miss=mean_component(event_miss_by_horizon),
        reference_event_miss=mean_component(reference_event_miss_by_horizon),
        false_alarms_per_patient_day=mean_component(false_alarms_by_horizon),
        reference_false_alarms_per_patient_day=mean_component(
            reference_false_alarms_by_horizon
        ),
        event_scale=event_scale,
        dual_value=dual_value,
        target_fapd=target_fapd,
    )


def source_event_regret_objective(
    *,
    base_source_risks: torch.Tensor,
    reference_base_risks: torch.Tensor,
    event_miss: torch.Tensor,
    reference_event_miss: torch.Tensor,
    false_alarms_per_patient_day: torch.Tensor,
    reference_false_alarms_per_patient_day: torch.Tensor,
    event_scale: Real,
    dual_value: Real,
    target_fapd: Real = 0.5,
) -> SourceEventRegretResult:
    """Form the hard maximum of cross-fitted source-and-event excess risks."""

    vectors = (
        base_source_risks,
        reference_base_risks,
        event_miss,
        reference_event_miss,
        false_alarms_per_patient_day,
        reference_false_alarms_per_patient_day,
    )
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError("source risk components must be vectors")
    if any(vector.shape != base_source_risks.shape for vector in vectors[1:]):
        raise ValueError("source risk components must share one shape")
    if base_source_risks.numel() == 0:
        raise ValueError("source risk components must be non-empty")
    if not base_source_risks.is_floating_point():
        raise TypeError("source risk components must be floating-point")
    if any(
        vector.device != base_source_risks.device
        or vector.dtype != base_source_risks.dtype
        or not bool(torch.isfinite(vector).all())
        for vector in vectors
    ):
        raise ValueError(
            "source risk components must be finite with shared dtype/device"
        )
    scale = _finite_scalar(event_scale, field="event_scale")
    dual = _finite_scalar(dual_value, field="dual_value")
    target = _finite_scalar(target_fapd, field="target_fapd")
    if scale < 0 or dual < 0 or target < 0:
        raise ValueError(
            "event_scale, dual_value, and target_fapd must be non-negative"
        )

    detached_reference_base = reference_base_risks.detach()
    detached_reference_miss = reference_event_miss.detach()
    detached_reference_burden = reference_false_alarms_per_patient_day.detach()
    current = base_source_risks + scale * (
        event_miss + dual * (false_alarms_per_patient_day - target)
    )
    reference = detached_reference_base + scale * (
        detached_reference_miss
        + dual * (detached_reference_burden - target)
    )
    excess = current - reference
    objective, active_index = torch.max(excess, dim=0)
    weights = torch.zeros_like(excess)
    weights[active_index] = 1.0
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError("source-event regret objective became non-finite")
    return SourceEventRegretResult(
        objective=objective,
        current_source_risks=current,
        reference_source_risks=reference,
        excess_source_risks=excess,
        active_source_weights=weights,
    )


def update_fapd_dual(
    current_value: Real,
    *,
    epoch_index: Integral,
    observed_equal_source_fapd: Real,
    target_fapd: Real = 0.5,
) -> float:
    """Apply the frozen projected ``1/sqrt(epoch)`` train-only dual update."""

    current = _finite_scalar(current_value, field="current_value")
    observed = _finite_scalar(
        observed_equal_source_fapd,
        field="observed_equal_source_fapd",
    )
    target = _finite_scalar(target_fapd, field="target_fapd")
    if current < 0 or observed < 0 or target < 0:
        raise ValueError("dual and FAPD values must be non-negative")
    if isinstance(epoch_index, bool) or not isinstance(epoch_index, Integral):
        raise TypeError("epoch_index must be an integer")
    epoch = int(epoch_index)
    if epoch < 0:
        raise ValueError("epoch_index must be non-negative")
    updated = max(0.0, current + (observed - target) / math.sqrt(epoch + 1))
    if not math.isfinite(updated):
        raise FloatingPointError("FAPD dual update became non-finite")
    return updated
