"""Train on the deployed decision rule's own estimand, on contiguous segments.

The frozen primary endpoint is equal-source participant-macro severe-event
sensitivity at a nominal `0.5` false alarms per participant-day.  `event_regret`
already supplies a differentiable relaxation of the exact alarm evaluator, but the
only consumer that exists binds it to complete participant trajectories and to a
55-fold cross-fitted reference, which together cost more than the whole campaign
budget (`FINDING-the-complete-stream-adapter-is-not-affordable.md`).

**Both expensive operators are local in the binary limit, and only approximately
local in the relaxation.**  `soft_refractory_alarm_mass` suppresses inside
`collapse_minutes`, about six decisions at five-minute resolution, and
`participant_event_surrogate` matches a decision only to events with onset in
`(t, t + horizon]`.  With probabilities in `{0, 1}` that makes a contiguous segment
exact.  With soft probabilities the recurrence `retained[i] = p[i] * prod(1 -
retained[j])` chains backward indefinitely, so a segment is **not** exact -- a claim
this module's first draft made and its own gate refuted.

The dependence decays geometrically and the decay was measured rather than assumed.
Relative error of a warmed segment's alarm mass against the true stream's, by warm-up
length and mean alarm probability:

```
mean p      W=0       W=8      W=16      W=32      W=64
 0.500   1.4e-02   8.2e-03   1.7e-03   4.0e-05   5.5e-07
 0.100   9.1e-03   4.5e-04   2.6e-05   2.9e-07   1.2e-12
 0.010   1.8e-03   7.1e-06   2.7e-09   4.6e-13   0.0e+00
```

So the construction **warms up rather than approximates**: a segment carries
`WARMUP_DECISIONS` leading decisions that are run for state and excluded from the
reported masses by the surrogate core's `score_from`.  At the default that is exact
to floating point for every alarm rate this endpoint operates at.

Event credit needs no such correction and is exact by construction.  Only events with
onset in `(t_start + horizon, t_last]` are scored, so every decision that could match
one lies inside the segment and no decision outside it could have consumed one.

Aggregation is written here rather than taken from `aggregate_source_event_components`
for one reason: that function reads the surrogate's `false_alarms_per_patient_day`
field, which divides by an **integer** `patient_days`, and a segment's exposure is
not a whole number of days.  The surrogate core is used unchanged; only the division
is ours, and it uses exact fractional days.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .event_regret import participant_event_surrogate, update_fapd_dual
from .target_alignment import SEVERE_SCORE_SOURCE_TASK

#: The horizons the operational endpoint scores.
SEGMENT_HORIZONS: tuple[int, ...] = (30, 60)
#: Minutes per day, for the exact fractional exposure a segment represents.
MINUTES_PER_DAY = 1440.0
#: The refractory interval the frozen evaluator uses.
COLLAPSE_MINUTES = 30.0
#: Leading decisions run for state and excluded from the reported masses. Measured
#: above: at 64 the warmed segment reproduces the stream to floating-point zero for
#: alarm rates at or below 0.1, which is far above this endpoint's operating point.
WARMUP_DECISIONS = 64


@dataclass(frozen=True)
class ParticipantSegmentIndex:
    """One participant's decision trajectory, in time order, as row positions."""

    source_id: str
    participant_id: str
    source_index: int
    positions: np.ndarray
    minutes: np.ndarray
    #: Integer codes for `patient_day`, the LOCAL day the frozen metric counts.
    day_codes: np.ndarray
    onsets: tuple[tuple[int, np.ndarray], ...]

    def onset_minutes(self, horizon: int) -> np.ndarray:
        for current, values in self.onsets:
            if current == horizon:
                return values
        raise KeyError(f"unknown horizon: {horizon}")

    @property
    def decision_count(self) -> int:
        return int(self.positions.size)


@dataclass(frozen=True)
class Segment:
    """A contiguous span of one participant, with the events it may be scored on."""

    participant: ParticipantSegmentIndex
    horizon: int
    #: First decision of the **warm-up**, not of the scored region.
    start: int
    stop: int
    #: Leading decisions run for state and excluded from the reported masses.
    warmup: int
    #: Onsets supplied to the surrogate, over the warmed span.
    supplied_onsets: np.ndarray
    #: Boolean over `supplied_onsets`: those with a full horizon inside the scored
    #: region, so that no decision outside the segment could have consumed them.
    scored_mask: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        return self.participant.positions[self.start : self.stop]

    @property
    def minutes(self) -> np.ndarray:
        return self.participant.minutes[self.start : self.stop]

    @property
    def scored_minutes(self) -> np.ndarray:
        return self.minutes[self.warmup :]

    @property
    def scored_day_codes(self) -> np.ndarray:
        return self.participant.day_codes[self.start + self.warmup : self.stop]

    @property
    def exposure_days(self) -> float:
        """Distinct observed days, matching what the frozen metric divides by.

        NOT the wall-clock span. A participant's eligible decisions can be very
        sparse: the real-data dry run found a 288-decision segment spanning
        **103.066 days**, and dividing its false alarms by 103 would have told the
        model that source had essentially unlimited budget. `metrics.py` counts
        patient-days with data, so this counts the same thing.
        """

        codes = self.scored_day_codes
        if codes.size == 0:
            raise ValueError("a segment must score at least one decision")
        return float(np.unique(codes).size)


def build_participant_segment_index(
    windows: pd.DataFrame,
    *,
    source_order: Sequence[str],
    sample_minutes: int,
    horizons: Sequence[int] = SEGMENT_HORIZONS,
) -> dict[tuple[str, str], ParticipantSegmentIndex]:
    """Index every participant's eligible decisions in time order.

    `positions` are positions into `windows` as supplied, so they address the
    training dataset directly.  Rows where a severe event is already active are
    dropped, which is the endpoint's own eligibility rule.
    """

    required = {"source_id", "participant_id", "context_end", "patient_day"}
    for horizon in horizons:
        required |= {
            f"severe_hypo_{horizon}_event_id",
            f"severe_hypo_{horizon}_event_onset",
        }
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"segment index requires columns: {missing}")
    if "severe_hypo_currently_active" not in windows.columns:
        raise ValueError("segment index requires severe_hypo_currently_active")
    if int(sample_minutes) <= 0:
        raise ValueError("sample_minutes must be positive")

    order = {str(name): index for index, name in enumerate(source_order)}
    times = pd.to_datetime(windows["context_end"], utc=True, errors="coerce")
    if bool(times.isna().any()):
        raise ValueError("segment index requires parseable context_end values")
    # DATETIME ONLY, deliberately. Accepting a numeric column too would leave the
    # units ambiguous -- epoch minutes, epoch seconds, nanoseconds -- and the wrong
    # reading is silent rather than loud. The real shards store a timestamp, so this
    # requires one and the fixtures were changed to match rather than the reverse.
    for horizon in horizons:
        column = windows[f"severe_hypo_{horizon}_event_onset"]
        if not pd.api.types.is_datetime64_any_dtype(column):
            raise TypeError(
                f"severe_hypo_{horizon}_event_onset must be a datetime column, got "
                f"{column.dtype}; a numeric column leaves the time unit ambiguous"
            )
    epoch_minutes = times.astype("int64").to_numpy() / 60_000_000_000.0
    active = windows["severe_hypo_currently_active"].to_numpy(dtype=bool)
    # The LOCAL day, taken from the column rather than by flooring UTC minutes: the
    # frozen metric counts distinct `["source_id", "participant_id", "patient_day"]`
    # (metrics.py:537), and a UTC floor disagrees with a local day at the boundary.
    day_codes = pd.factorize(windows["patient_day"].astype(str))[0]
    sources = windows["source_id"].astype(str).to_numpy()
    participants = windows["participant_id"].astype(str).to_numpy()

    # Event onsets are TIMESTAMPS in the real shards, not numbers. `pd.to_numeric`
    # on a `datetime64[ns, UTC]` column returns NANOSECONDS, which compares silently
    # and wrongly against decision minutes, and it maps `NaT` to the int64 minimum,
    # which passes `np.isfinite`. Both faults are silent; the first version of this
    # function had both, and only the sampler's fail-closed guard exposed them.
    # Parsed the same way `context_end` is, and missingness read from the parsed
    # values rather than from a stringified identifier.
    onset_columns: dict[int, np.ndarray] = {}
    present_columns: dict[int, np.ndarray] = {}
    for horizon in horizons:
        column = windows[f"severe_hypo_{horizon}_event_onset"]
        parsed = pd.to_datetime(column, utc=True, errors="coerce")
        present_columns[horizon] = (~parsed.isna()).to_numpy()
        onset_columns[horizon] = np.where(
            present_columns[horizon],
            parsed.astype("int64").to_numpy() / 60_000_000_000.0,
            np.nan,
        )
    identifier_columns = {
        horizon: windows[f"severe_hypo_{horizon}_event_id"].astype(str).to_numpy()
        for horizon in horizons
    }

    index: dict[tuple[str, str], ParticipantSegmentIndex] = {}
    frame_order = np.lexsort((epoch_minutes, participants, sources))
    boundaries = np.flatnonzero(
        np.r_[
            True,
            (sources[frame_order][1:] != sources[frame_order][:-1])
            | (participants[frame_order][1:] != participants[frame_order][:-1]),
            True,
        ]
    )
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        block = frame_order[low:high]
        source = str(sources[block[0]])
        participant = str(participants[block[0]])
        if source not in order:
            continue
        keep = block[~active[block]]
        if keep.size == 0:
            continue
        minutes = epoch_minutes[keep]
        if minutes.size > 1 and not bool((np.diff(minutes) > 0).all()):
            raise ValueError(
                f"decision minutes are not strictly increasing for {participant}"
            )
        onsets: list[tuple[int, np.ndarray]] = []
        for horizon in horizons:
            identifiers = identifier_columns[horizon][block]
            values = onset_columns[horizon][block]
            present = present_columns[horizon][block]
            pairs = {
                str(identifier): float(value)
                for identifier, value in zip(
                    identifiers[present], values[present], strict=True
                )
            }
            resolved = np.array(sorted(pairs.values()), dtype=float)
            if resolved.size > 1 and not bool((np.diff(resolved) > 0).all()):
                raise ValueError(
                    f"event onsets are not strictly increasing for {participant}"
                )
            onsets.append((int(horizon), resolved))
        index[(source, participant)] = ParticipantSegmentIndex(
            source_id=source,
            participant_id=participant,
            source_index=int(order[source]),
            positions=keep.astype(np.int64),
            minutes=minutes.astype(np.float64),
            day_codes=day_codes[keep].astype(np.int64),
            onsets=tuple(onsets),
        )
    return index


def _collate_sequences(batch: Sequence[Mapping[str, torch.Tensor]]) -> torch.Tensor:
    """Module level so `num_workers > 0` can pickle it.

    A lambda here is importable in the parent and unpicklable in a forked worker, so
    the failure appears only under multiprocessing -- which on this project means only
    on the training host, at `num_workers: 8`, after the data is loaded.
    """

    return torch.stack([item["sequence"] for item in batch])


def _valid_starts(
    participant: ParticipantSegmentIndex,
    *,
    onset: float,
    horizon: int,
    decisions: int,
    warmup: int = WARMUP_DECISIONS,
) -> tuple[int, int]:
    """The closed range of starts that place `onset` strictly inside the scored region.

    A start is valid when `minutes[start] + horizon < onset` (so the event is not in
    the unscored leading region) and `minutes[start + decisions - 1] >= onset` (so the
    event's own decision window is covered).
    """

    minutes = participant.minutes
    count = int(minutes.size)
    if count < decisions + warmup:
        return (1, 0)
    below = int(np.searchsorted(minutes, onset - float(horizon), side="left"))
    at_or_after = int(np.searchsorted(minutes, onset, side="left"))
    low = max(warmup, at_or_after - decisions + 1)
    high = min(below - 1, count - decisions)
    return (low, high)


def _build_segment(
    participant: ParticipantSegmentIndex,
    *,
    horizon: int,
    start: int,
    decisions: int,
    warmup: int = WARMUP_DECISIONS,
) -> Segment:
    """Build the warmed span whose SCORED region begins at `start`.

    `start` indexes the first scored decision. The span actually run is
    `[start - warmup, start + decisions)`, and the surrogate is told to report masses
    only from `warmup` onward.
    """

    if warmup < 0 or start < warmup:
        raise ValueError("a segment needs its full warm-up inside the trajectory")
    begin = start - warmup
    stop = start + decisions
    if stop > participant.decision_count:
        raise ValueError("segment extends past the trajectory")
    minutes = participant.minutes[begin:stop]
    scored_minutes = participant.minutes[start:stop]
    onsets = participant.onset_minutes(horizon)
    low = float(minutes[0])
    high = float(minutes[-1])
    supplied = onsets[(onsets > low) & (onsets <= high + float(horizon))]
    scored = (supplied > float(scored_minutes[0]) + float(horizon)) & (
        supplied <= high
    )
    return Segment(
        participant=participant,
        horizon=int(horizon),
        start=int(begin),
        stop=int(stop),
        warmup=int(warmup),
        supplied_onsets=supplied.astype(np.float64),
        scored_mask=scored,
    )


@dataclass(frozen=True)
class SegmentPack:
    """One update's miss and burden segments, in lexical source order."""

    epoch: int
    update_index: int
    miss: tuple[Segment, ...]
    burden: tuple[Segment, ...]


class SegmentPackSampler:
    """Draw contiguous miss and burden segments, one participant per source.

    Miss and burden are **separate** samples on purpose.  Conditioning a burden
    segment on containing an event would bias the false-alarm estimate upward, and
    `aggregate_source_event_components` keeps the same separation for the same
    reason.
    """

    def __init__(
        self,
        index: Mapping[tuple[str, str], ParticipantSegmentIndex],
        *,
        source_order: Sequence[str],
        segment_decisions: int,
        seed: int,
        horizons: Sequence[int] = SEGMENT_HORIZONS,
        attempts: int = 24,
        warmup: int = WARMUP_DECISIONS,
    ) -> None:
        if int(segment_decisions) <= 1:
            raise ValueError("segment_decisions must exceed one")
        if int(warmup) < 0:
            raise ValueError("warmup must be non-negative")
        self.source_order = tuple(str(name) for name in source_order)
        self.segment_decisions = int(segment_decisions)
        self.warmup = int(warmup)
        self.horizons = tuple(int(value) for value in horizons)
        self.seed = int(seed)
        self.attempts = int(attempts)
        self.index = dict(index)
        self.by_source: dict[str, tuple[ParticipantSegmentIndex, ...]] = {}
        self.event_positive: dict[int, dict[str, tuple[ParticipantSegmentIndex, ...]]]
        self.event_positive = {horizon: {} for horizon in self.horizons}
        for source in self.source_order:
            members = tuple(
                value
                for key, value in sorted(self.index.items())
                if key[0] == source
                and value.decision_count >= self.segment_decisions + self.warmup
            )
            if not members:
                raise ValueError(f"source {source} has no long-enough participant")
            self.by_source[source] = members
            for horizon in self.horizons:
                positive = tuple(
                    member
                    for member in members
                    if member.onset_minutes(horizon).size > 0
                )
                if not positive:
                    raise ValueError(
                        f"source {source} has no event-positive participant "
                        f"at horizon {horizon}"
                    )
                self.event_positive[horizon][source] = positive

    def _generator(self, epoch: int, update_index: int) -> np.random.Generator:
        return np.random.default_rng(
            (self.seed, int(epoch), int(update_index))
        )

    def pack(self, epoch: int, update_index: int) -> SegmentPack:
        rng = self._generator(epoch, update_index)
        burden: list[Segment] = []
        for source in self.source_order:
            members = self.by_source[source]
            member = members[int(rng.integers(len(members)))]
            span = member.decision_count - self.segment_decisions - self.warmup
            start = self.warmup + int(rng.integers(span + 1))
            burden.append(
                _build_segment(
                    member,
                    horizon=self.horizons[0],
                    start=start,
                    decisions=self.segment_decisions,
                    warmup=self.warmup,
                )
            )
        miss: list[Segment] = []
        for horizon in self.horizons:
            for source in self.source_order:
                miss.append(self._miss_segment(rng, source=source, horizon=horizon))
        return SegmentPack(
            epoch=int(epoch),
            update_index=int(update_index),
            miss=tuple(miss),
            burden=tuple(burden),
        )

    def _miss_segment(
        self,
        rng: np.random.Generator,
        *,
        source: str,
        horizon: int,
    ) -> Segment:
        members = self.event_positive[horizon][source]
        for _ in range(self.attempts):
            member = members[int(rng.integers(len(members)))]
            onsets = member.onset_minutes(horizon)
            onset = float(onsets[int(rng.integers(onsets.size))])
            low, high = _valid_starts(
                member,
                onset=onset,
                horizon=horizon,
                decisions=self.segment_decisions,
                warmup=self.warmup,
            )
            if low > high:
                continue
            start = int(rng.integers(low, high + 1))
            segment = _build_segment(
                member,
                horizon=horizon,
                start=start,
                decisions=self.segment_decisions,
                warmup=self.warmup,
            )
            if bool(segment.scored_mask.any()):
                return segment
        raise ValueError(
            f"no scorable segment for source {source} at horizon {horizon}; "
            "segment_decisions is too long for this cohort"
        )


@dataclass(frozen=True)
class SegmentObjectiveResult:
    """One update's Lagrangian and the two quantities it is built from."""

    loss: torch.Tensor
    event_miss: torch.Tensor
    false_alarms_per_patient_day: torch.Tensor
    miss_by_source: torch.Tensor
    fapd_by_source: torch.Tensor
    scored_event_count: int
    dual_value: float


def _segment_surrogate(
    probabilities: torch.Tensor,
    segment: Segment,
    *,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(event_credit, false_alarm_mass)` for one segment at one horizon.

    `patient_days=1` is a placeholder that satisfies the surrogate's positive-integer
    validator.  Its `false_alarms_per_patient_day` field is therefore **not** the
    segment's rate and is never read; the caller divides `false_alarm_mass` by the
    segment's exact fractional exposure instead.
    """

    onsets = torch.as_tensor(segment.supplied_onsets, dtype=torch.float64)
    surrogate = participant_event_surrogate(
        probabilities,
        torch.as_tensor(segment.minutes, dtype=torch.float64),
        onsets,
        horizon_minutes=float(horizon),
        patient_days=1,
        collapse_minutes=COLLAPSE_MINUTES,
        score_from=segment.warmup,
    )
    return surrogate.event_credit, surrogate.false_alarm_mass


def segment_objective(
    *,
    miss_probabilities: Sequence[torch.Tensor],
    miss_segments: Sequence[Segment],
    burden_probabilities: Mapping[int, Sequence[torch.Tensor]],
    burden_segments: Sequence[Segment],
    source_order: Sequence[str],
    dual_value: float,
    target_fapd: float = 0.5,
    horizons: Sequence[int] = SEGMENT_HORIZONS,
) -> SegmentObjectiveResult:
    """Form the frozen endpoint's Lagrangian from segment estimates.

    `L = mean_s event_miss_s + lambda * (mean_s FAPD_s - target)`, where both means
    are **equal-source** because the endpoint's estimand is an equal-source mean.
    This is the one place the construction departs from
    `source_event_regret_objective`, which takes a hard maximum of excess risks
    across sources and therefore optimises a different quantity.
    """

    order = {str(name): index for index, name in enumerate(source_order)}
    count = len(order)
    if count == 0:
        raise ValueError("source_order must be non-empty")
    if len(miss_probabilities) != len(miss_segments):
        raise ValueError("miss probabilities and segments must align")
    if len(burden_segments) != count:
        raise ValueError("burden requires exactly one segment per source")
    if float(dual_value) < 0:
        raise ValueError("dual_value must be non-negative")

    miss_terms: dict[tuple[int, int], torch.Tensor] = {}
    scored_total = 0
    for probabilities, segment in zip(miss_probabilities, miss_segments, strict=True):
        credit, _ = _segment_surrogate(
            probabilities, segment, horizon=segment.horizon
        )
        mask = torch.as_tensor(segment.scored_mask, dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError("a miss segment carries no scored event")
        scored = credit[mask]
        scored_total += int(scored.numel())
        key = (order[segment.participant.source_id], int(segment.horizon))
        if key in miss_terms:
            raise ValueError("one miss segment per source and horizon is required")
        miss_terms[key] = (1.0 - scored).mean()

    fapd_terms: dict[tuple[int, int], torch.Tensor] = {}
    for horizon in horizons:
        probabilities_at = burden_probabilities[int(horizon)]
        if len(probabilities_at) != count:
            raise ValueError("burden probabilities must cover every source")
        for probabilities, segment in zip(
            probabilities_at, burden_segments, strict=True
        ):
            _, false_alarm_mass = _segment_surrogate(
                probabilities, segment, horizon=int(horizon)
            )
            days = segment.exposure_days
            if not days > 0:
                raise ValueError("segment exposure must be positive")
            key = (order[segment.participant.source_id], int(horizon))
            if key in fapd_terms:
                raise ValueError("one burden segment per source is required")
            fapd_terms[key] = false_alarm_mass / days

    horizon_list = [int(value) for value in horizons]
    missing = [
        key
        for index in range(count)
        for horizon in horizon_list
        if (key := (index, horizon)) not in miss_terms or key not in fapd_terms
    ]
    if missing:
        raise ValueError(f"segment objective is missing cells: {sorted(set(missing))}")

    miss_by_source = torch.stack(
        [
            torch.stack([miss_terms[(index, h)] for h in horizon_list]).mean()
            for index in range(count)
        ]
    )
    fapd_by_source = torch.stack(
        [
            torch.stack([fapd_terms[(index, h)] for h in horizon_list]).mean()
            for index in range(count)
        ]
    )
    event_miss = miss_by_source.mean()
    fapd = fapd_by_source.mean()
    loss = event_miss + float(dual_value) * (fapd - float(target_fapd))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("segment objective became non-finite")
    return SegmentObjectiveResult(
        loss=loss,
        event_miss=event_miss,
        false_alarms_per_patient_day=fapd,
        miss_by_source=miss_by_source,
        fapd_by_source=fapd_by_source,
        scored_event_count=int(scored_total),
        dual_value=float(dual_value),
    )


class SegmentSurrogateTrainer:
    """Score sampled segments through the live model and form the Lagrangian.

    The surrogate is evaluated on CPU while the sequence forward stays on the
    training device.  Moving the probabilities across preserves the autograd graph,
    and the surrogate's inner loops are thousands of very small operations whose
    cost on an accelerator is kernel-launch overhead rather than arithmetic.
    """

    def __init__(
        self,
        *,
        dataset: object,
        sampler: SegmentPackSampler,
        device: torch.device,
        target_fapd: float = 0.5,
        forward_chunk: int = 1024,
        horizons: Sequence[int] = SEGMENT_HORIZONS,
        num_workers: int = 0,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.device = device
        self.target_fapd = float(target_fapd)
        self.forward_chunk = int(forward_chunk)
        self.horizons = tuple(int(value) for value in horizons)
        self.num_workers = int(num_workers)
        if self.forward_chunk <= 0:
            raise ValueError("forward_chunk must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.dual_value = 0.0
        self._epoch_fapd: list[float] = []
        self._fetch_generator = torch.Generator()
        self._fetch_index = 0

    def _pack_probabilities(
        self,
        model: nn.Module,
        segments: Sequence[Segment],
    ) -> list[dict[int, torch.Tensor]]:
        """Score every segment of one pack in a single batched pass.

        Sequence construction is the CPU-bound half of this package, and the main
        training loader parallelises it across eight workers. Fetching one segment at
        a time would put this path an order of magnitude behind the loop it rides on,
        and building a loader PER SEGMENT would spend more on worker startup than it
        saves. So the whole pack is fetched once, streamed in chunks so the fetch
        overlaps the forward, and split back afterwards.

        `shuffle=False` throughout: the refractory recurrence depends on decision
        order, so a shuffled fetch would silently compute a different quantity.
        """

        lengths = [int(segment.positions.size) for segment in segments]
        indices = [
            int(position) for segment in segments for position in segment.positions
        ]
        if self.num_workers > 0:
            # An OWNED generator, for the reason `deep_training.py:188` records: a
            # worker-seeded DataLoader with `generator=None` draws its worker base
            # seed from the GLOBAL torch RNG, so building one inside the training
            # loop would shift the main loader's stream and make the run's
            # reproducibility depend on how often the surrogate fired.
            self._fetch_generator.manual_seed(
                (self.sampler.seed * 1_000_003 + self._fetch_index) % (2**31 - 1)
            )
            self._fetch_index += 1
            source: object = DataLoader(
                Subset(self.dataset, indices),
                batch_size=self.forward_chunk,
                shuffle=False,
                num_workers=self.num_workers,
                generator=self._fetch_generator,
                collate_fn=_collate_sequences,
            )
        else:
            source = (
                torch.stack(
                    [
                        self.dataset[index]["sequence"]
                        for index in indices[start : start + self.forward_chunk]
                    ]
                )
                for start in range(0, len(indices), self.forward_chunk)
            )
        collected: dict[int, list[torch.Tensor]] = {h: [] for h in self.horizons}
        seen = 0
        # EVAL MODE, matching `c2_training.chunked_severe_logits`, which scores "a
        # complete trajectory in eval mode while retaining model gradients". The
        # surrogate estimates a DEPLOYMENT quantity -- the alarms this model would
        # raise in service -- and dropout is off in service. It also matters more
        # here than it would for a per-example loss: the refractory recurrence is a
        # product over neighbouring decisions, so dropout noise compounds through it.
        recurrent_modules = tuple(
            module
            for module in model.modules()
            if isinstance(module, (nn.RNN, nn.GRU, nn.LSTM))
        )
        recurrent_dropout = tuple(
            float(module.dropout) for module in recurrent_modules
        )
        was_training = model.training
        model.eval()
        # cuDNN does not retain the RNN reserve state for an eval-mode forward,
        # so backward would fail.  Keep the rest of the model in deployment mode
        # and re-arm only recurrent modules with their internal dropout disabled.
        for module in recurrent_modules:
            module.dropout = 0.0
            module.train()
        try:
            for chunk in source:
                batch = chunk.to(self.device)
                seen += int(batch.shape[0])
                representation = model.encode_sequence(batch)[:, -1]
                outputs = model.forward_from_representation(representation)
                for horizon in self.horizons:
                    task = SEVERE_SCORE_SOURCE_TASK[int(horizon)]
                    collected[horizon].append(outputs["logits"][task])
        finally:
            for module, dropout in zip(
                recurrent_modules, recurrent_dropout, strict=True
            ):
                module.dropout = dropout
            model.train(was_training)
        if seen != len(indices):
            raise ValueError(
                f"segment fetch returned {seen} rows for {len(indices)} positions"
            )
        joined = {
            horizon: torch.sigmoid(torch.cat(values).to(torch.float32).cpu())
            for horizon, values in collected.items()
        }
        out: list[dict[int, torch.Tensor]] = []
        offset = 0
        for length in lengths:
            out.append(
                {h: joined[h][offset : offset + length] for h in self.horizons}
            )
            offset += length
        return out

    def step(
        self,
        model: nn.Module,
        *,
        epoch: int,
        update_index: int,
    ) -> SegmentObjectiveResult:
        if bool(getattr(model, "supports_memory", False)):
            raise ValueError(
                "the segment surrogate scores segments without a memory bank; "
                "a memory-enabled arm would train and score through different paths"
            )
        pack = self.sampler.pack(epoch, update_index)
        segments = tuple(pack.miss) + tuple(pack.burden)
        scored = self._pack_probabilities(model, segments)
        miss_probabilities = [
            scored[index][segment.horizon]
            for index, segment in enumerate(pack.miss)
        ]
        offset = len(pack.miss)
        burden_at: dict[int, list[torch.Tensor]] = {h: [] for h in self.horizons}
        for index in range(len(pack.burden)):
            for horizon in self.horizons:
                burden_at[horizon].append(scored[offset + index][horizon])
        result = segment_objective(
            miss_probabilities=miss_probabilities,
            miss_segments=pack.miss,
            burden_probabilities=burden_at,
            burden_segments=pack.burden,
            source_order=self.sampler.source_order,
            dual_value=self.dual_value,
            target_fapd=self.target_fapd,
            horizons=self.horizons,
        )
        self._epoch_fapd.append(
            float(result.false_alarms_per_patient_day.detach().cpu())
        )
        return result

    def end_epoch(self, epoch: int) -> dict[str, float]:
        """Advance the frozen projected `1/sqrt(epoch)` train-only dual."""

        if not self._epoch_fapd:
            return {"dual_value": self.dual_value, "observed_fapd": float("nan")}
        observed = float(np.mean(self._epoch_fapd))
        previous = self.dual_value
        self.dual_value = update_fapd_dual(
            previous,
            epoch_index=int(epoch),
            observed_equal_source_fapd=observed,
            target_fapd=self.target_fapd,
        )
        self._epoch_fapd = []
        return {
            "epoch": int(epoch),
            "observed_equal_source_fapd": observed,
            "dual_before": previous,
            "dual_value": self.dual_value,
        }

    def stream_agreement(
        self,
        model: nn.Module,
        *,
        epoch: int,
        update_index: int,
    ) -> dict[str, float]:
        """G4: check the warm-up against the true stream instead of assuming it.

        For each burden segment the segment's reported alarm mass is compared with
        the same decisions' mass inside a much longer span. The warm-up is exact by
        measurement rather than by argument, so this reports the residual and would
        expose a cohort whose alarm rate is high enough to need a longer one.
        """

        pack = self.sampler.pack(epoch, update_index)
        errors: list[float] = []
        with torch.no_grad():
            for segment in pack.burden:
                participant = segment.participant
                reach = 4 * self.sampler.warmup
                begin = segment.start - reach
                if begin < 0:
                    continue
                decisions = segment.stop - segment.start - segment.warmup
                longer = _build_segment(
                    participant,
                    horizon=segment.horizon,
                    start=segment.start + segment.warmup,
                    decisions=decisions,
                    warmup=segment.warmup + reach,
                )
                scored = self._pack_probabilities(model, (segment, longer))
                masses = []
                for probabilities, candidate in zip(
                    scored, (segment, longer), strict=True
                ):
                    horizon = int(candidate.horizon)
                    _, mass = _segment_surrogate(
                        probabilities[horizon],
                        candidate,
                        horizon=horizon,
                    )
                    masses.append(float(mass))
                if masses[1] > 0:
                    errors.append(abs(masses[0] - masses[1]) / masses[1])
        if not errors:
            return {"segments": 0}
        return {
            "epoch": int(epoch),
            "segments": len(errors),
            "mean_relative_error": float(np.mean(errors)),
            "max_relative_error": float(np.max(errors)),
            "warmup_decisions": int(self.sampler.warmup),
        }
