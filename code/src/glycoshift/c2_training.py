from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .event_reference_authority import (
    read_production_event_reference_artifact,
    validate_production_event_reference_artifact,
)
from .event_regret import (
    SourceEventRegretResult,
    aggregate_source_event_components,
    equal_weight_horizon_source_event_regret,
    participant_event_surrogate,
    update_fapd_dual,
)
from .event_streams import (
    EVENT_HORIZONS,
    AllSourceParticipantPackSampler,
    ParticipantEventPack,
    ParticipantEventStream,
)
from .mero_weighted_reference import (
    PRODUCTION_SOURCE_ORDER,
    read_weighted_reference_artifact,
    validate_weighted_reference_artifact,
)


@runtime_checkable
class DecisionSequenceSource(Protocol):
    """Lazy ordered decision contexts for one complete participant stream."""

    @property
    def decision_count(self) -> int: ...

    def load_chunk(self, start: int, stop: int) -> torch.Tensor: ...


@dataclass(frozen=True)
class IndexedDecisionSequenceSource:
    """Adapt an indexable window dataset without materializing a trajectory."""

    dataset: Any
    indices: tuple[int, ...] | None = None
    sequence_key: str = "sequence"

    def __post_init__(self) -> None:
        if self.indices is not None:
            if not self.indices or len(set(self.indices)) != len(self.indices):
                raise ValueError(
                    "decision sequence indices must be non-empty and unique"
                )
            if any(
                isinstance(index, bool)
                or not isinstance(index, Integral)
                or not 0 <= int(index) < len(self.dataset)
                for index in self.indices
            ):
                raise ValueError("decision sequence index is outside the dataset")

    @property
    def decision_count(self) -> int:
        return len(self.dataset) if self.indices is None else len(self.indices)

    def load_chunk(self, start: int, stop: int) -> torch.Tensor:
        if not 0 <= start < stop <= self.decision_count:
            raise ValueError("decision sequence chunk bounds are invalid")
        rows: list[torch.Tensor] = []
        for index in range(start, stop):
            dataset_index = index if self.indices is None else self.indices[index]
            item = self.dataset[dataset_index]
            if not isinstance(item, Mapping):
                raise TypeError("indexed decision dataset rows must be mappings")
            value = item.get(self.sequence_key)
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError("indexed decision rows must contain sequence matrices")
            rows.append(value)
        return torch.stack(rows)


SequenceProvider = Callable[
    [str, str],
    torch.Tensor | DecisionSequenceSource,
]


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return resolved


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field} must be positive")
    return resolved


def _source_order(values: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(values)
    if (
        not resolved
        or any(not isinstance(source, str) or not source for source in resolved)
        or len(set(resolved)) != len(resolved)
        or resolved != tuple(sorted(resolved))
    ):
        raise ValueError("source_order must be non-empty, unique, and lexical")
    return resolved


def _component_rows(
    values: Mapping[int, Sequence[float]],
    *,
    source_count: int,
    field: str,
    maximum: float | None = None,
) -> tuple[tuple[int, tuple[float, ...]], ...]:
    if set(values) != set(EVENT_HORIZONS):
        raise ValueError(f"{field} must cover exactly {EVENT_HORIZONS}")
    output: list[tuple[int, tuple[float, ...]]] = []
    for horizon in EVENT_HORIZONS:
        row = tuple(
            _finite(value, field=f"{field}[{horizon}]", minimum=0.0)
            for value in values[horizon]
        )
        if len(row) != source_count:
            raise ValueError(f"{field}[{horizon}] must match source_order")
        if maximum is not None and any(value > maximum for value in row):
            raise ValueError(f"{field}[{horizon}] exceeds {maximum}")
        output.append((horizon, row))
    return tuple(output)


@dataclass(frozen=True)
class C2ReferenceVectors:
    """Detached train-only base and event references for the C2 objective."""

    source_order: tuple[str, ...]
    base_source_risks: tuple[float, ...]
    event_miss_by_horizon: tuple[tuple[int, tuple[float, ...]], ...]
    false_alarms_by_horizon: tuple[tuple[int, tuple[float, ...]], ...]
    base_artifact_sha256: str
    event_artifact_sha256: str
    event_authority_sha256: str
    split_manifest_sha256: str
    window_manifest_sha256: str
    preprocessing_config_sha256: str
    base_input_manifest_sha256: str
    base_loss_contract_sha256: str
    event_loss_score_contract_sha256: str

    def __post_init__(self) -> None:
        sources = _source_order(self.source_order)
        if self.source_order != sources:
            raise ValueError("source_order must be stored canonically")
        if len(self.base_source_risks) != len(sources):
            raise ValueError("base_source_risks must match source_order")
        for value in self.base_source_risks:
            _finite(value, field="base_source_risk", minimum=0.0)
        canonical_misses = _component_rows(
            dict(self.event_miss_by_horizon),
            source_count=len(sources),
            field="event_miss_by_horizon",
            maximum=1.0,
        )
        if self.event_miss_by_horizon != canonical_misses:
            raise ValueError("event_miss_by_horizon must be stored canonically")
        canonical_burdens = _component_rows(
            dict(self.false_alarms_by_horizon),
            source_count=len(sources),
            field="false_alarms_by_horizon",
        )
        if self.false_alarms_by_horizon != canonical_burdens:
            raise ValueError("false_alarms_by_horizon must be stored canonically")
        for field in (
            "base_artifact_sha256",
            "event_artifact_sha256",
            "event_authority_sha256",
            "split_manifest_sha256",
            "window_manifest_sha256",
            "preprocessing_config_sha256",
            "base_input_manifest_sha256",
            "base_loss_contract_sha256",
            "event_loss_score_contract_sha256",
        ):
            _sha256(getattr(self, field), field=field)

    def tensor_components(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        if not dtype.is_floating_point:
            raise TypeError("C2 reference tensors require a floating-point dtype")
        base = torch.tensor(self.base_source_risks, device=device, dtype=dtype)
        misses = {
            horizon: torch.tensor(values, device=device, dtype=dtype)
            for horizon, values in self.event_miss_by_horizon
        }
        burdens = {
            horizon: torch.tensor(values, device=device, dtype=dtype)
            for horizon, values in self.false_alarms_by_horizon
        }
        return base.detach(), {
            horizon: value.detach() for horizon, value in misses.items()
        }, {horizon: value.detach() for horizon, value in burdens.items()}

    def receipt(self) -> dict[str, Any]:
        return {
            "mode": "weighted_cf5_c2_event_reference_v1_3",
            "source_order": list(self.source_order),
            "base_artifact_sha256": self.base_artifact_sha256,
            "event_artifact_sha256": self.event_artifact_sha256,
            "event_authority_sha256": self.event_authority_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "window_manifest_sha256": self.window_manifest_sha256,
            "preprocessing_config_sha256": self.preprocessing_config_sha256,
            "base_input_manifest_sha256": self.base_input_manifest_sha256,
            "base_loss_contract_sha256": self.base_loss_contract_sha256,
            "event_loss_score_contract_sha256": (
                self.event_loss_score_contract_sha256
            ),
        }


def _c2_reference_vectors_from_validated_artifacts(
    *,
    base_artifact: Mapping[str, Any],
    event_artifact: Mapping[str, Any],
) -> C2ReferenceVectors:
    """Bind the event reference to its exact weighted base parent."""

    base = dict(base_artifact)
    event = dict(event_artifact)
    sources = tuple(base["lexical_source_order"])
    if sources != PRODUCTION_SOURCE_ORDER:
        raise ValueError("C2 requires the frozen production source order")
    if tuple(event["lexical_source_order"]) != sources:
        raise ValueError("C2 base and event source orders disagree")
    bindings = event["bindings"]
    if bindings["parent_base_reference_artifact_sha256"] != base["artifact_sha256"]:
        raise ValueError("C2 event reference is not bound to the base artifact")
    if bindings["window_manifest_sha256"] != base["window_manifest_sha256"]:
        raise ValueError("C2 base and event window-manifest bindings disagree")

    components = event["reference_components"]
    misses: dict[int, tuple[float, ...]] = {}
    burdens: dict[int, tuple[float, ...]] = {}
    if set(components) != set(sources):
        raise ValueError("C2 event reference does not cover every source")
    for horizon in EVENT_HORIZONS:
        key = str(horizon)
        if any(set(components[source]) != {"30", "60"} for source in sources):
            raise ValueError("C2 event reference does not cover both horizons")
        misses[horizon] = tuple(
            float(components[source][key]["event_miss"]) for source in sources
        )
        burdens[horizon] = tuple(
            float(components[source][key]["false_alarms_per_patient_day"])
            for source in sources
        )
    return C2ReferenceVectors(
        source_order=sources,
        base_source_risks=tuple(
            float(base["per_source_reference_risks"][source])
            for source in sources
        ),
        event_miss_by_horizon=_component_rows(
            misses,
            source_count=len(sources),
            field="event_miss_by_horizon",
            maximum=1.0,
        ),
        false_alarms_by_horizon=_component_rows(
            burdens,
            source_count=len(sources),
            field="false_alarms_by_horizon",
        ),
        base_artifact_sha256=base["artifact_sha256"],
        event_artifact_sha256=event["artifact_sha256"],
        event_authority_sha256=bindings["expected_authority_sha256"],
        split_manifest_sha256=bindings["split_manifest_sha256"],
        window_manifest_sha256=bindings["window_manifest_sha256"],
        preprocessing_config_sha256=bindings["preprocessing_config_sha256"],
        base_input_manifest_sha256=base["input_manifest_sha256"],
        base_loss_contract_sha256=base["loss_contract_sha256"],
        event_loss_score_contract_sha256=bindings[
            "loss_score_contract_sha256"
        ],
    )


def c2_reference_vectors_from_artifacts(
    *,
    base_artifact: Mapping[str, Any],
    event_artifact: Mapping[str, Any],
) -> C2ReferenceVectors:
    """Validate supplied artifacts before binding their reference vectors."""

    return _c2_reference_vectors_from_validated_artifacts(
        base_artifact=validate_weighted_reference_artifact(base_artifact),
        event_artifact=validate_production_event_reference_artifact(
            event_artifact
        ),
    )


def load_c2_reference_vectors(
    *,
    base_reference_path: str | Path,
    event_reference_path: str | Path,
    expected_sha256: Mapping[str, str],
) -> C2ReferenceVectors:
    """Read canonical artifacts and reject every caller pin mismatch."""

    required = {
        "base_artifact_sha256",
        "event_artifact_sha256",
        "event_authority_sha256",
        "split_manifest_sha256",
        "window_manifest_sha256",
        "preprocessing_config_sha256",
        "base_input_manifest_sha256",
        "base_loss_contract_sha256",
        "event_loss_score_contract_sha256",
    }
    if set(expected_sha256) != required:
        raise ValueError("C2 expected SHA-256 pins differ from the frozen schema")
    reference = _c2_reference_vectors_from_validated_artifacts(
        base_artifact=read_weighted_reference_artifact(base_reference_path),
        event_artifact=read_production_event_reference_artifact(
            event_reference_path
        ),
    )
    receipt = reference.receipt()
    for field in sorted(required):
        expected = _sha256(expected_sha256[field], field=f"expected.{field}")
        if receipt[field] != expected:
            raise ValueError(f"C2 reference pin mismatch: {field}")
    return reference


def chunked_severe_logits(
    model: nn.Module,
    sequences: torch.Tensor | DecisionSequenceSource,
    *,
    chunk_size: int,
    use_checkpoint: bool = True,
) -> dict[int, torch.Tensor]:
    """Score a complete trajectory in eval mode while retaining model gradients."""

    size = _positive_integer(chunk_size, field="chunk_size")
    if isinstance(sequences, torch.Tensor):
        if sequences.ndim != 3 or sequences.shape[0] == 0:
            raise ValueError(
                "sequences must be a non-empty [decision, time, feature] tensor"
            )
        decision_count = int(sequences.shape[0])

        def load_chunk(start: int, stop: int) -> torch.Tensor:
            return sequences[start:stop]

    elif isinstance(sequences, DecisionSequenceSource):
        decision_count = _positive_integer(
            sequences.decision_count,
            field="decision_count",
        )
        load_chunk = sequences.load_chunk
    else:
        raise TypeError("sequences must be a tensor or DecisionSequenceSource")
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise ValueError("C2 model must contain trainable parameters") from error
    device = parameter.device
    was_training = model.training
    chunks: list[torch.Tensor] = []

    def severe_logits(values: torch.Tensor) -> torch.Tensor:
        recompute_was_training = model.training
        model.eval()
        try:
            outputs = model(values)
        finally:
            model.train(recompute_was_training)
        if not isinstance(outputs, Mapping) or not isinstance(
            outputs.get("logits"), Mapping
        ):
            raise ValueError("C2 model must return a logits mapping")
        logits = outputs["logits"]
        rows: list[torch.Tensor] = []
        for horizon in EVENT_HORIZONS:
            value = logits.get(f"hypo_{horizon}")
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise ValueError("C2 model severe logits must be vectors")
            rows.append(value)
        return torch.stack(rows, dim=1).to(dtype=torch.float32)

    model.eval()
    try:
        for start in range(0, decision_count, size):
            values = load_chunk(start, min(start + size, decision_count))
            if (
                not isinstance(values, torch.Tensor)
                or values.ndim != 3
                or int(values.shape[0]) != min(size, decision_count - start)
                or not values.is_floating_point()
                or not bool(torch.isfinite(values).all())
            ):
                raise ValueError(
                    "decision sequence chunks must be complete finite 3D tensors"
                )
            values = values.to(device=device)
            if use_checkpoint and torch.is_grad_enabled():
                logits = checkpoint(
                    severe_logits,
                    values,
                    use_reentrant=False,
                    determinism_check="default",
                )
            else:
                logits = severe_logits(values)
            chunks.append(logits)
    finally:
        model.train(was_training)
    logits = torch.cat(chunks, dim=0)
    if logits.shape != (decision_count, len(EVENT_HORIZONS)):
        raise AssertionError("C2 severe-score shape drift")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("C2 severe logits became non-finite")
    return {
        horizon: logits[:, index]
        for index, horizon in enumerate(EVENT_HORIZONS)
    }


def chunked_severe_probabilities(
    model: nn.Module,
    sequences: torch.Tensor | DecisionSequenceSource,
    *,
    chunk_size: int,
    use_checkpoint: bool = True,
) -> dict[int, torch.Tensor]:
    """Apply the frozen float32 sigmoid to complete-trajectory raw logits."""

    return {
        horizon: torch.sigmoid(logits)
        for horizon, logits in chunked_severe_logits(
            model,
            sequences,
            chunk_size=chunk_size,
            use_checkpoint=use_checkpoint,
        ).items()
    }


@dataclass(frozen=True)
class C2EventUpdateResult:
    pack: ParticipantEventPack
    objective: SourceEventRegretResult
    event_miss_by_horizon: dict[int, torch.Tensor]
    false_alarms_by_horizon: dict[int, torch.Tensor]
    equal_source_fapd: torch.Tensor


class C2ObjectiveCoordinator:
    """Transactional complete-stream C1/C2 updates with exact resume."""

    STATE_VERSION = "1.1"

    def __init__(
        self,
        *,
        model: nn.Module,
        reference: C2ReferenceVectors,
        sampler: AllSourceParticipantPackSampler,
        streams: Mapping[tuple[str, str], ParticipantEventStream],
        sequence_provider: SequenceProvider,
        event_scale: Real,
        target_fapd: Real = 0.5,
        chunk_size: int = 256,
        use_checkpoint: bool = True,
        initial_dual_value: Real = 0.0,
        update_dual: bool = True,
    ) -> None:
        if tuple(sampler.source_order) != reference.source_order:
            raise ValueError("C2 sampler and reference source orders disagree")
        self.model = model
        self.reference = reference
        self.sampler = sampler
        self.sequence_provider = sequence_provider
        self.event_scale = _finite(
            event_scale,
            field="event_scale",
            minimum=0.0,
        )
        self.update_dual = bool(update_dual)
        if (self.event_scale == 0.0) != (not self.update_dual):
            raise ValueError(
                "event_scale must be zero exactly when dual updates are disabled"
            )
        self.target_fapd = _finite(
            target_fapd,
            field="target_fapd",
            minimum=0.0,
        )
        self.chunk_size = _positive_integer(chunk_size, field="chunk_size")
        self.use_checkpoint = bool(use_checkpoint)
        self.dual_value = _finite(
            initial_dual_value,
            field="initial_dual_value",
            minimum=0.0,
        )
        if not self.update_dual and self.dual_value != 0.0:
            raise ValueError("disabled dual updates require initial_dual_value=0")
        expected_keys = {
            (source, participant)
            for source in reference.source_order
            for participant in sampler.all_participants[source]
        }
        if set(streams) != expected_keys:
            raise ValueError("C2 streams must exactly cover the all-participant pools")
        self.streams = dict(streams)
        for key, stream in self.streams.items():
            if key != (stream.source_id, stream.participant_id):
                raise ValueError("C2 stream key disagrees with stream identity")
            if stream.partition != "train":
                raise ValueError("C2 streams must be train-only")
            for horizon in EVENT_HORIZONS:
                observed_positive = stream.event_count(horizon) > 0
                listed_positive = stream.participant_id in set(
                    sampler.event_positive_participants[horizon][stream.source_id]
                )
                if observed_positive != listed_positive:
                    raise ValueError("C2 event-positive pools disagree with streams")
        self.epoch = 0
        self.next_update_index = 0
        self._committed_equal_source_fapd: list[float] = []
        self.last_complete_train_fapd: float | None = None

    def expected_pack(self) -> ParticipantEventPack:
        if self.next_update_index >= len(self.sampler):
            raise RuntimeError("C2 epoch is complete; call finish_epoch")
        self.sampler.set_epoch(self.epoch)
        for pack in self.sampler:
            if pack.update_index == self.next_update_index:
                return pack
        raise AssertionError("C2 sampler did not yield the expected update")

    def compute_update(
        self,
        *,
        pack: ParticipantEventPack,
        base_source_risks: torch.Tensor,
    ) -> C2EventUpdateResult:
        expected = self.expected_pack()
        if pack != expected:
            raise ValueError("C2 pack is not the next deterministic update")
        if (
            base_source_risks.ndim != 1
            or base_source_risks.numel() != len(self.reference.source_order)
            or not base_source_risks.is_floating_point()
            or not bool(torch.isfinite(base_source_risks).all())
        ):
            raise ValueError("base_source_risks must be a finite source vector")

        participant_keys = set(pack.burden_participants)
        for horizon in EVENT_HORIZONS:
            participant_keys.update(pack.miss_participants_for(horizon))
        probabilities: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
        for source, participant in sorted(participant_keys):
            stream = self.streams[(source, participant)]
            sequences = self.sequence_provider(source, participant)
            if isinstance(sequences, torch.Tensor):
                observed_count = int(sequences.shape[0])
            elif isinstance(sequences, DecisionSequenceSource):
                observed_count = sequences.decision_count
            else:
                raise TypeError(
                    "C2 sequence_provider must return a tensor or lazy source"
                )
            if observed_count != stream.decision_count:
                raise ValueError("C2 sequence count disagrees with stream authority")
            probabilities[(source, participant)] = chunked_severe_probabilities(
                self.model,
                sequences,
                chunk_size=self.chunk_size,
                use_checkpoint=self.use_checkpoint,
            )

        source_to_index = {
            source: index for index, source in enumerate(self.reference.source_order)
        }
        misses: dict[int, torch.Tensor] = {}
        burdens: dict[int, torch.Tensor] = {}
        for horizon in EVENT_HORIZONS:
            miss_pairs = pack.miss_participants_for(horizon)
            burden_pairs = pack.burden_participants
            miss_surrogates = tuple(
                participant_event_surrogate(
                    probabilities[key][horizon],
                    self.streams[key].decision_minutes,
                    self.streams[key].event_onset_minutes(horizon),
                    horizon_minutes=horizon,
                    patient_days=self.streams[key].patient_days,
                )
                for key in miss_pairs
            )
            burden_surrogates = tuple(
                participant_event_surrogate(
                    probabilities[key][horizon],
                    self.streams[key].decision_minutes,
                    self.streams[key].event_onset_minutes(horizon),
                    horizon_minutes=horizon,
                    patient_days=self.streams[key].patient_days,
                )
                for key in burden_pairs
            )
            device = base_source_risks.device
            aggregate = aggregate_source_event_components(
                miss_participants=miss_surrogates,
                miss_source_indices=torch.tensor(
                    [source_to_index[source] for source, _ in miss_pairs],
                    device=device,
                    dtype=torch.long,
                ),
                burden_participants=burden_surrogates,
                burden_source_indices=torch.tensor(
                    [source_to_index[source] for source, _ in burden_pairs],
                    device=device,
                    dtype=torch.long,
                ),
                source_count=len(self.reference.source_order),
            )
            misses[horizon] = aggregate.event_miss
            burdens[horizon] = aggregate.false_alarms_per_patient_day

        reference_base, reference_miss, reference_burden = (
            self.reference.tensor_components(
                device=base_source_risks.device,
                dtype=base_source_risks.dtype,
            )
        )
        objective = equal_weight_horizon_source_event_regret(
            base_source_risks=base_source_risks,
            reference_base_risks=reference_base,
            event_miss_by_horizon=misses,
            reference_event_miss_by_horizon=reference_miss,
            false_alarms_by_horizon=burdens,
            reference_false_alarms_by_horizon=reference_burden,
            event_scale=self.event_scale,
            dual_value=self.dual_value,
            target_fapd=self.target_fapd,
        )
        equal_source_fapd = torch.stack(
            [burdens[horizon].mean() for horizon in EVENT_HORIZONS]
        ).mean()
        return C2EventUpdateResult(
            pack=pack,
            objective=objective,
            event_miss_by_horizon=misses,
            false_alarms_by_horizon=burdens,
            equal_source_fapd=equal_source_fapd,
        )

    def commit_update(self, result: C2EventUpdateResult) -> None:
        if result.pack != self.expected_pack():
            raise ValueError("cannot commit a stale or out-of-order C2 update")
        observed = float(result.equal_source_fapd.detach().cpu())
        if not math.isfinite(observed) or observed < 0:
            raise FloatingPointError(
                "committed C2 FAPD must be finite and non-negative"
            )
        self._committed_equal_source_fapd.append(observed)
        self.next_update_index += 1

    def measure_complete_train_fapd(self) -> float:
        """Measure the detached equal-source macro FAPD at the current model."""

        participant_values = {
            horizon: {source: [] for source in self.reference.source_order}
            for horizon in EVENT_HORIZONS
        }
        with torch.no_grad():
            for source in self.reference.source_order:
                for participant in self.sampler.all_participants[source]:
                    stream = self.streams[(source, participant)]
                    sequences = self.sequence_provider(source, participant)
                    if isinstance(sequences, torch.Tensor):
                        observed_count = int(sequences.shape[0])
                    elif isinstance(sequences, DecisionSequenceSource):
                        observed_count = sequences.decision_count
                    else:
                        raise TypeError(
                            "C2 sequence_provider must return a tensor or lazy source"
                        )
                    if observed_count != stream.decision_count:
                        raise ValueError(
                            "C2 sequence count disagrees with stream authority"
                        )
                    probabilities = chunked_severe_probabilities(
                        self.model,
                        sequences,
                        chunk_size=self.chunk_size,
                        use_checkpoint=False,
                    )
                    for horizon in EVENT_HORIZONS:
                        surrogate = participant_event_surrogate(
                            probabilities[horizon],
                            stream.decision_minutes,
                            stream.event_onset_minutes(horizon),
                            horizon_minutes=horizon,
                            patient_days=stream.patient_days,
                        )
                        value = float(
                            surrogate.false_alarms_per_patient_day.detach().cpu()
                        )
                        if not math.isfinite(value) or value < 0:
                            raise FloatingPointError(
                                "complete-train participant FAPD is invalid"
                            )
                        participant_values[horizon][source].append(value)

        horizon_means: list[float] = []
        for horizon in EVENT_HORIZONS:
            source_means = [
                math.fsum(participant_values[horizon][source])
                / len(participant_values[horizon][source])
                for source in self.reference.source_order
            ]
            horizon_means.append(math.fsum(source_means) / len(source_means))
        observed = math.fsum(horizon_means) / len(horizon_means)
        if not math.isfinite(observed) or observed < 0:
            raise FloatingPointError("complete-train equal-source FAPD is invalid")
        return observed

    def finish_epoch(self) -> float:
        if self.next_update_index != len(self.sampler):
            raise RuntimeError("cannot finish C2 epoch before every update commits")
        if len(self._committed_equal_source_fapd) != len(self.sampler):
            raise AssertionError("C2 committed-update ledger drift")
        observed = self.measure_complete_train_fapd()
        self.last_complete_train_fapd = observed
        if self.update_dual:
            self.dual_value = update_fapd_dual(
                self.dual_value,
                epoch_index=self.epoch,
                observed_equal_source_fapd=observed,
                target_fapd=self.target_fapd,
            )
        elif self.dual_value != 0.0:
            raise AssertionError("disabled dual state drifted from zero")
        self.epoch += 1
        self.next_update_index = 0
        self._committed_equal_source_fapd = []
        return self.dual_value

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_VERSION,
            "source_order": list(self.reference.source_order),
            "base_artifact_sha256": self.reference.base_artifact_sha256,
            "event_artifact_sha256": self.reference.event_artifact_sha256,
            "sampler_seed": self.sampler.seed,
            "updates_per_epoch": len(self.sampler),
            "event_scale": self.event_scale,
            "update_dual": self.update_dual,
            "target_fapd": self.target_fapd,
            "chunk_size": self.chunk_size,
            "use_checkpoint": self.use_checkpoint,
            "epoch": self.epoch,
            "next_update_index": self.next_update_index,
            "dual_value": self.dual_value,
            "last_complete_train_fapd": self.last_complete_train_fapd,
            "committed_equal_source_fapd": list(
                self._committed_equal_source_fapd
            ),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        expected_fields = set(self.state_dict())
        if set(payload) != expected_fields:
            raise ValueError("C2 resume-state fields differ from the frozen schema")
        immutable = self.state_dict()
        for field in (
            "schema_version",
            "source_order",
            "base_artifact_sha256",
            "event_artifact_sha256",
            "sampler_seed",
            "updates_per_epoch",
            "event_scale",
            "update_dual",
            "target_fapd",
            "chunk_size",
            "use_checkpoint",
        ):
            if payload[field] != immutable[field]:
                raise ValueError(f"C2 resume-state contract mismatch: {field}")
        epoch = payload["epoch"]
        next_update = payload["next_update_index"]
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, Integral)
            or int(epoch) < 0
            or isinstance(next_update, bool)
            or not isinstance(next_update, Integral)
            or not 0 <= int(next_update) <= len(self.sampler)
        ):
            raise ValueError("C2 resume epoch/update indices are invalid")
        ledger = tuple(
            _finite(value, field="committed_equal_source_fapd", minimum=0.0)
            for value in payload["committed_equal_source_fapd"]
        )
        if len(ledger) != int(next_update):
            raise ValueError("C2 resume ledger does not match next_update_index")
        self.epoch = int(epoch)
        self.next_update_index = int(next_update)
        self.dual_value = _finite(
            payload["dual_value"],
            field="dual_value",
            minimum=0.0,
        )
        if not self.update_dual and self.dual_value != 0.0:
            raise ValueError("disabled dual resume state must remain zero")
        last_complete = payload["last_complete_train_fapd"]
        self.last_complete_train_fapd = (
            None
            if last_complete is None
            else _finite(
                last_complete,
                field="last_complete_train_fapd",
                minimum=0.0,
            )
        )
        self._committed_equal_source_fapd = list(ledger)
