"""Metadata-only participant cross-fitting for the prospective round2 correction.

The caller supplies the authoritative development identity inventory. No outcome,
CGM, action, or window values enter assignment. This module does not itself wire
outer evaluation into training or load any participant values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .split import assert_disjoint, participant_split

OUTER_FOLDS = 5
CROSSFIT_SEED = 20261109
_IDENTITY_COLUMNS = ["source_id", "participant_id"]
_INNER_ROLES = ("train", "selection", "calibration")
_HELD_OUT_SOURCES = ("Loop Observational Study", "HUPA-UCM", "OhioT1DM")


def _participant_keys(
    metadata: pd.DataFrame, *, sources: set[str], name: str
) -> set[tuple[str, str]]:
    if not set(_IDENTITY_COLUMNS).issubset(metadata.columns):
        raise ValueError(f"{name} is missing participant identity columns")
    identities = metadata[_IDENTITY_COLUMNS]
    for column in _IDENTITY_COLUMNS:
        if (
            identities[column].isna().any()
            or identities[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"{name} participant identities must be complete")
    identities = identities.astype(str)
    if identities.duplicated().any():
        raise ValueError(f"{name} contains duplicate source-scoped participants")
    unauthorized = set(identities["source_id"]).difference(sources)
    if unauthorized:
        raise ValueError(
            f"{name} contains unauthorized sources: {sorted(unauthorized)}"
        )
    return set(identities.itertuples(index=False, name=None))


def _require_same_keys(
    actual: set[tuple[str, str]], expected: set[tuple[str, str]], *, name: str
) -> None:
    missing = expected.difference(actual)
    unexpected = actual.difference(expected)
    if missing or unexpected:
        raise ValueError(
            f"{name} identity coverage mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )


@dataclass(frozen=True)
class ParticipantCrossfitPlan:
    """Immutable participant assignments with fold-specific inner role metadata."""

    source_order: tuple[str, ...]
    _assignments: tuple[tuple[str, str, int], ...]

    @property
    def assignments(self) -> pd.DataFrame:
        """Return fresh metadata so caller edits cannot alter the bound plan."""
        return pd.DataFrame(
            self._assignments, columns=[*_IDENTITY_COLUMNS, "outer_fold"]
        )

    def partitions(self, outer_fold: int) -> dict[str, set[tuple[str, str]]]:
        """Split the complement with the existing 70/15/15 participant helper."""
        if (
            isinstance(outer_fold, (bool, np.bool_))
            or not isinstance(outer_fold, (int, np.integer))
            or not 0 <= outer_fold < OUTER_FOLDS
        ):
            raise ValueError("outer_fold must be an integer in [0, 4]")
        assignments = self.assignments
        evaluation = assignments["outer_fold"].eq(outer_fold)
        complement = assignments.loc[~evaluation, _IDENTITY_COLUMNS]
        inner = participant_split(
            complement,
            development_sources=list(self.source_order),
            external_source=_HELD_OUT_SOURCES[0],
            forbidden_development_sources=list(_HELD_OUT_SOURCES),
            train_fraction=0.70,
            selection_fraction=0.15,
            calibration_fraction=0.15,
            include_holdouts=False,
            seed=CROSSFIT_SEED + int(outer_fold) + 1,
        )
        roles = {role: inner[role] for role in _INNER_ROLES}
        roles["outer_evaluation"] = set(
            assignments.loc[evaluation, _IDENTITY_COLUMNS].itertuples(
                index=False, name=None
            )
        )
        assert_disjoint(**roles)
        _require_same_keys(
            set().union(*roles.values()),
            {(source, participant) for source, participant, _ in self._assignments},
            name=f"outer fold {outer_fold}",
        )
        return roles

    def role_metadata(self, outer_fold: int) -> pd.DataFrame:
        """Return all participant roles without changing original split metadata."""
        roles = self.partitions(outer_fold)
        return pd.DataFrame(
            [
                (source, participant, outer_fold, role)
                for role, keys in roles.items()
                for source, participant in sorted(keys)
            ],
            columns=[*_IDENTITY_COLUMNS, "outer_fold", "role"],
        ).sort_values(_IDENTITY_COLUMNS, ignore_index=True)

    def guard_fit_metadata(
        self, metadata: pd.DataFrame, *, outer_fold: int, role: str
    ) -> None:
        """Check complete fit-role identities before a caller opens any values.

        Model fitting and train-derived preprocessing use ``train``. Temperature,
        ECDF and threshold fitting use ``calibration``. Selection is reserved for
        checkpoint decisions. All three roles exclude outer evaluation, but this
        guard deliberately admits only the two roles that fit quantities.
        """
        if role == "outer_evaluation":
            raise ValueError("outer_evaluation is forbidden for fitting")
        if role not in ("train", "calibration"):
            raise ValueError("fitting role must be train or calibration")
        self.guard_role_metadata(metadata, outer_fold=outer_fold, role=role)

    def guard_role_metadata(
        self, metadata: pd.DataFrame, *, outer_fold: int, role: str
    ) -> None:
        """Bind a complete role before loading its values, including selection."""
        if role not in (*_INNER_ROLES, "outer_evaluation"):
            raise ValueError("unknown cross-fitting role")
        expected = self.partitions(outer_fold)[role]
        actual = _participant_keys(
            metadata, sources=set(self.source_order), name="fit metadata"
        )
        _require_same_keys(actual, expected, name=f"{role} fit metadata")
        if "role" in metadata and not metadata["role"].eq(role).all():
            raise ValueError("fit metadata has a mismatched role")
        if "outer_fold" in metadata and not metadata["outer_fold"].eq(outer_fold).all():
            raise ValueError("fit metadata has a mismatched outer_fold")


def build_crossfit_assignments(
    participants: pd.DataFrame,
    *,
    expected_participants: pd.DataFrame,
    development_sources: Sequence[str],
) -> ParticipantCrossfitPlan:
    """Assign each authoritative source-scoped identity to exactly one outer fold.

    Sources and participant identities are sorted before one seeded NumPy shuffle
    per source. Round-robin assignment retains tiny sources without requiring
    every source in every evaluation fold. The inner helper's minimum per-source
    complement size is checked without dropping or reshuffling participants.
    """
    source_order = tuple(development_sources)
    if (
        not source_order
        or any(
            not isinstance(source, str) or not source.strip()
            for source in source_order
        )
        or len(set(source_order)) != len(source_order)
    ):
        raise ValueError("development_sources must contain unique nonempty names")
    sources = set(source_order)
    if sources.intersection(_HELD_OUT_SOURCES):
        raise ValueError("held-out sources cannot be authorized for cross-fitting")
    expected = _participant_keys(
        expected_participants, sources=sources, name="expected participants"
    )
    actual = _participant_keys(participants, sources=sources, name="participants")
    _require_same_keys(actual, expected, name="cross-fitting population")
    missing_sources = sources.difference(source for source, _ in expected)
    if missing_sources:
        raise ValueError(f"missing development sources: {sorted(missing_sources)}")

    rng = np.random.default_rng(CROSSFIT_SEED)
    assignments = []
    for source in sorted(sources):
        keys = sorted(key for key in actual if key[0] == source)
        rng.shuffle(keys)
        assignments.extend(
            (source, participant, index % OUTER_FOLDS)
            for index, (_, participant) in enumerate(keys)
        )
    plan = ParticipantCrossfitPlan(source_order, tuple(sorted(assignments)))
    # Validate every inner complement now, rather than discovering an impossible
    # tiny-source split only after a caller has started loading values.
    for fold in range(OUTER_FOLDS):
        plan.partitions(fold)
    return plan
