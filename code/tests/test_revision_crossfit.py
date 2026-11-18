from collections import Counter

import numpy as np
import pandas as pd
import pytest

from glycoshift.revision_crossfit import (
    CROSSFIT_SEED,
    OUTER_FOLDS,
    build_crossfit_assignments,
)
from glycoshift.split import participant_split


def _participants(counts: dict[str, int] | None = None) -> pd.DataFrame:
    counts = counts or {"small": 9, "large": 21}
    return pd.DataFrame(
        [
            (source, f"p{participant:04d}")
            for source, count in counts.items()
            for participant in range(count)
        ],
        columns=["source_id", "participant_id"],
    )


def _plan(participants: pd.DataFrame | None = None):
    participants = _participants() if participants is None else participants
    return build_crossfit_assignments(
        participants,
        expected_participants=participants,
        development_sources=list(participants["source_id"].unique()),
    )


def test_outer_assignment_matches_seeded_source_stratified_toy_oracle() -> None:
    participants = _participants()
    plan = _plan(participants)
    rng = np.random.default_rng(20261109)
    expected = []
    for source in ("large", "small"):
        identifiers = sorted(
            participants.loc[participants["source_id"].eq(source), "participant_id"]
        )
        shuffled = rng.permutation(identifiers)
        expected.extend(
            (source, identifier, index % 5)
            for index, identifier in enumerate(shuffled)
        )
    assert list(plan.assignments.itertuples(index=False, name=None)) == sorted(expected)
    assert CROSSFIT_SEED == 20261109
    assert OUTER_FOLDS == 5
    counts = plan.assignments.groupby(["source_id", "outer_fold"]).size()
    for source in ("small", "large"):
        assert counts.loc[source].max() - counts.loc[source].min() <= 1


def test_order_and_nonidentity_values_do_not_change_assignment_or_inner_roles() -> None:
    participants = _participants()
    reordered = participants.sample(frac=1, random_state=3).copy()
    reordered["outcome"] = np.arange(len(reordered))
    first = _plan(participants)
    second = build_crossfit_assignments(
        reordered,
        expected_participants=participants.iloc[::-1],
        development_sources=["large", "small"],
    )
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    for fold in range(5):
        assert first.partitions(fold) == second.partitions(fold)


def test_inner_helper_parity_disjoint_roles_and_full_coverage() -> None:
    participants = _participants()
    plan = _plan(participants)
    all_keys = set(participants.itertuples(index=False, name=None))
    evaluation_counts = Counter()
    for fold in range(5):
        roles = plan.partitions(fold)
        evaluation_counts.update(roles["outer_evaluation"])
        assert set(roles) == {"train", "selection", "calibration", "outer_evaluation"}
        assert sum(map(len, roles.values())) == len(all_keys)
        assert set().union(*roles.values()) == all_keys
        complement = participants[
            ~pd.MultiIndex.from_frame(participants).isin(roles["outer_evaluation"])
        ]
        oracle = participant_split(
            complement,
            development_sources=["small", "large"],
            external_source="Loop Observational Study",
            forbidden_development_sources=[
                "Loop Observational Study", "HUPA-UCM", "OhioT1DM"
            ],
            train_fraction=0.70,
            selection_fraction=0.15,
            calibration_fraction=0.15,
            include_holdouts=False,
            seed=20261109 + fold + 1,
        )
        for role in ("train", "selection", "calibration"):
            assert roles[role] == oracle[role]
            assert {source for source, _ in roles[role]} == {"small", "large"}
    assert evaluation_counts == Counter({key: 1 for key in all_keys})


def test_synthetic_1020_population_keeps_small_sources_and_scoped_ids() -> None:
    counts = dict(zip(
        ("AZT1D", "BrisT1D", "CTR3", "DCLP3", "DCLP5", "Flair", "IOBP2",
         "PEDAP", "ReplaceBG", "Shanghai T1DM", "T1D-UOM"),
        (23, 19, 30, 112, 100, 113, 328, 64, 208, 9, 14), strict=True,
    ))
    plan = _plan(_participants(counts))
    assert len(plan.assignments) == 1020
    evaluations = Counter()
    for fold in range(5):
        evaluations.update(plan.partitions(fold)["outer_evaluation"])
    assert len(evaluations) == 1020
    assert set(evaluations.values()) == {1}
    assert sum(source == "Shanghai T1DM" for source, _ in evaluations) == 9


@pytest.mark.parametrize("inventory", ["actual", "expected"])
def test_duplicate_identity_inventory_fails(inventory: str) -> None:
    participants = _participants()
    duplicate = pd.concat([participants, participants.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_crossfit_assignments(
            duplicate if inventory == "actual" else participants,
            expected_participants=(
                duplicate if inventory == "expected" else participants
            ),
            development_sources=["small", "large"],
        )


@pytest.mark.parametrize("column", ["source_id", "participant_id"])
@pytest.mark.parametrize("missing", [None, "", " "])
def test_incomplete_identity_fails(column: str, missing) -> None:
    participants = _participants()
    broken = participants.copy()
    broken.loc[0, column] = missing
    with pytest.raises(ValueError, match="complete"):
        build_crossfit_assignments(
            broken, expected_participants=participants,
            development_sources=["small", "large"],
        )


@pytest.mark.parametrize("kind", ["missing", "unexpected", "unauthorized"])
def test_population_inventory_difference_fails(kind: str) -> None:
    expected = _participants()
    actual = expected.copy()
    if kind == "missing":
        actual = actual.iloc[1:]
    elif kind == "unexpected":
        actual.loc[0, "participant_id"] = "unknown"
    else:
        actual.loc[0, "source_id"] = "OhioT1DM"
    with pytest.raises(ValueError, match="coverage mismatch|unauthorized"):
        build_crossfit_assignments(
            actual,
            expected_participants=expected,
            development_sources=["small", "large"],
        )


def test_held_out_source_cannot_be_declared_development() -> None:
    participants = _participants({"OhioT1DM": 9})
    with pytest.raises(ValueError, match="held-out"):
        _plan(participants)


def test_source_too_small_for_existing_inner_split_fails_without_reassignment() -> None:
    with pytest.raises(ValueError, match="at least three participants"):
        _plan(_participants({"tiny": 3}))


def test_fit_guard_rejects_outer_evaluation_before_accessing_input_metadata() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="outer_evaluation is forbidden"):
        plan.guard_fit_metadata(None, outer_fold=0, role="outer_evaluation")
    with pytest.raises(ValueError, match="train or calibration"):
        plan.guard_fit_metadata(None, outer_fold=0, role="selection")


def test_fit_guard_binds_complete_role_population_and_fold() -> None:
    plan = _plan()
    metadata = plan.role_metadata(0)
    for role in ("train", "calibration"):
        selected = metadata.loc[metadata["role"].eq(role)].copy()
        plan.guard_fit_metadata(selected, outer_fold=0, role=role)
        with pytest.raises(ValueError, match="coverage mismatch"):
            plan.guard_fit_metadata(selected.iloc[1:], outer_fold=0, role=role)
        contaminated = pd.concat([
            selected, metadata.loc[metadata["role"].eq("outer_evaluation")].iloc[[0]]
        ])
        with pytest.raises(ValueError, match="coverage mismatch"):
            plan.guard_fit_metadata(contaminated, outer_fold=0, role=role)
        selected["role"] = "outer_evaluation"
        with pytest.raises(ValueError, match="mismatched role"):
            plan.guard_fit_metadata(selected, outer_fold=0, role=role)
        selected["role"] = role
        selected["outer_fold"] = 1
        with pytest.raises(ValueError, match="mismatched outer_fold"):
            plan.guard_fit_metadata(selected, outer_fold=0, role=role)


def test_assignment_and_role_metadata_are_defensive_copies() -> None:
    plan = _plan()
    before = plan.partitions(0)
    assignments = plan.assignments
    assignments.loc[:, "outer_fold"] = 4
    roles = plan.role_metadata(0)
    roles.loc[:, "role"] = "train"
    assert plan.partitions(0) == before


@pytest.mark.parametrize("fold", [-1, 5, True, 0.5, "0", None])
def test_invalid_outer_fold_fails(fold) -> None:
    with pytest.raises(ValueError, match="outer_fold"):
        _plan().partitions(fold)
