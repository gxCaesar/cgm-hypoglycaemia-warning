from __future__ import annotations

import numpy as np
import pandas as pd


def participant_split(
    frame: pd.DataFrame,
    *,
    development_sources: list[str],
    external_source: str,
    forbidden_development_sources: list[str],
    validation_fraction: float | None = None,
    train_fraction: float | None = None,
    selection_fraction: float | None = None,
    calibration_fraction: float | None = None,
    stress_sources: list[str] | None = None,
    include_holdouts: bool = True,
    seed: int,
) -> dict[str, set[tuple[str, str]]]:
    development = set(development_sources)
    forbidden = set(forbidden_development_sources)
    if development.intersection(forbidden):
        raise ValueError("forbidden held-out source is listed for development")
    if external_source in development:
        raise ValueError("external source cannot be a development source")

    available_sources = set(frame["source_id"].astype(str).unique())
    missing_sources = development.difference(available_sources)
    if missing_sources:
        raise ValueError(f"missing development sources: {sorted(missing_sources)}")
    if include_holdouts and external_source not in available_sources:
        raise ValueError(f"external source is absent: {external_source}")

    stress = set(stress_sources or [])
    if stress.intersection(development):
        raise ValueError("stress sources cannot be development sources")
    if external_source in stress:
        raise ValueError("external source cannot also be a stress source")
    missing_stress = (
        stress.difference(available_sources)
        if include_holdouts
        else set()
    )
    if missing_stress:
        raise ValueError(f"missing stress sources: {sorted(missing_stress)}")

    legacy_split = validation_fraction is not None
    if legacy_split:
        train_fraction = 1.0 - float(validation_fraction)
        selection_fraction = float(validation_fraction)
        calibration_fraction = 0.0
    elif None in (train_fraction, selection_fraction, calibration_fraction):
        raise ValueError("three-way split fractions are required")

    assert train_fraction is not None
    assert selection_fraction is not None
    assert calibration_fraction is not None
    if abs(train_fraction + selection_fraction + calibration_fraction - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to one")

    rng = np.random.default_rng(seed)
    train: set[tuple[str, str]] = set()
    selection: set[tuple[str, str]] = set()
    calibration: set[tuple[str, str]] = set()
    for source in sorted(development):
        participants = sorted(
            {
                (source, str(participant))
                for participant in frame.loc[
                    frame["source_id"] == source, "participant_id"
                ].unique()
            }
        )
        rng.shuffle(participants)
        if calibration_fraction > 0 and len(participants) < 3:
            raise ValueError(
                f"source {source} needs at least three participants for a "
                "train/selection/calibration split"
            )
        selection_count = max(
            1,
            int(round(len(participants) * selection_fraction)),
        )
        calibration_count = (
            max(1, int(round(len(participants) * calibration_fraction)))
            if calibration_fraction > 0
            else 0
        )
        while selection_count + calibration_count >= len(participants):
            if selection_count >= calibration_count and selection_count > 1:
                selection_count -= 1
            elif calibration_count > 1:
                calibration_count -= 1
            else:
                raise ValueError(
                    f"cannot split source {source} without an empty train set"
                )
        selection.update(participants[:selection_count])
        calibration.update(
            participants[selection_count : selection_count + calibration_count]
        )
        train.update(participants[selection_count + calibration_count :])

    external = (
        {
            (external_source, str(participant))
            for participant in frame.loc[
                frame["source_id"] == external_source, "participant_id"
            ].unique()
        }
        if include_holdouts
        else set()
    )
    if legacy_split:
        partitions: dict[str, set[tuple[str, str]]] = {
            "train": train,
            "validation": selection,
            "external": external,
        }
    else:
        partitions = {
            "train": train,
            "selection": selection,
            "calibration": calibration,
            "external_loop": external,
        }
    for source in sorted(stress):
        partitions[f"stress_{source.lower().replace('-', '_')}"] = (
            {
                (source, str(participant))
                for participant in frame.loc[
                    frame["source_id"].astype(str) == source,
                    "participant_id",
                ].unique()
            }
            if include_holdouts
            else set()
        )
    assert_disjoint(**partitions)
    return partitions


def assert_disjoint(**partitions: set[tuple[str, str]]) -> None:
    names = list(partitions)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = partitions[left_name].intersection(partitions[right_name])
            if overlap:
                raise ValueError(
                    f"participant leakage between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:3]}"
                )
