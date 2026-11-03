from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

ACTION_FIELDS = (
    "basal_insulin_u",
    "bolus_insulin_u",
    "carbs_g",
    "activity_value",
)

ACTION_MASK_SUFFIXES = ("available", "observed", "zero_ambiguous")

REQUIRED_COLUMNS = (
    "source_id",
    "participant_id",
    "timestamp",
    "cgm_mg_dl",
    *(f"{field}_available" for field in ACTION_FIELDS),
    *(f"{field}_observed" for field in ACTION_FIELDS),
    *(f"{field}_zero_ambiguous" for field in ACTION_FIELDS),
    *ACTION_FIELDS,
)


@dataclass(frozen=True)
class DataAudit:
    rows: int
    participants: int
    sources: tuple[str, ...]
    duplicate_keys: int
    timestamp_violations: int
    glucose_out_of_range: int
    unknown_timezone_rows: int
    ambiguous_zero_rows: int
    cross_participant_duplicate_day_groups: int

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "participants": self.participants,
            "sources": list(self.sources),
            "duplicate_keys": self.duplicate_keys,
            "timestamp_violations": self.timestamp_violations,
            "glucose_out_of_range": self.glucose_out_of_range,
            "unknown_timezone_rows": self.unknown_timezone_rows,
            "ambiguous_zero_rows": self.ambiguous_zero_rows,
            "cross_participant_duplicate_day_groups": (
                self.cross_participant_duplicate_day_groups
            ),
        }


def cgm_day_fingerprints(
    frame: pd.DataFrame,
    *,
    minimum_samples: int = 96,
) -> list[tuple[str, tuple[str, str, str]]]:
    """Return source-independent hashes for sufficiently complete CGM days."""
    timestamp_column = (
        "timestamp_local" if "timestamp_local" in frame else "timestamp"
    )
    work = frame[
        ["source_id", "participant_id", timestamp_column, "cgm_mg_dl"]
    ].copy()
    work["_local_time"] = pd.to_datetime(
        work[timestamp_column],
        errors="coerce",
    )
    work["_day"] = work["_local_time"].dt.strftime("%Y-%m-%d")
    work["_clock_second"] = (
        work["_local_time"].dt.hour * 3600
        + work["_local_time"].dt.minute * 60
        + work["_local_time"].dt.second
    )
    work["_cgm_key"] = pd.to_numeric(
        work["cgm_mg_dl"],
        errors="coerce",
    ).round(4)

    fingerprints: list[tuple[str, tuple[str, str, str]]] = []
    for (source, participant, day), group in work.groupby(
        ["source_id", "participant_id", "_day"],
        sort=False,
        observed=False,
    ):
        if len(group) < minimum_samples:
            continue
        ordered = group.sort_values("_local_time")
        canonical = ordered[["_clock_second", "_cgm_key"]].to_csv(
            index=False,
            header=False,
            lineterminator="\n",
        )
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        fingerprints.append(
            (
                fingerprint,
                (str(source), str(participant), str(day)),
            )
        )
    return fingerprints


def cross_participant_duplicate_day_groups(
    frame: pd.DataFrame,
    *,
    minimum_samples: int = 96,
) -> list[list[tuple[str, str, str]]]:
    """Find exact CGM-day copies assigned to different participant keys."""
    fingerprints: dict[str, list[tuple[str, str, str]]] = {}
    for fingerprint, record in cgm_day_fingerprints(
        frame,
        minimum_samples=minimum_samples,
    ):
        fingerprints.setdefault(fingerprint, []).append(record)
    duplicates: list[list[tuple[str, str, str]]] = []
    for records in fingerprints.values():
        participant_keys = {
            (source, participant)
            for source, participant, _ in records
        }
        if len(participant_keys) > 1:
            duplicates.append(records)
    return duplicates


def validate_frame(frame: pd.DataFrame) -> DataAudit:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if timestamp.isna().any():
        raise ValueError("timestamp contains unparseable values")

    duplicate_keys = int(
        frame.duplicated(["source_id", "participant_id", "timestamp"]).sum()
    )
    if duplicate_keys:
        raise ValueError(f"duplicate participant-timestamp keys: {duplicate_keys}")

    ordered = frame.assign(timestamp=timestamp).sort_values(
        ["source_id", "participant_id", "timestamp"]
    )
    differences = ordered.groupby(
        ["source_id", "participant_id"],
        sort=False,
        observed=False,
    )["timestamp"].diff()
    timestamp_violations = int((differences <= pd.Timedelta(0)).sum())
    if timestamp_violations:
        raise ValueError(f"non-monotonic timestamps: {timestamp_violations}")

    glucose = pd.to_numeric(frame["cgm_mg_dl"], errors="coerce")
    if glucose.isna().any():
        raise ValueError("cgm_mg_dl contains non-numeric or missing values")
    glucose_out_of_range = int(((glucose < 20) | (glucose > 600)).sum())
    if glucose_out_of_range:
        raise ValueError(
            f"cgm_mg_dl contains {glucose_out_of_range} values outside 20–600"
        )
    if "cgm_lower_censored" in frame:
        lower_censored = frame["cgm_lower_censored"]
        if not lower_censored.isin([True, False]).all():
            raise ValueError("cgm_lower_censored must be boolean")
        if "cgm_interpolated" in frame and (
            lower_censored & frame["cgm_interpolated"].astype(bool)
        ).any():
            raise ValueError(
                "lower-censored CGM records cannot also be interpolated"
            )
        if "cgm_observed" in frame and (
            lower_censored & ~frame["cgm_observed"].astype(bool)
        ).any():
            raise ValueError(
                "lower-censored CGM records must remain observed"
            )

    for field in ACTION_FIELDS:
        available = frame[f"{field}_available"]
        observed = frame[f"{field}_observed"]
        zero_ambiguous = frame[f"{field}_zero_ambiguous"]
        if not available.isin([True, False]).all():
            raise ValueError(f"{field}_available must be boolean")
        if not observed.isin([True, False]).all():
            raise ValueError(f"{field}_observed must be boolean")
        if not zero_ambiguous.isin([True, False]).all():
            raise ValueError(f"{field}_zero_ambiguous must be boolean")
        impossible = observed & ~available
        if impossible.any():
            raise ValueError(
                f"{field} has observed values where the source field is unavailable"
            )
        values = frame[field]
        if values[~observed].notna().any():
            raise ValueError(
                f"{field} contains a value while its observed mask is false"
            )
        invalid_ambiguity = zero_ambiguous & (
            ~observed | pd.to_numeric(values, errors="coerce").ne(0)
        )
        if invalid_ambiguity.any():
            raise ValueError(
                f"{field}_zero_ambiguous requires an observed numeric zero"
            )

    participants = frame[["source_id", "participant_id"]].drop_duplicates()
    if "timezone_known" in frame:
        timezone_known = frame["timezone_known"]
        if not timezone_known.isin([True, False]).all():
            raise ValueError("timezone_known must be boolean")
        unknown_timezone_rows = int((~timezone_known).sum())
    else:
        unknown_timezone_rows = 0
    ambiguous_zero_rows = int(
        frame[
            [f"{field}_zero_ambiguous" for field in ACTION_FIELDS]
        ].any(axis=1).sum()
    )
    duplicate_day_groups = cross_participant_duplicate_day_groups(frame)
    if duplicate_day_groups:
        examples = duplicate_day_groups[:3]
        raise ValueError(
            "cross-participant duplicate CGM days detected: "
            f"{len(duplicate_day_groups)} groups; examples={examples}"
        )
    return DataAudit(
        rows=len(frame),
        participants=len(participants),
        sources=tuple(sorted(frame["source_id"].astype(str).unique())),
        duplicate_keys=duplicate_keys,
        timestamp_violations=timestamp_violations,
        glucose_out_of_range=glucose_out_of_range,
        unknown_timezone_rows=unknown_timezone_rows,
        ambiguous_zero_rows=ambiguous_zero_rows,
        cross_participant_duplicate_day_groups=len(duplicate_day_groups),
    )
