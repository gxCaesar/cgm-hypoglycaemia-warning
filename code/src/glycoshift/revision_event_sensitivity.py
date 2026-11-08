"""G05 outcome-catalogue sensitivity with already-emitted selection alarms.

No score, fit, policy or refractory sequence is recomputed. Original online
eligibility is retained for reachability; the emissions are never regenerated.
Only event truth and its greedy strictly-future one-to-one matches change. These are
descriptive sensitivities with variant-specific catalogues, not same-event
comparisons. An adapter's observed flag cannot establish raw acquisition
provenance for source-preprocessed data.
Primary onset-confirmation counts describe the first three readings at each
primary horizon-base event's onset, not later runs inside the merged event.

From code/: ``PYTHONPATH=src python -m glycoshift.revision_event_sensitivity
--config accepted.json``. The explicit JSON contains revision_contract,
partition="selection", prepared_dir, replay_config (file binding), output_dir,
preparation_started_at, input_acceptance and runs. input_acceptance requires
status="accepted", binding_time="current_sensitivity_acceptance", evidence
(file bindings), normalized_files and prepared_files (file bindings). The latter
must cover exactly the selection normalized shards and prepared audit, manifest,
selection timelines/Ebase/Dbase. They are current accepted bindings, never
represented as preparation-time bindings. Normalized mtimes must precede the
declared preparation start; replay parity and start/end bindings are mandatory.

Each runs entry has arm, model_seed, run_dir, exactly one model_summary or
composition_recipe (binding), accepted_evidence (bindings), ledger_root and
conditions. Model arms use the producing summary's architecture/baseline name;
composition uses arm="composition" and the recipe's comparison_lstm_seed.
Each condition has
horizon_minutes, glucose_cut, nominal_fapd and bound alarms/events/participants
Parquet files from the accepted primary run. Only selection is allowed. Private
participant/catalogue/alarm/event ledgers stay private; exportable/ has aggregates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .config import load_config, resolve_path
from .events import Event, detect_events
from .fixed_frame_evaluation import _aggregate, _online_eligible
from .io import _validate_real_data_provenance
from .revision_external import _bound_file, _json, _parquet, _plain, _recheck_bindings
from .revision_inputs import build_participant_replay
from .revision_prepare import _fragment_frame
from .schema import validate_frame
from .sequences import REVISION_CONTRACT

HORIZONS = (30, 60)
VARIANTS = ("observed_only", "strict_merge_boundary")
_IDS = ["source_id", "participant_id"]
_KEYS = [*_IDS, "context_end"]
_TRUTH = [*_IDS, "event_id", "event_onset"]
_MATCH = [
    "matched",
    "unmatched",
    "matched_event_id",
    "matched_event_onset",
    "lead_minutes",
]
_COVERAGE = (5, 10, 15, 20)
_FORBIDDEN = {"Loop Observational Study", "HUPA-UCM", "OhioT1DM"}


def _boolean(frame: pd.DataFrame, name: str) -> np.ndarray:
    if (
        name not in frame
        or frame[name].isna().any()
        or not (pd.api.types.is_bool_dtype(frame[name].dtype))
    ):
        raise ValueError(f"complete boolean provenance is required: {name}")
    return frame[name].to_numpy(bool)


def _identity(frame: pd.DataFrame, key: tuple[str, str], *, empty=False) -> None:
    if not set(_IDS).issubset(frame) or (frame.empty and not empty):
        raise ValueError("participant identity columns/rows are missing")
    if frame[_IDS].isna().any().any() or any(
        not frame[name].astype(str).eq(value).all()
        for name, value in zip(_IDS, key, strict=True)
    ):
        raise ValueError("source-scoped participant identity differs")


def _equal(left: pd.DataFrame, right: pd.DataFrame, columns: list[str], name: str):
    if not set(columns).issubset(left) or not set(columns).issubset(right):
        raise ValueError(f"{name} required scientific columns are missing")
    a, b = left[columns].copy(), right[columns].copy()
    for column in columns:
        if column == "patient_day":
            for table in (a, b):
                table[column] = table[column].map(
                    lambda value: pd.Timestamp(pd.Timestamp(value).date(), tz="UTC")
                )
        elif column == "base_patient_days":
            for table in (a, b):
                table[column] = table[column].map(
                    lambda values: tuple(pd.Timestamp(v).date() for v in values)
                )
        elif column == "matched_lead_minutes":
            a[column] = a[column].map(tuple)
            b[column] = b[column].map(tuple)
        elif column.endswith(("timestamp", "context_end", "context_start", "onset")):
            a[column] = pd.to_datetime(a[column], utc=True)
            b[column] = pd.to_datetime(b[column], utc=True)
    try:
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True),
            b.reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
            check_exact=True,
        )
    except AssertionError:
        raise ValueError(f"{name} differs from accepted primary data") from None


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*_IDS, "timestamp", "cgm_mg_dl"}
    if frame.empty or not required.issubset(frame):
        raise ValueError("normalized participant columns/rows are missing")
    out = frame.sort_values("timestamp").reset_index(drop=True).copy()
    if len(out[_IDS].drop_duplicates()) != 1 or out[_IDS].isna().any().any():
        raise ValueError("normalized input must contain one complete participant")
    out["timestamp"] = pd.to_datetime(out.timestamp, utc=True, errors="coerce")
    if out.timestamp.isna().any() or out.timestamp.duplicated().any():
        raise ValueError("normalized timestamps must be complete and unique")
    glucose = pd.to_numeric(out.cgm_mg_dl, errors="coerce").to_numpy(float)
    if not np.isfinite(glucose).all():
        raise ValueError("normalized glucose must be finite; no provenance inference")
    observed = _boolean(out, "cgm_observed")
    interpolated = _boolean(out, "cgm_interpolated")
    _boolean(out, "cgm_lower_censored")
    if np.any(observed & interpolated):
        raise ValueError("observed and interpolated provenance conflict")
    return out


def build_variant_catalogue(normalized: pd.DataFrame, variant: str) -> list[Event]:
    """Change only observed support or the exact merge boundary, as specified."""
    if variant not in VARIANTS:
        raise ValueError("unknown outcome sensitivity variant")
    frame = _normalized(normalized)
    if variant == "observed_only":
        keep = frame.cgm_observed & ~frame.cgm_interpolated & ~frame.cgm_lower_censored
        frame = frame.loc[keep].reset_index(drop=True)
    if frame.empty:
        return []
    low = frame.cgm_mg_dl.to_numpy(float) < 54
    times = frame.timestamp
    starts, ends, run_start = [], [], None
    for position, value in enumerate(low):
        gap = times.iloc[position] - times.iloc[position - 1] if position else None
        # The boundary-only variant retains detect_events' other gap checks.
        continuous = position > 0 and (
            gap == pd.Timedelta(minutes=5)
            if variant == "observed_only"
            else gap <= pd.Timedelta(minutes=5)
        )
        if run_start is not None and (not value or not continuous):
            if position - run_start >= 3:
                starts.append(run_start)
                ends.append(position - 1)
            run_start = None
        if value and run_start is None:
            run_start = position
    if run_start is not None and len(frame) - run_start >= 3:
        starts.append(run_start)
        ends.append(len(frame) - 1)
    merged = []
    for start, end in zip(starts, ends, strict=True):
        if merged:
            old_start, old_end = merged[-1]
            gap = times.iloc[start] - times.iloc[old_end]
            complete_grid = (
                gap == (start - old_end) * pd.Timedelta(minutes=5)
                if variant == "observed_only"
                else int(gap / pd.Timedelta(minutes=5)) == start - old_end
            )
            boundary = (
                gap < pd.Timedelta(minutes=30)
                if (variant == "strict_merge_boundary")
                else gap <= pd.Timedelta(minutes=30)
            )
            if complete_grid and boundary:
                merged[-1] = (old_start, end)
                continue
        merged.append((start, end))
    source, participant = frame[_IDS].iloc[0].astype(str)
    return [
        Event(
            f"{source}:{participant}:severe_hypo_{variant}:{index:04d}",
            f"severe_hypo_{variant}",
            times.iloc[start],
            times.iloc[end],
        )
        for index, (start, end) in enumerate(merged)
    ]


def _primary_catalogue(frame: pd.DataFrame) -> list[Event]:
    return detect_events(
        frame,
        threshold=54,
        direction="below",
        minimum_length=3,
        merge_gap_minutes=30,
        kind="severe_hypo",
        sample_minutes=5,
    )


def _event_frame(events: list[Event], key: tuple[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "source_id": key[0],
                "participant_id": key[1],
                "event_id": event.event_id,
                "event_onset": event.onset,
                "event_end": event.end,
            }
            for event in events
        ],
        columns=[*_TRUTH, "event_end"],
    )
    for column in ("event_onset", "event_end"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def base_catalogue(
    events: list[Event], timeline: pd.DataFrame, horizon: int
) -> list[Event]:
    """Every variant uses ORIGINAL base-evaluable lead-ins, before any cut."""
    if horizon not in HORIZONS:
        raise ValueError("outcome sensitivity horizon must be 30 or 60")
    base = _boolean(timeline, f"base_evaluable_{horizon}")
    times = pd.to_datetime(timeline.loc[base, "context_end"], utc=True).array.asi8
    if np.any(np.diff(times) <= 0):
        raise ValueError("original base timeline must be strictly increasing")
    width = pd.Timedelta(minutes=horizon).value
    return [
        event
        for event in events
        if np.searchsorted(times, event.onset.value - width, side="left")
        < np.searchsorted(times, event.onset.value, side="left")
    ]


def _ratio(a, b):
    return float(a / b) if b else float("nan")


def rematch_emitted_alarms(
    timeline: pd.DataFrame,
    events: pd.DataFrame,
    exposure: pd.DataFrame,
    alarms: pd.DataFrame,
    *,
    horizon: int,
    glucose_cut: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Copy emissions exactly; replace only retrospective event-match columns."""
    if horizon not in HORIZONS or not np.isfinite(glucose_cut) or glucose_cut < 54:
        raise ValueError("invalid frozen horizon/glucose cut")
    key = tuple(timeline[_IDS].iloc[0].astype(str))
    for table in (timeline, exposure, events, alarms):
        _identity(table, key, empty=table is events or table is alarms)
    if len(exposure) != 1 or not set(_TRUTH).issubset(events):
        raise ValueError("one original base exposure and explicit event truth required")
    required = {*_KEYS, "patient_day", "cgm_last", "base_evaluable", *_MATCH}
    if not required.issubset(alarms):
        raise ValueError("original alarm ledger is missing required columns")
    frame = timeline.copy()
    frame["context_end"] = pd.to_datetime(frame.context_end, utc=True)
    if (
        frame.context_end.isna().any()
        or frame.context_end.duplicated().any()
        or (not frame.context_end.is_monotonic_increasing)
    ):
        raise ValueError(
            "original timeline keys are missing, duplicate or out of order"
        )
    frame["base_evaluable"] = _boolean(frame, f"base_evaluable_{horizon}")
    online = _online_eligible(frame, glucose_cut)
    for column in (
        "anchor_cgm_observed",
        "anchor_cgm_interpolated",
        "anchor_cgm_censored",
    ):
        _boolean(frame, column)
    emitted = alarms.copy(deep=True)
    emitted["context_end"] = pd.to_datetime(emitted.context_end, utc=True)
    _boolean(emitted, "base_evaluable")
    if (
        emitted.context_end.isna().any()
        or emitted.context_end.duplicated().any()
        or (not emitted.context_end.is_monotonic_increasing)
    ):
        raise ValueError("emitted alarm keys are missing, duplicate or out of order")
    anchor = frame.set_index("context_end")
    if not emitted.context_end.isin(anchor.index).all():
        raise ValueError("an original emission is missing from the original timeline")
    expected = anchor.loc[emitted.context_end].reset_index()
    _equal(
        emitted,
        expected,
        [*_KEYS, "patient_day", "cgm_last", "base_evaluable"],
        "emitted alarm anchor",
    )
    if not online[frame.context_end.isin(emitted.context_end)].all():
        raise ValueError("an original emission violates its declared glucose cut")
    truth = events.copy()
    truth["event_onset"] = pd.to_datetime(truth.event_onset, utc=True)
    if (
        truth.event_id.isna().any()
        or truth.event_onset.isna().any()
        or (truth.event_id.duplicated().any() or truth.event_onset.duplicated().any())
    ):
        raise ValueError("variant event identities/onsets are missing or duplicated")
    truth = truth.sort_values("event_onset").reset_index(drop=True)
    for column in _MATCH:
        emitted[f"primary_{column}"] = emitted[column]
    emitted["matched"] = False
    emitted["unmatched"] = emitted.base_evaluable
    emitted["matched_event_id"] = pd.Series(None, index=emitted.index, dtype="object")
    emitted["matched_event_onset"] = pd.NaT
    emitted["matched_event_onset"] = pd.to_datetime(
        emitted.matched_event_onset, utc=True
    )
    emitted["lead_minutes"] = np.nan
    matched = {}
    for index in emitted.index:
        if not emitted.at[index, "base_evaluable"]:
            continue
        timestamp = emitted.at[index, "context_end"]
        for event in truth.to_dict("records"):
            if event["event_id"] not in matched and (
                timestamp
                < event["event_onset"]
                <= timestamp + pd.Timedelta(minutes=horizon)
            ):
                lead = (event["event_onset"] - timestamp).total_seconds() / 60
                emitted.loc[index, ["matched", "unmatched"]] = [True, False]
                emitted.at[index, "matched_event_id"] = event["event_id"]
                emitted.at[index, "matched_event_onset"] = event["event_onset"]
                emitted.at[index, "lead_minutes"] = lead
                matched[event["event_id"]] = (timestamp, lead)
                break
    base = frame.loc[frame.base_evaluable]
    times = base.context_end.array.asi8
    allowed = online[frame.base_evaluable.to_numpy()]
    event_rows = []
    for event in truth.to_dict("records"):
        lower = np.searchsorted(
            times, event["event_onset"].value - (pd.Timedelta(minutes=horizon).value)
        )
        upper = np.searchsorted(times, event["event_onset"].value)
        if lower == upper:
            raise ValueError(
                "variant base event lacks an ORIGINAL base-evaluable lead-in"
            )
        timestamp, lead = matched.get(event["event_id"], (pd.NaT, float("nan")))
        detected = event["event_id"] in matched
        event_rows.append(
            {
                **event,
                "base_lead_in_decision_count": int(upper - lower),
                "eligible_lead_in_decision_count": int(allowed[lower:upper].sum()),
                "reachable": bool(allowed[lower:upper].any()),
                "detected": detected,
                "alarm_context_end": timestamp,
                "lead_minutes": lead,
                "late_detection": detected and lead < 5,
                "missed": not detected,
                **{
                    f"coverage_{lead_min}": detected and lead >= lead_min
                    for lead_min in _COVERAGE
                },
            }
        )
    event_columns = [
        *truth.columns,
        "base_lead_in_decision_count",
        "eligible_lead_in_decision_count",
        "reachable",
        "detected",
        "alarm_context_end",
        "lead_minutes",
        "late_detection",
        "missed",
        *(f"coverage_{lead}" for lead in _COVERAGE),
    ]
    ledger = pd.DataFrame(event_rows, columns=event_columns)
    for column in ("event_onset", "event_end", "alarm_context_end"):
        if column in ledger:
            ledger[column] = pd.to_datetime(ledger[column], utc=True)
    denominator, detected, delivered = (
        len(truth),
        len(matched),
        int(emitted.base_evaluable.sum()),
    )
    reachable = int(ledger.reachable.sum())
    item = exposure.iloc[0].to_dict()
    days, minutes = float(item["patient_days"]), float(item["observation_minutes"])
    if days <= 0 or not days.is_integer() or minutes <= 0 or minutes > days * 1440:
        raise ValueError("original base exposure is invalid")
    leads = tuple(value[1] for value in matched.values())
    participant = pd.DataFrame(
        [
            {
                **item,
                "base_patient_days": tuple(sorted(base.patient_day.unique())),
                "effective_24h_days": minutes / 1440,
                "base_event_count": denominator,
                "reachable_event_count": reachable,
                "detected_event_count": detected,
                "fixed_sensitivity": _ratio(detected, denominator),
                "conditional_sensitivity": _ratio(detected, reachable),
                "eligible_decision_count": int(online.sum()),
                "base_evaluable_decision_count": len(base),
                "eligible_evaluable_decision_count": int(allowed.sum()),
                "emitted_alarm_count": len(emitted),
                "delivered_alarm_count": delivered,
                "unevaluable_alarm_count": len(emitted) - delivered,
                "matched_alarm_count": detected,
                "unmatched_alarm_count": delivered - detected,
                "ppv": _ratio(detected, delivered),
                "matched_lead_minutes": leads,
                "unmatched_alarms_per_patient_day": (delivered - detected) / days,
                "delivered_alarms_per_patient_day": delivered / days,
                "late_detection_count": sum(lead < 5 for lead in leads),
                "missed_event_count": denominator - detected,
                **{
                    f"coverage_{lead}_event_count": int(
                        ledger[f"coverage_{lead}"].sum()
                    )
                    for lead in _COVERAGE
                },
            }
        ]
    )
    _equal(
        emitted,
        alarms,
        [column for column in alarms if column not in _MATCH],
        "unchanged emitted alarm sequence",
    )
    return emitted, ledger, participant


def _catalogue_difference(primary, variant, normalized, timeline, horizon):
    primary_base, variant_base = (
        base_catalogue(catalogue, timeline, horizon) for catalogue in (primary, variant)
    )
    old = {event.onset for event in primary_base}
    new = {event.onset for event in variant_base}
    confirmations = normalized.set_index("timestamp")
    counts = {}
    for name in ("interpolated", "nonobserved", "censored"):
        count = 0
        for event in primary_base:
            onset_position = confirmations.index.get_loc(event.onset)
            first = confirmations.iloc[onset_position : onset_position + 3]
            if len(first) != 3 or not first.cgm_mg_dl.lt(54).all():
                raise ValueError(
                    "primary onset does not have the original three-reading support"
                )
            values = {
                "interpolated": first.cgm_interpolated,
                "nonobserved": ~first.cgm_observed,
                "censored": first.cgm_lower_censored,
            }[name]
            count += int(values.any())
        counts[f"primary_onset_confirmations_with_{name}_reading"] = count
    overlaps = [
        sum(v.onset <= p.end and p.onset <= v.end for v in variant_base)
        for p in primary_base
    ]
    return {
        "primary_full_catalogue_count": len(primary),
        "variant_full_catalogue_count": len(variant),
        "primary_base_event_count": len(primary_base),
        "variant_base_event_count": len(variant_base),
        "variant_events_without_original_base_leadin": len(variant) - len(variant_base),
        "shared_base_onset_count": len(old & new),
        "primary_only_base_onset_count": len(old - new),
        "variant_only_base_onset_count": len(new - old),
        "primary_base_events_overlapped_by_variant": sum(
            value > 0 for value in overlaps
        ),
        "primary_base_events_split_by_variant": sum(value > 1 for value in overlaps),
        "variant_base_events_overlapping_primary": sum(
            any(v.onset <= p.end and p.onset <= v.end for p in primary_base)
            for v in variant_base
        ),
        **counts,
    }


def _binding_map(bindings):
    result = {}
    for binding in bindings:
        path = _bound_file(binding)
        if path in result:
            raise ValueError("duplicate accepted file binding")
        result[path] = binding
    return result


def _bound_path(path, bindings):
    resolved = Path(path).resolve()
    if resolved not in bindings:
        raise ValueError("missing explicit current sensitivity input binding")
    return _bound_file(bindings[resolved], resolved)


def _read_participant(paths, key, root):
    if not paths or any(
        not (path == root if root.is_file() else path.is_relative_to(root))
        for path in paths
    ):
        raise ValueError("normalized selection paths escape their accepted source root")
    dataset = ds.dataset(
        [str(path) for path in paths],
        format="parquet",
        partitioning="hive",
        partition_base_dir=str(root if root.is_dir() else root.parent),
    )
    frames, row_count = [], 0
    for fragment in dataset.get_fragments():
        ids = _fragment_frame(fragment, _IDS)
        _identity(ids, key)
        row_count += len(ids)
        if row_count > 1_000_000:
            raise ValueError("normalized participant row bound exceeded")
        frames.append(_fragment_frame(fragment))
    out = pd.concat(frames, ignore_index=True)
    validate_frame(out)
    return _normalized(out)


def _read_key(path, key):
    return (
        ds.dataset(str(path), format="parquet")
        .to_table(
            filter=(ds.field("source_id") == key[0])
            & (ds.field("participant_id") == key[1]),
            use_threads=False,
        )
        .to_pandas()
    )


def _selection_identities(path, selected, *, require_all=False):
    parquet = pq.ParquetFile(path)
    columns = [*_IDS, *(["partition"] if "partition" in parquet.schema.names else [])]
    actual = set()
    for batch in parquet.iter_batches(
        batch_size=65536, columns=columns, use_threads=False
    ):
        table = batch.to_pandas()
        keys = set(table[_IDS].astype(str).itertuples(index=False, name=None))
        if (
            table[_IDS].isna().any().any()
            or not keys.issubset(selected)
            or ("partition" in table and not table.partition.eq("selection").all())
        ):
            raise ValueError(
                "ledger contains missing/foreign selection identities or roles"
            )
        if require_all and (table.duplicated(_IDS).any() or keys & actual):
            raise ValueError("base exposure/participant identities are duplicated")
        actual |= keys
    if require_all and actual != selected:
        raise ValueError("ledger is missing original selection participants")


def _validate_protocol(config, sources):
    protocol = config["protocol"]
    if (
        protocol["sample_minutes"] != 5
        or protocol["minimum_consecutive_samples"] != 3
        or protocol["merge_event_gap_minutes"] != 30
        or protocol["thresholds_mg_dl"]["severe_hypoglycemia"] != 54
        or tuple(protocol["horizons_minutes"]) != HORIZONS
        or tuple(sorted(config["data"]["development_sources"])) != sources
        or set(sources)
        & (_FORBIDDEN | set(config["data"].get("forbidden_development_sources", [])))
    ):
        raise ValueError(
            "sensitivity config differs from the frozen development contract"
        )
    if not config["project"].get("synthetic", False):
        _validate_real_data_provenance(config)


def _aggregate_participants(table, sources):
    source_rows = [
        {"source_id": source, **_aggregate(table.loc[table.source_id.eq(source)])}
        for source in sources
    ]
    source_table = pd.DataFrame(source_rows)
    summary = _aggregate(table)
    undefined = source_table.loc[
        source_table.event_positive_participant_count.eq(0), "source_id"
    ].tolist()
    summary["participant_pooled_sensitivity"] = summary.pop("fixed_sensitivity")
    summary.update(
        {
            "source_count": len(sources),
            "undefined_sensitivity_sources": undefined,
            "equal_source_participant_macro_fixed_sensitivity": float("nan")
            if undefined
            else float(source_table.fixed_sensitivity.mean()),
            "equal_source_participant_macro_unmatched_alarms_per_patient_day": float(
                source_table.participant_macro_unmatched_alarms_per_patient_day.mean()
            ),
        }
    )
    return source_table, summary


def _catalogue_summary(differences: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in differences
        if column not in (*_IDS, "variant", "horizon_minutes")
    ]
    aggregate = differences.groupby(
        ["source_id", "variant", "horizon_minutes"],
        as_index=False,
        observed=True,
    )[columns].sum()
    pooled = differences.groupby(
        ["variant", "horizon_minutes"], as_index=False, observed=True
    )[columns].sum()
    pooled["source_id"] = "pooled"
    return pd.concat([aggregate, pooled], ignore_index=True)


def run_event_sensitivity(config: dict) -> Path:
    if config.get("revision_contract") != REVISION_CONTRACT or (
        config.get("partition") != "selection"
    ):
        raise ValueError("only explicit revision development selection is allowed")
    acceptance = config["input_acceptance"]
    if acceptance.get("status") != "accepted" or (
        acceptance.get("binding_time") != "current_sensitivity_acceptance"
    ):
        raise ValueError("current sensitivity input acceptance is required")
    if not acceptance.get("evidence"):
        raise ValueError("actual normalization/parity acceptance evidence is required")
    _binding_map(acceptance["evidence"])
    normalized_bindings = _binding_map(acceptance["normalized_files"])
    prepared_bindings = _binding_map(acceptance["prepared_files"])
    started = pd.Timestamp(config["preparation_started_at"])
    if (
        pd.isna(started)
        or started.tzinfo is None
        or any(
            binding["mtime_ns"] >= started.value
            for binding in normalized_bindings.values()
        )
    ):
        raise ValueError("normalized mtime must precede the actual preparation start")
    prepared = Path(config["prepared_dir"]).resolve()
    audit_path = _bound_path(prepared / "input-audit.json", prepared_bindings)
    manifest_path = _bound_path(
        prepared / "participant-manifest.private.json", prepared_bindings
    )
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "passed"
        or audit.get("revision_contract") != REVISION_CONTRACT
    ):
        raise ValueError("accepted primary replay preparation is required")
    if started.value > audit_path.stat().st_mtime_ns:
        raise ValueError("preparation start cannot follow its completed audit")
    sources = tuple(audit["source_order"])
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("source order is missing or duplicated")
    replay_config = load_config(_bound_file(config["replay_config"]))
    _validate_protocol(replay_config, sources)
    normalized_root = Path(audit["normalized_path"]).resolve()
    if not replay_config["project"].get("synthetic", False):
        configured = resolve_path(
            replay_config, replay_config["data"].get("input_path")
        )
        if configured is None or configured.resolve() != normalized_root:
            raise ValueError(
                "normalized root differs from the original replay configuration"
            )
    manifest = json.loads(manifest_path.read_text())["participants"]
    keys = [(str(e["source_id"]), str(e["participant_id"])) for e in manifest]
    if len(set(keys)) != len(keys):
        raise ValueError("prepared participant roles overlap or duplicate")
    entries = [entry for entry in manifest if entry["partition"] == "selection"]
    selected = {(str(e["source_id"]), str(e["participant_id"])) for e in entries}
    if not selected or {key[0] for key in selected} != set(sources):
        raise ValueError("selection participant/source population is incomplete")
    expected_normalized = {
        Path(p).resolve() for e in entries for p in e["normalized_paths"]
    }
    if set(normalized_bindings) != expected_normalized:
        raise ValueError("accepted normalized bindings differ from selection manifest")
    endpoints = {}
    expected_prepared = {audit_path, manifest_path}
    for entry in entries:
        path = _bound_path(entry["timeline_path"], prepared_bindings)
        if not path.is_relative_to(prepared):
            raise ValueError("prepared timeline escapes its accepted bundle")
        expected_prepared.add(path)
    for horizon in HORIZONS:
        for category in ("events", "exposure"):
            path = _bound_path(
                audit["endpoint_paths"][f"selection/{horizon}"][category],
                prepared_bindings,
            )
            if not path.is_relative_to(prepared):
                raise ValueError("prepared endpoint escapes its accepted bundle")
            endpoints[horizon, category] = path
            expected_prepared.add(path)
            _selection_identities(path, selected, require_all=category == "exposure")
    if set(prepared_bindings) != expected_prepared:
        raise ValueError("prepared bindings must cover exactly the selection inputs")
    runs = config["runs"]
    identities = [(r["arm"], r["model_seed"]) for r in runs]
    if not runs or len(set(identities)) != len(identities):
        raise ValueError("accepted run identities are empty or duplicated")
    for run in runs:
        if not run.get("accepted_evidence"):
            raise ValueError("actual primary run acceptance evidence is required")
        _binding_map(run["accepted_evidence"])
        identity_fields = {"model_summary", "composition_recipe"}.intersection(run)
        if len(identity_fields) != 1:
            raise ValueError(
                "one exact producing model summary or composition recipe is required"
            )
        field = next(iter(identity_fields))
        path = _bound_file(run[field])
        identity = json.loads(path.read_text())
        if not path.is_relative_to(Path(run["run_dir"]).resolve()):
            raise ValueError("producing identity binding escapes its accepted run")
        if identity.get("revision_contract") != REVISION_CONTRACT:
            raise ValueError("primary run is outside the corrected revision contract")
        if field == "model_summary":
            if path != Path(run["run_dir"]).resolve() / "model_summary.json" or (
                run["arm"] != identity.get("architecture", identity.get("baseline"))
            ):
                raise ValueError("model summary identity differs from accepted arm/run")
            if run["model_seed"] is not None and run["model_seed"] != identity.get(
                "training_seed"
            ):
                raise ValueError("model seed differs from its producing summary")
        elif (
            run["arm"] != "composition"
            or run["model_seed"] != identity.get("comparison_lstm_seed")
            or identity.get("algebra")
            != "arithmetic_mean_of_two_right_step_member_ECDF_ranks"
            or tuple(identity.get("source_order", ())) != sources
            or identity.get("reference_partition") != "calibration"
        ):
            raise ValueError("composition identity differs from the accepted recipe")
        seen = set()
        if not run.get("conditions"):
            raise ValueError("accepted primary run has no original alarm conditions")
        for condition in run["conditions"]:
            signature = tuple(
                condition[k] for k in ("horizon_minutes", "glucose_cut", "nominal_fapd")
            )
            if (
                signature in seen
                or signature[0] not in HORIZONS
                or (
                    not np.isfinite(signature[1])
                    or signature[1] < 54
                    or not np.isfinite(signature[2])
                    or signature[2] <= 0
                )
            ):
                raise ValueError("primary ledger condition is duplicated or invalid")
            seen.add(signature)
            for category in ("alarms", "events", "participants"):
                path = _bound_file(condition[category])
                if not path.is_relative_to(Path(run["ledger_root"]).resolve()):
                    raise ValueError("primary ledger escapes its accepted root")
                _selection_identities(
                    path, selected, require_all=category == "participants"
                )
        for cut, budget in {(cut, budget) for _, cut, budget in seen}:
            if {h for h, c, b in seen if (c, b) == (cut, budget)} != set(HORIZONS):
                raise ValueError(
                    "each accepted cut/budget requires both original horizons"
                )
    output = Path(config["output_dir"]).resolve()
    inputs = set(normalized_bindings) | set(prepared_bindings)
    if any(path.is_relative_to(output) for path in inputs):
        raise ValueError("sensitivity output overlaps accepted inputs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a sensitivity output")
    private, exportable = output / "private", output / "exportable"
    private.mkdir(parents=True, exist_ok=True)
    exportable.mkdir()
    differences, summaries = [], []
    try:
        for ordinal, entry in enumerate(entries):
            key = (str(entry["source_id"]), str(entry["participant_id"]))
            timeline = pq.ParquetFile(entry["timeline_path"]).read().to_pandas()
            _identity(timeline, key)
            if (
                len(timeline) != entry["stored_rows"]
                or not timeline.partition.eq("selection").all()
            ):
                raise ValueError(
                    "prepared timeline row count/role differs from manifest"
                )
            normalized = _read_participant(
                [Path(p).resolve() for p in entry["normalized_paths"]],
                key,
                normalized_root,
            )
            replay = build_participant_replay(
                normalized, replay_config, partition="selection"
            )
            _equal(
                replay.timeline,
                timeline,
                sorted(set(replay.timeline) | set(timeline)),
                "rebuilt complete primary timeline",
            )
            primary = _primary_catalogue(normalized)
            for horizon in HORIZONS:
                for category, actual in (
                    ("events", replay.events[horizon]),
                    ("exposure", replay.exposure[horizon]),
                ):
                    original = _read_key(endpoints[horizon, category], key)
                    if (
                        "partition" in original
                        and not original.partition.eq("selection").all()
                    ):
                        raise ValueError("prepared endpoint has a foreign partition")
                    _equal(
                        actual, original, list(actual), "rebuilt primary " + category
                    )
                _equal(
                    _event_frame(base_catalogue(primary, timeline, horizon), key),
                    replay.events[horizon],
                    _TRUTH,
                    "primary full catalogue lead-in",
                )
                for variant in VARIANTS:
                    catalogue = build_variant_catalogue(normalized, variant)
                    base_events = base_catalogue(catalogue, timeline, horizon)
                    events = _event_frame(base_events, key)
                    complete = _event_frame(catalogue, key)
                    complete["has_original_base_leadin"] = complete.event_id.isin(
                        {event.event_id for event in base_events}
                    )
                    full_folder = private / "full-catalogues" / variant / str(horizon)
                    full_folder.mkdir(parents=True, exist_ok=True)
                    _parquet(full_folder / f"part-{ordinal:05d}.parquet", complete)
                    destination = private / "catalogues" / variant / str(horizon)
                    destination.mkdir(parents=True, exist_ok=True)
                    _parquet(destination / f"part-{ordinal:05d}.parquet", events)
                    differences.append(
                        {
                            "source_id": key[0],
                            "participant_id": key[1],
                            "variant": variant,
                            "horizon_minutes": horizon,
                            "participant_count": 1,
                            "normalized_reading_count": len(normalized),
                            "normalized_interpolated_reading_count": int(
                                normalized.cgm_interpolated.sum()
                            ),
                            "normalized_lower_censored_reading_count": int(
                                normalized.cgm_lower_censored.sum()
                            ),
                            **_catalogue_difference(
                                primary, catalogue, normalized, timeline, horizon
                            ),
                        }
                    )
        pd.DataFrame(differences).to_parquet(
            private / "catalogue-differences.parquet", index=False
        )
        for run_index, run in enumerate(runs):
            for condition in run["conditions"]:
                horizon, cut, budget = (
                    condition[name]
                    for name in ("horizon_minutes", "glucose_cut", "nominal_fapd")
                )
                collected = {variant: [] for variant in VARIANTS}
                for ordinal, entry in enumerate(entries):
                    key = (str(entry["source_id"]), str(entry["participant_id"]))
                    timeline = pq.ParquetFile(entry["timeline_path"]).read().to_pandas()
                    exposure = _read_key(endpoints[horizon, "exposure"], key)
                    original_events = _read_key(endpoints[horizon, "events"], key)
                    emitted = _read_key(Path(condition["alarms"]["path"]), key)
                    original = _read_key(Path(condition["events"]["path"]), key)
                    _equal(
                        original, original_events, _TRUTH, "primary run event catalogue"
                    )
                    participants = _read_key(
                        Path(condition["participants"]["path"]), key
                    )
                    _equal(
                        participants,
                        exposure,
                        [*_IDS, "patient_days", "observation_minutes"],
                        "primary run exposure",
                    )
                    matched, _, original_metrics = rematch_emitted_alarms(
                        timeline,
                        original_events,
                        exposure,
                        emitted,
                        horizon=horizon,
                        glucose_cut=cut,
                    )
                    _equal(
                        matched, emitted, [*_KEYS, *_MATCH], "original primary matching"
                    )
                    _equal(
                        original_metrics,
                        participants,
                        list(original_metrics),
                        "primary run participant allocation",
                    )
                    for variant in VARIANTS:
                        events = (
                            pq.ParquetFile(
                                private
                                / "catalogues"
                                / variant
                                / str(horizon)
                                / f"part-{ordinal:05d}.parquet"
                            )
                            .read()
                            .to_pandas()
                        )
                        alarms, ledger, metrics = rematch_emitted_alarms(
                            timeline,
                            events,
                            exposure,
                            emitted,
                            horizon=horizon,
                            glucose_cut=cut,
                        )
                        root = (
                            private
                            / "evaluations"
                            / f"run-{run_index:03d}"
                            / variant
                            / str(cut)
                            / str(horizon)
                            / str(budget)
                        )
                        for category, table in (
                            ("alarms", alarms),
                            ("events", ledger),
                            ("participants", metrics),
                            ("exposure", exposure),
                        ):
                            folder = root / category
                            folder.mkdir(parents=True, exist_ok=True)
                            _parquet(folder / f"part-{ordinal:05d}.parquet", table)
                        collected[variant].append(metrics)
                for variant in VARIANTS:
                    source_metrics, summary = _aggregate_participants(
                        pd.concat(collected[variant], ignore_index=True), sources
                    )
                    descriptor = {
                        "arm": run["arm"],
                        "model_seed": run["model_seed"],
                        "variant": variant,
                        "horizon_minutes": horizon,
                        "glucose_cut": cut,
                        "nominal_fapd": budget,
                    }
                    for row in source_metrics.to_dict("records"):
                        summaries.append({**descriptor, "scope": "source", **row})
                    summaries.append(
                        {**descriptor, "scope": "equal_source_and_pooled", **summary}
                    )
        catalogue_rows = _catalogue_summary(pd.DataFrame(differences))
        catalogue_rows.to_csv(exportable / "catalogue-differences.csv", index=False)
        _json(
            exportable / "catalogue-differences.json",
            {"rows": catalogue_rows.to_dict("records")},
        )
        pd.DataFrame(
            [
                {
                    k: json.dumps(_plain(v))
                    if isinstance(v, (list, tuple, dict))
                    else v
                    for k, v in row.items()
                }
                for row in summaries
            ]
        ).to_csv(exportable / "event-sensitivity-summary.csv", index=False)
        _json(
            exportable / "event-sensitivity-summary.json",
            {
                "revision_contract": REVISION_CONTRACT,
                "partition": "selection",
                "interpretation": "descriptive_variant_catalogues_not_same_events",
                "raw_acquisition_provenance_established": False,
                "global_cohort_audit": (
                    "reused accepted source/preparation evidence; "
                    "not recomputed per participant"
                ),
                "alarms_scores_policies_refractory_changed": False,
                "base_exposure_changed": False,
                "rows": summaries,
            },
        )
        _recheck_bindings(config)
        _validate_protocol(replay_config, sources)
        _json(private / "accepted-inputs-used.json", config)
        _json(output / "status.json", {"status": "complete"})
    except Exception as error:
        _json(
            output / "status.json",
            {"status": "failed", "exception_type": type(error).__name__},
        )
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_event_sensitivity(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
