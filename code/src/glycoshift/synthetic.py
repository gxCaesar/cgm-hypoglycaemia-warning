from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .schema import ACTION_FIELDS

# Every synthetic source must contain at least one severe-hypoglycaemia event, because
# the operational endpoint refuses a source with no event-positive participant. 45 sits
# clear of the 54 mg/dL severe threshold and above the 38 mg/dL clip.
SYNTHETIC_SEVERE_TROUGH_MG_DL = 45.0


def make_synthetic_cgm(config: dict, seed: int = 42) -> pd.DataFrame:
    data_config = config["data"]
    protocol = config["protocol"]
    rng = np.random.default_rng(seed)
    sample_minutes = int(protocol["sample_minutes"])
    samples_per_day = 24 * 60 // sample_minutes
    days = int(data_config["days_per_participant"])
    n_samples = samples_per_day * days
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    timestamps = pd.date_range(
        start=start,
        periods=n_samples,
        freq=f"{sample_minutes}min",
    )

    rows: list[pd.DataFrame] = []
    for source_index, source in enumerate(data_config["synthetic_sources"]):
        source_shift = (-6.0, 4.0, 10.0)[source_index % 3]
        action_available = {
            "basal_insulin_u": True,
            "bolus_insulin_u": True,
            "carbs_g": source != "dev_beta",
            "activity_value": source == "HUPA-UCM",
        }
        for participant_index in range(int(data_config["participants_per_source"])):
            phase = participant_index * 0.31
            time = np.arange(n_samples)
            circadian = 18 * np.sin(2 * math.pi * time / samples_per_day + phase)
            ultradian = 7 * np.sin(2 * math.pi * time / 48 + phase / 2)
            glucose = (
                115
                + source_shift
                + circadian
                + ultradian
                + rng.normal(0, 3.5, n_samples)
            )

            event_centers = [
                samples_per_day // 3 + participant_index % 7,
                samples_per_day * 2 // 3 + participant_index % 11,
                samples_per_day + samples_per_day // 3 + participant_index % 5,
                samples_per_day + samples_per_day * 2 // 3 + participant_index % 9,
            ]
            for event_number, center in enumerate(event_centers):
                width = 8
                window = np.arange(
                    max(0, center - width),
                    min(n_samples, center + width),
                )
                if event_number % 2 == 0:
                    # Drive the trough to a fixed severe value rather than subtracting a
                    # fixed amount. A fixed -62 leaves the trough above 54 mg/dL
                    # whenever the source shift and the circadian term are both
                    # high, so some
                    # sources got no severe event at all and the operational endpoint,
                    # which requires an event-positive participant in every source,
                    # refused. The refusal was correct. The fixture was not.
                    # Depth comes from the window's own peak, not from glucose[center].
                    # The window is clipped to the array and the centre is not, so a
                    # fixture short enough for a centre to fall past the end made the
                    # direct index raise, which thirty-three tests caught.
                    if window.size:
                        profile = np.exp(-((window - center) / 4.0) ** 2)
                        peak = float(glucose[window].max())
                        depth = max(peak - SYNTHETIC_SEVERE_TROUGH_MG_DL, 0.0)
                        glucose[window] -= depth * profile
                else:
                    glucose[window] += 105 * np.exp(-((window - center) / 5.0) ** 2)
            glucose = np.clip(glucose, 38, 320)

            participant = pd.DataFrame(
                {
                    "source_id": source,
                    "participant_id": f"{source}-p{participant_index:03d}",
                    "timestamp": timestamps,
                    "cgm_mg_dl": glucose,
                    "timestamp_local": timestamps.tz_localize(None),
                    "timestamp_utc": timestamps,
                    "timezone_known": True,
                    "cgm_observed": True,
                    "cgm_interpolated": False,
                }
            )

            for field in ACTION_FIELDS:
                available = action_available[field]
                observed_probability = {
                    "basal_insulin_u": 0.95,
                    "bolus_insulin_u": 0.08,
                    "carbs_g": 0.05,
                    "activity_value": 0.03,
                }[field]
                observed = (
                    rng.random(n_samples) < observed_probability
                    if available
                    else np.zeros(n_samples, dtype=bool)
                )
                values = np.full(n_samples, np.nan)
                if field == "basal_insulin_u":
                    values[observed] = rng.uniform(0.02, 0.08, observed.sum())
                elif field == "bolus_insulin_u":
                    values[observed] = rng.uniform(0.5, 5.0, observed.sum())
                elif field == "carbs_g":
                    values[observed] = rng.uniform(10, 80, observed.sum())
                else:
                    values[observed] = rng.uniform(0.1, 1.0, observed.sum())
                participant[field] = values
                participant[f"{field}_available"] = bool(available)
                participant[f"{field}_observed"] = observed
                participant[f"{field}_zero_ambiguous"] = False

            rows.append(participant)

    return pd.concat(rows, ignore_index=True)
