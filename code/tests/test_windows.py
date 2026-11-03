from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glycoshift.config import load_config
from glycoshift.deep_training import _multitask_loss
from glycoshift.sequences import (
    TASKS,
    ParticipantSequenceStore,
    SequenceScaler,
    WindowSequenceDataset,
    sample_training_windows,
)
from glycoshift.synthetic import make_synthetic_cgm
from glycoshift.windows import build_windows


def test_day_coverage_and_patient_day_use_local_time() -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml"
    )
    config["data"]["participants_per_source"] = 1
    config["data"]["days_per_participant"] = 1
    config["protocol"]["context_hours"] = 0.5
    frame = make_synthetic_cgm(config)
    participant = frame.loc[
        (frame["source_id"] == "dev_alpha")
        & (frame["participant_id"] == "dev_alpha-p000")
    ].copy()
    participant["timestamp"] = (
        pd.to_datetime(participant["timestamp"], utc=True)
        - pd.Timedelta(hours=8)
    )
    windows = build_windows(participant, config)
    assert set(pd.to_datetime(windows["patient_day"])) == {
        pd.Timestamp("2026-01-01")
    }


def test_lower_censoring_is_label_only_and_masks_regression_target() -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml"
    )
    config["data"]["participants_per_source"] = 1
    config["data"]["days_per_participant"] = 1
    config["protocol"]["context_hours"] = 0.5
    frame = make_synthetic_cgm(config)
    participant = frame.loc[
        (frame["source_id"] == "dev_alpha")
        & (frame["participant_id"] == "dev_alpha-p000")
    ].copy()
    participant["cgm_mg_dl"] = 100.0
    participant["cgm_lower_censored"] = False
    censored_indices = [100, 101, 102]
    participant.loc[
        participant.index[censored_indices],
        "cgm_mg_dl",
    ] = 39.0
    participant.loc[
        participant.index[censored_indices],
        "cgm_lower_censored",
    ] = True
    participant = participant.reset_index(drop=True)

    windows = build_windows(participant, config)
    query_time = participant.loc[94, "timestamp"]
    query = windows.loc[windows["context_end"] == query_time].iloc[0]
    assert query["hypo_30_label"] == 1
    assert query["severe_hypo_30_label"] == 1
    assert not query["cgm_30_target_available"]
    assert pd.isna(query["cgm_30_target"])
    forbidden_query_times = set(
        participant.loc[100:107, "timestamp"]
    )
    assert not set(windows["context_end"]).intersection(
        forbidden_query_times
    )

    scaler = SequenceScaler.fit(
        participant,
        {("dev_alpha", "dev_alpha-p000")},
    )
    dataset = WindowSequenceDataset(
        participant,
        windows.loc[windows["context_end"] == query_time],
        scaler=scaler,
        context_samples=6,
    )
    item = dataset[0]
    assert item["regression_mask"].tolist() == [False, True]
    assert np.isfinite(item["regression_targets"].numpy()).all()
    assert scaler.values["cgm_mg_dl"].mean == 100.0

    batch = {
        key: value.unsqueeze(0)
        for key, value in item.items()
        if isinstance(value, torch.Tensor)
    }
    outputs = {
        "logits": {task: torch.zeros(1) for task in TASKS},
        "regression": {
            "30": torch.tensor([1_000_000.0]),
            "60": torch.tensor([0.0]),
        },
    }
    masked_loss = _multitask_loss(
        outputs,
        batch,
        positive_weights=torch.ones(len(TASKS)),
        auxiliary_weight=0.1,
    )
    outputs["regression"]["30"] = torch.tensor([0.0])
    reference_loss = _multitask_loss(
        outputs,
        batch,
        positive_weights=torch.ones(len(TASKS)),
        auxiliary_weight=0.1,
    )
    torch.testing.assert_close(masked_loss, reference_loss)


def test_shared_participant_store_preserves_sequence_items() -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml"
    )
    config["data"]["participants_per_source"] = 1
    config["data"]["days_per_participant"] = 1
    config["protocol"]["context_hours"] = 0.5
    frame = make_synthetic_cgm(config)
    participant = frame.loc[
        (frame["source_id"] == "dev_alpha")
        & (frame["participant_id"] == "dev_alpha-p000")
    ].copy()
    windows = build_windows(participant, config).head(3)
    scaler = SequenceScaler.fit(
        participant,
        {("dev_alpha", "dev_alpha-p000")},
    )
    full_store = ParticipantSequenceStore(
        participants={
            ("dev_alpha", "dev_alpha-p000"): participant.sort_values(
                "timestamp"
            ).reset_index(drop=True)
        }
    )
    compact_store = ParticipantSequenceStore.from_frame(participant)
    full_dataset = WindowSequenceDataset(
        participant,
        windows,
        scaler=scaler,
        context_samples=6,
        participant_store=full_store,
    )
    compact_dataset = WindowSequenceDataset(
        participant,
        windows,
        scaler=scaler,
        context_samples=6,
        participant_store=compact_store,
    )
    no_copy_windows = windows.reset_index(drop=True)
    no_copy_dataset = WindowSequenceDataset(
        participant,
        no_copy_windows,
        scaler=scaler,
        context_samples=6,
        participant_store=compact_store,
        copy_windows=False,
    )
    second_compact_dataset = WindowSequenceDataset(
        participant,
        windows.iloc[:1],
        scaler=scaler,
        context_samples=6,
        participant_store=compact_store,
    )

    for index in range(len(windows)):
        full_item = full_dataset[index]
        compact_item = compact_dataset[index]
        assert full_item.keys() == compact_item.keys()
        for key in full_item:
            if isinstance(full_item[key], torch.Tensor):
                torch.testing.assert_close(
                    compact_item[key],
                    full_item[key],
                    rtol=0.0,
                    atol=0.0,
                )
            else:
                assert compact_item[key] == full_item[key]

    stored = compact_store.participants[("dev_alpha", "dev_alpha-p000")]
    assert "source_id" not in stored
    assert "participant_id" not in stored
    assert compact_dataset._participants is second_compact_dataset._participants
    assert "sampling_weight" not in no_copy_windows
    assert "sampling_weight" in no_copy_dataset.windows
    assert np.shares_memory(
        no_copy_windows["participant_end_index"].to_numpy(),
        no_copy_dataset.windows["participant_end_index"].to_numpy(),
    )
    for key, value in compact_dataset[0].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(
                no_copy_dataset[0][key],
                value,
                rtol=0.0,
                atol=0.0,
            )
        else:
            assert no_copy_dataset[0][key] == value


def _legacy_sample_training_windows(
    windows: pd.DataFrame,
    *,
    negative_ratio: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output = []
    positive_mask = windows[
        [f"{task}_label" for task in TASKS]
    ].fillna(0).any(axis=1)
    for _, source_frame in windows.groupby(
        "source_id",
        sort=True,
        observed=False,
    ):
        positives = source_frame.loc[
            positive_mask.loc[source_frame.index]
        ].copy()
        negatives = source_frame.loc[
            ~positive_mask.loc[source_frame.index]
        ].copy()
        if positives.empty:
            keep_negative = min(len(negatives), 1000)
        else:
            keep_negative = min(
                len(negatives),
                int(np.ceil(len(positives) * negative_ratio)),
            )
        if keep_negative < len(negatives):
            chosen = rng.choice(
                negatives.index.to_numpy(),
                size=keep_negative,
                replace=False,
            )
            sampled_negatives = negatives.loc[np.sort(chosen)].copy()
        else:
            sampled_negatives = negatives
        negative_weight = (
            float(len(negatives) / len(sampled_negatives))
            if len(sampled_negatives)
            else 1.0
        )
        positives["sampling_weight"] = 1.0
        sampled_negatives["sampling_weight"] = negative_weight
        output.extend((positives, sampled_negatives))
    return pd.concat(output, ignore_index=False).sort_index().reset_index(drop=True)


def test_training_window_sampler_matches_legacy_rows_and_weights() -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml"
    )
    config["data"]["participants_per_source"] = 2
    config["data"]["days_per_participant"] = 1
    config["protocol"]["context_hours"] = 0.5
    windows = build_windows(make_synthetic_cgm(config), config)
    windows["source_id"] = windows["source_id"].astype("category")
    windows["source_id"] = windows["source_id"].cat.add_categories(
        ["unused-source"]
    )

    expected = _legacy_sample_training_windows(
        windows,
        negative_ratio=1.5,
        seed=17,
    )
    observed = sample_training_windows(
        windows,
        negative_ratio=1.5,
        seed=17,
    )

    pd.testing.assert_frame_equal(observed, expected)
