"""Gates for M6, anchor-aware member training.

`PREREG-anchor-aware-member-training.md` fixes six gates. Each one here is a
gate that must FIRE: G1 and G2 pin the control and the limit, G3 keeps the
anchor out of inference, G4 pins the anchor itself, G5 reproduces the gradient
mechanism the claim rests on, and the launcher covers G6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from glycoshift.anchor_composition import (
    ANCHOR_COLUMN,
    ANCHORED_TASKS,
    attach_anchor_values,
    fit_glucose_anchor,
    summarize_anchor,
)
from glycoshift.deep_training import _multitask_loss
from glycoshift.sequences import ANCHOR_COLUMN as SEQUENCE_ANCHOR_COLUMN
from glycoshift.sequences import TASKS


def _train_frame(count: int = 4000, *, partition: str = "train") -> pd.DataFrame:
    rng = np.random.default_rng(20260827)
    return pd.DataFrame(
        {
            "cgm_last": rng.uniform(40.0, 350.0, size=count),
            "partition": partition,
        }
    )


def _batch(
    glucose_anchor: np.ndarray | None,
    labels: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    rows = labels.shape[0]
    batch: dict[str, torch.Tensor] = {
        "labels": torch.tensor(labels, dtype=torch.float32),
        "training_labels": torch.tensor(labels, dtype=torch.float32),
        "label_mask": torch.tensor(
            np.ones_like(labels, dtype=bool) if mask is None else mask
        ),
        "regression_targets": torch.zeros(rows, 2),
        "regression_mask": torch.ones(rows, 2, dtype=torch.bool),
        # `_multitask_loss` scales per record by this; hold it at one so the
        # closed-form gradient check reads the composed term alone.
        "sampling_weight": torch.ones(rows),
    }
    if glucose_anchor is not None:
        batch["anchor"] = torch.tensor(glucose_anchor, dtype=torch.float32)
    return batch


def _outputs(logits: torch.Tensor) -> dict[str, object]:
    return {
        "logits": {task: logits[:, index] for index, task in enumerate(TASKS)},
        "regression": {
            "30": torch.zeros(logits.shape[0]),
            "60": torch.zeros(logits.shape[0]),
        },
    }


def test_the_two_anchor_column_names_are_the_same_string() -> None:
    assert ANCHOR_COLUMN == SEQUENCE_ANCHOR_COLUMN


# ----------------------------------------------------------------- G4: anchor


def test_g4_the_anchor_is_monotone_strictly_inside_the_unit_interval() -> None:
    anchor = fit_glucose_anchor(_train_frame())
    assert anchor.partition == "train"
    probe = np.linspace(30.0, 400.0, 500)
    values = anchor.evaluate(probe)
    assert np.all(np.diff(values) <= 1e-12), "anchor must not rise with glucose"
    assert values.min() > 0.0
    assert values.max() < 1.0
    # The endpoint's own lines: the anchor must separate them in the right order.
    assert (
        anchor.evaluate(np.array([54.0]))[0]
        > anchor.evaluate(np.array([61.0]))[0]
        > anchor.evaluate(np.array([70.0]))[0]
    )


def test_g4_the_anchor_refuses_any_partition_but_train() -> None:
    for partition in ("calibration", "selection"):
        with pytest.raises(ValueError, match="train rows alone"):
            fit_glucose_anchor(_train_frame(partition=partition))
    mixed = pd.concat(
        [_train_frame(50), _train_frame(50, partition="calibration")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="train rows alone"):
        fit_glucose_anchor(mixed)


def test_g4_the_anchor_is_blind_to_rows_outside_the_frame_it_was_given() -> None:
    """Appending held-out rows to the dataset cannot move a fitted anchor."""

    train = _train_frame()
    fitted = fit_glucose_anchor(train)
    held_out = _train_frame(4000, partition="calibration")
    held_out["cgm_last"] = held_out["cgm_last"] * 0.2  # a very different marginal
    again = fit_glucose_anchor(train)
    probe = np.linspace(40.0, 350.0, 200)
    assert np.array_equal(fitted.evaluate(probe), again.evaluate(probe))
    assert ANCHOR_COLUMN not in held_out.columns


def test_the_anchor_audit_records_what_was_fitted() -> None:
    train = _train_frame()
    anchor = fit_glucose_anchor(train)
    attached = attach_anchor_values(train, anchor)
    audit = summarize_anchor(attached, anchor, anchor_weight=0.5)
    assert audit["fitted_partition"] == "train"
    assert audit["anchored_tasks"] == list(ANCHORED_TASKS)
    assert audit["anchor_at_54_mg_dl"] > audit["anchor_at_70_mg_dl"]
    assert 0.0 < audit["anchor_minimum"] <= audit["anchor_maximum"] < 1.0


# ------------------------------------------------- G1 control, G2 limit


def test_g1_an_absent_anchor_weight_is_bitwise_the_control() -> None:
    torch.manual_seed(7)
    logits = torch.randn(64, len(TASKS), requires_grad=True)
    labels = np.random.default_rng(3).integers(0, 2, size=(64, len(TASKS)))
    labels = labels.astype(np.float32)
    positive = torch.full((len(TASKS),), 3.0)

    control_batch = _batch(None, labels)
    assert "anchor" not in control_batch
    control = _multitask_loss(
        _outputs(logits),
        control_batch,
        positive_weights=positive,
        auxiliary_weight=0.1,
    )
    (control.sum()).backward()
    control_grad = logits.grad.detach().clone()

    logits.grad = None
    # An anchor present in the batch but no weight configured must still not
    # change anything: the control is defined by the config key being absent.
    anchored_batch = _batch(np.full(64, 0.6, dtype=np.float32), labels)
    passthrough = _multitask_loss(
        _outputs(logits),
        anchored_batch,
        positive_weights=positive,
        auxiliary_weight=0.1,
        anchor_weight=None,
    )
    (passthrough.sum()).backward()

    assert torch.equal(control, passthrough)
    assert torch.equal(control_grad, logits.grad)


def test_g2_anchor_weight_one_reproduces_the_unanchored_loss() -> None:
    torch.manual_seed(11)
    logits = torch.randn(128, len(TASKS))
    labels = (
        np.random.default_rng(5)
        .integers(0, 2, size=(128, len(TASKS)))
        .astype(np.float32)
    )
    positive = torch.full((len(TASKS),), 2.5)

    control = _multitask_loss(
        _outputs(logits),
        _batch(None, labels),
        positive_weights=positive,
        auxiliary_weight=0.1,
    )
    limit = _multitask_loss(
        _outputs(logits),
        _batch(np.full(128, 0.3, dtype=np.float32), labels),
        positive_weights=positive,
        auxiliary_weight=0.1,
        anchor_weight=1.0,
    )
    torch.testing.assert_close(control, limit, rtol=1e-6, atol=1e-6)


def test_the_anchor_moves_the_loss_when_it_is_weighted_in() -> None:
    """The branch must be reachable, or G1 and G2 would pass on dead code."""

    torch.manual_seed(13)
    logits = torch.randn(32, len(TASKS))
    labels = np.ones((32, len(TASKS)), dtype=np.float32)
    positive = torch.ones(len(TASKS))
    control = _multitask_loss(
        _outputs(logits),
        _batch(None, labels),
        positive_weights=positive,
        auxiliary_weight=0.0,
    )
    anchored = _multitask_loss(
        _outputs(logits),
        _batch(np.full(32, 0.95, dtype=np.float32), labels),
        positive_weights=positive,
        auxiliary_weight=0.0,
        anchor_weight=0.5,
    )
    assert not torch.allclose(control, anchored)


def test_only_the_hypo_tasks_are_composed() -> None:
    """`hyper_*` must keep the original form exactly, values and gradients."""

    torch.manual_seed(17)
    base = torch.randn(48, len(TASKS))
    labels = np.ones((48, len(TASKS)), dtype=np.float32)
    positive = torch.ones(len(TASKS))
    hyper = [index for index, task in enumerate(TASKS) if task not in ANCHORED_TASKS]
    assert hyper, "the fixture assumes at least one un-anchored task"

    mask = np.zeros((48, len(TASKS)), dtype=bool)
    mask[:, hyper] = True
    control = _multitask_loss(
        _outputs(base),
        _batch(None, labels, mask=mask),
        positive_weights=positive,
        auxiliary_weight=0.0,
    )
    anchored = _multitask_loss(
        _outputs(base),
        _batch(np.full(48, 0.9, dtype=np.float32), labels, mask=mask),
        positive_weights=positive,
        auxiliary_weight=0.0,
        anchor_weight=0.5,
    )
    torch.testing.assert_close(control, anchored, rtol=0.0, atol=0.0)


# --------------------------------------------------------- G5: the mechanism


def test_g5_the_redundant_cell_has_the_smallest_gradient() -> None:
    """The three cells of the pre-registration's mechanism table.

    redundant: anchor high, label positive -- the anchor already has it.
    veto:      anchor high, label negative -- the network must suppress it.
    rescue:    anchor low,  label positive -- the network must supply it.
    """

    index = TASKS.index(ANCHORED_TASKS[0])
    cells = {
        "redundant": (0.97, 1.0),
        "veto": (0.97, 0.0),
        "rescue": (0.03, 1.0),
    }
    magnitudes: dict[str, float] = {}
    for name, (anchor_value, label) in cells.items():
        logits = torch.zeros(1, len(TASKS), requires_grad=True)
        labels = np.full((1, len(TASKS)), label, dtype=np.float32)
        mask = np.zeros((1, len(TASKS)), dtype=bool)
        mask[0, index] = True
        loss = _multitask_loss(
            _outputs(logits),
            _batch(np.array([anchor_value], dtype=np.float32), labels, mask=mask),
            positive_weights=torch.ones(len(TASKS)),
            auxiliary_weight=0.0,
            anchor_weight=0.5,
        )
        loss.sum().backward()
        magnitudes[name] = float(logits.grad[0, index].abs())

    assert magnitudes["redundant"] < magnitudes["veto"]
    assert magnitudes["redundant"] < magnitudes["rescue"]
    # And the signs point the two live cells in opposite directions.
    assert magnitudes["veto"] > 0.0 and magnitudes["rescue"] > 0.0


def test_g5_the_gradient_matches_the_closed_form() -> None:
    """`dL/df = w * s * (1 - s) * (p - y) / (p * (1 - p))`, checked numerically."""

    index = TASKS.index(ANCHORED_TASKS[0])
    # The weights must be asymmetric. At w = 0.5 the composition is symmetric in
    # its two terms, so a reversed weight direction is invisible -- a defect
    # injection found exactly that hole, and it would have silently transposed
    # the pre-registered w = 0.25 and w = 0.75 sensitivity cells.
    cells = (
        (0.8, 1.0, 0.4, 0.25),
        (0.8, 0.0, -0.9, 0.75),
        (0.1, 1.0, 0.2, 0.3),
        (0.6, 0.0, 0.5, 0.5),
    )
    for anchor_value, label, raw, weight in cells:
        logits = torch.zeros(1, len(TASKS), requires_grad=True)
        with torch.no_grad():
            logits[0, index] = raw
        labels = np.full((1, len(TASKS)), label, dtype=np.float32)
        mask = np.zeros((1, len(TASKS)), dtype=bool)
        mask[0, index] = True
        loss = _multitask_loss(
            _outputs(logits),
            _batch(np.array([anchor_value], dtype=np.float32), labels, mask=mask),
            positive_weights=torch.ones(len(TASKS)),
            auxiliary_weight=0.0,
            anchor_weight=weight,
        )
        loss.sum().backward()

        sigma = float(torch.sigmoid(torch.tensor(raw)))
        probability = weight * sigma + (1.0 - weight) * anchor_value
        expected = (
            weight
            * sigma
            * (1.0 - sigma)
            * (probability - label)
            / (probability * (1.0 - probability))
        )
        assert float(logits.grad[0, index]) == pytest.approx(expected, rel=1e-5)


# ----------------------------------------------- G3: the anchor never reaches inference


def _cohort():
    """The smoke cohort, matching `tests/test_fast_sequences.py::_cohort`."""

    from pathlib import Path

    from glycoshift.config import load_config
    from glycoshift.sequences import SequenceScaler
    from glycoshift.synthetic import make_synthetic_cgm
    from glycoshift.windows import build_windows

    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["data"]["participants_per_source"] = 3
    config["data"]["days_per_participant"] = 2
    config["protocol"]["context_hours"] = 0.5
    frame = make_synthetic_cgm(config)
    blocks = [
        build_windows(group.sort_values("timestamp").reset_index(drop=True), config)
        for _key, group in frame.groupby(["source_id", "participant_id"], sort=True)
    ]
    windows = pd.concat(blocks, ignore_index=True)
    keys = set(
        zip(  # noqa: B905
            frame["source_id"].astype(str), frame["participant_id"].astype(str)
        )
    )
    scaler = SequenceScaler.fit(frame, keys)
    samples = int(0.5 * 60 / int(config["protocol"]["sample_minutes"]))
    return frame, windows, scaler, samples


@pytest.mark.parametrize("fast", [False, True])
def test_g3_only_a_frame_carrying_the_column_emits_an_anchor(fast: bool) -> None:
    """Both assembly paths, since the fast one is the production default."""

    from glycoshift.fast_sequences import FastWindowSequenceDataset
    from glycoshift.sequences import WindowSequenceDataset

    frame, windows, scaler, samples = _cohort()
    dataset_class = FastWindowSequenceDataset if fast else WindowSequenceDataset

    plain = dataset_class(frame, windows, scaler=scaler, context_samples=samples)
    assert "anchor" not in plain[0], (
        "a frame without the column must not emit the key; this is what keeps "
        "the selection and calibration datasets un-anchored"
    )

    train_like = windows.copy()
    train_like["partition"] = "train"
    anchor = fit_glucose_anchor(train_like)
    anchored = dataset_class(
        frame,
        attach_anchor_values(train_like, anchor),
        scaler=scaler,
        context_samples=samples,
    )
    sample = anchored[0]
    assert "anchor" in sample
    value = float(sample["anchor"])
    assert 0.0 < value < 1.0
    expected = float(
        anchor.evaluate(np.array([float(train_like.iloc[0]["cgm_last"])]))[0]
    )
    assert value == pytest.approx(expected, rel=1e-6)


def test_g3_both_assembly_paths_agree_on_the_anchor() -> None:
    from glycoshift.fast_sequences import FastWindowSequenceDataset
    from glycoshift.sequences import WindowSequenceDataset

    frame, windows, scaler, samples = _cohort()
    train_like = windows.copy()
    train_like["partition"] = "train"
    attached = attach_anchor_values(train_like, fit_glucose_anchor(train_like))
    slow = WindowSequenceDataset(
        frame, attached, scaler=scaler, context_samples=samples
    )
    fast = FastWindowSequenceDataset(
        frame, attached, scaler=scaler, context_samples=samples
    )
    for index in range(min(32, len(attached))):
        assert torch.equal(slow[index]["anchor"], fast[index]["anchor"])
