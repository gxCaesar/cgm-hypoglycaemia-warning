"""The endpoint scores an event no head is trained on, and that is pinned here.

This file uses the same device as `test_group_dro_rules.py`, which asserts that the
cumulative rule DOES collapse: an empirical finding is written down as a regression
test, so a change that alters it fails the suite instead of quietly invalidating the
documents that describe it.

The finding here is that `TASKS` holds four heads, none of them severe, while the
primary endpoint scores severe hypoglycaemia below 54 mg/dL. Nothing in this package
asked that question before 2026-08-21, and it went unnoticed for eleven days after
the severe endpoint arrived, while eight method attempts failed around it.
"""

from __future__ import annotations

import pytest

from glycoshift.sequences import REGRESSION_HORIZONS, TASKS
from glycoshift.target_alignment import (
    SCORED_TASKS,
    SEVERE_SCORE_SOURCE_TASK,
    audit_target_metric_alignment,
    severe_source_logit_column,
)


def test_no_head_is_trained_on_the_event_the_endpoint_scores() -> None:
    """The state of the world as of 2026-08-21, pinned so a change is deliberate.

    **If this test fails because a severe head was added, that is the intended
    outcome and the fix is to update this test**, together with the manuscript and
    the pre-registrations that describe the mismatch. It is not a defect to route
    around.
    """

    audit = audit_target_metric_alignment()
    assert audit["aligned"] is False
    assert audit["unaligned_scored_tasks"] == ["severe_hypo_30", "severe_hypo_60"]
    for task in SCORED_TASKS:
        entry = audit["tasks"][task]
        assert entry["trained_on_its_own_label"] is False
        # The score does come from a head that IS trained, which is what makes the
        # arms comparable even while none of them optimises the scored event.
        assert entry["source_task_is_trained"] is True


def test_the_binding_has_exactly_one_source_of_truth() -> None:
    """`_attach_severe_scores` must read the mapping, not restate it.

    A binding written down twice is a binding that can drift, which is the same
    reason `runtime_provenance` re-reads bytes instead of trusting a pin.
    """

    from glycoshift import deep_training

    for horizon in REGRESSION_HORIZONS:
        assert severe_source_logit_column(horizon) == (
            f"{SEVERE_SCORE_SOURCE_TASK[horizon]}_logit"
        )
    # The literal must not have come back into the function that consumes it.
    import inspect

    body = inspect.getsource(deep_training._attach_severe_scores)
    assert 'f"hypo_{horizon}_logit"' not in body
    assert "severe_source_logit_column" in body


def test_every_scored_task_has_a_source_that_exists() -> None:
    for horizon in REGRESSION_HORIZONS:
        assert SEVERE_SCORE_SOURCE_TASK[horizon] in TASKS
    with pytest.raises(ValueError, match="no severe score source"):
        severe_source_logit_column(15)
