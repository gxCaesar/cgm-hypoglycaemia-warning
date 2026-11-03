"""Which head supplies each scored task's score, and whether it was trained on it.

**This module exists because nothing in this package asked the question.** About
seven thousand lines enforce provenance, source isolation, partition disjointness,
receipt scoping and GPU occupancy, and not one of them checks that the model is
optimised for the event the endpoint scores. It is not: `TASKS` holds four heads,
none of them severe, and the severe score is the 70 mg/dL `hypo_<h>` logit bound
verbatim. Git dates that as drift rather than design, the heads fixed 2026-07-25
and the severe endpoint arriving 2026-08-10, and it went unnoticed for eleven days
while eight method attempts failed around it.

The binding is recorded here **once** and read by both sites that perform it, so
this audit is derived from the code that runs rather than restating it. That is the
same discipline `runtime_provenance` applies to bytes: a fact asserted in a second
place is a fact that can drift.

**The audit does not forbid a mismatch.** The frozen protocol chose this binding and
every arm shares it, which is exactly what makes the cross-arm comparison fair. What
it does is make the mismatch *visible in the test suite* instead of discoverable only
by reading `deep_training.py:601`, so a future reader meets it as a stated property
rather than as a surprise.
"""

from __future__ import annotations

from typing import Any

from .sequences import REGRESSION_HORIZONS, TASKS

#: The head whose logit becomes each severe-event score. Both
#: `deep_training._attach_severe_scores` and the summary-baseline path in
#: `pipeline` read this rather than spelling out `hypo_<h>` themselves.
SEVERE_SCORE_SOURCE_TASK: dict[int, str] = {
    horizon: f"hypo_{horizon}" for horizon in REGRESSION_HORIZONS
}

#: The tasks the operational endpoint actually scores.
SCORED_TASKS: tuple[str, ...] = tuple(
    f"severe_hypo_{horizon}" for horizon in REGRESSION_HORIZONS
)


def audit_target_metric_alignment() -> dict[str, Any]:
    """Report, per scored task, whether any head is trained on its own label.

    ``aligned`` is true only when every scored task has a head of its own. It is
    false today, deliberately and by protocol, and the test that pins it is what
    turns a future change into a conversation rather than a silent drift.
    """

    tasks: dict[str, Any] = {}
    for horizon in REGRESSION_HORIZONS:
        scored = f"severe_hypo_{horizon}"
        source = SEVERE_SCORE_SOURCE_TASK[horizon]
        tasks[scored] = {
            "score_source_task": source,
            "source_task_is_trained": source in TASKS,
            "trained_on_its_own_label": scored in TASKS,
        }
    return {
        "trained_tasks": list(TASKS),
        "scored_tasks": list(SCORED_TASKS),
        "tasks": tasks,
        "aligned": all(
            entry["trained_on_its_own_label"] for entry in tasks.values()
        ),
        "unaligned_scored_tasks": sorted(
            task
            for task, entry in tasks.items()
            if not entry["trained_on_its_own_label"]
        ),
    }


def severe_source_logit_column(horizon: int) -> str:
    """Return the logit column the severe score at this horizon is bound to."""

    try:
        source = SEVERE_SCORE_SOURCE_TASK[int(horizon)]
    except KeyError as error:
        raise ValueError(f"no severe score source for horizon {horizon}") from error
    return f"{source}_logit"
