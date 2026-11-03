"""The two GroupDRO weighting rules, and the collapse that motivated the fix.

The cumulative rule is Sagawa et al. 2020 and stays the default so the frozen
arm remains reproducible. These tests pin what it does and what the memoryless
alternative does instead, because the difference is a scientific claim in the
manuscript rather than an implementation detail.

The collapse test is a regression test for an empirical finding: on eleven CGM
sources with intrinsically unequal difficulty, the cumulative rule drove all
weight onto one source at every epoch of every seed, and that arm scored below
plain ERM. If a future change makes the cumulative rule stop collapsing under
these conditions, the manuscript's diagnosis no longer matches the code and the
test should fail loudly.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from glycoshift.deep_training import GroupDRO

GROUPS = 11
DEVICE = torch.device("cpu")


def _step(dro: GroupDRO, group_losses: list[float]) -> torch.Tensor:
    """One aggregation step where every group is present exactly once."""
    losses = torch.tensor(group_losses, dtype=torch.float32)
    groups = torch.arange(len(group_losses))
    dro.aggregate(losses, groups)
    return dro.weights.clone()


def _persistent_difficulty() -> list[float]:
    """Per-group losses spanning the range measured on the real sources.

    The observed spread was roughly 0.28 to 1.19 with one source persistently
    worst; that persistence is the condition under which the cumulative rule
    misbehaves.
    """
    return [0.28, 0.44, 0.53, 0.58, 0.61, 0.64, 0.71, 1.19, 0.76, 0.32, 0.75]


def test_cumulative_rule_collapses_under_persistent_difficulty() -> None:
    dro = GroupDRO.create(
        group_count=GROUPS, eta=0.01, device=DEVICE, rule="cumulative"
    )
    losses = _persistent_difficulty()
    worst = int(np.argmax(losses))
    for _ in range(4000):
        weights = _step(dro, losses)
    assert weights.argmax().item() == worst
    # Not merely "leading" - dominating, which is what makes the arm
    # single-source in practice.
    assert weights.max().item() > 0.99
    others = weights.sum().item() - weights.max().item()
    assert others < 0.01


def test_softmax_rule_does_not_collapse() -> None:
    """The fix: memoryless weights cannot accumulate, so no source takes over."""
    dro = GroupDRO.create(
        group_count=GROUPS,
        eta=0.0,
        device=DEVICE,
        rule="softmax",
        temperature=0.5,
    )
    losses = _persistent_difficulty()
    for _ in range(4000):
        weights = _step(dro, losses)
    assert weights.max().item() < 0.5, "softmax rule collapsed"
    # Every source keeps a real share of the objective.
    assert weights.min().item() > 1e-3
    assert weights.argmax().item() == int(np.argmax(losses))


@pytest.mark.parametrize("steps", [1, 10, 500])
def test_softmax_rule_is_memoryless(steps: int) -> None:
    """Weights after N steps equal weights after one step, for fixed losses.

    This is the property that rules out collapse by construction, so it is
    asserted directly rather than inferred from the previous test.
    """
    losses = _persistent_difficulty()
    once = GroupDRO.create(
        group_count=GROUPS, eta=0.0, device=DEVICE,
        rule="softmax", temperature=0.7,
    )
    many = GroupDRO.create(
        group_count=GROUPS, eta=0.0, device=DEVICE,
        rule="softmax", temperature=0.7,
    )
    first = _step(once, losses)
    for _ in range(steps):
        later = _step(many, losses)
    assert torch.equal(first, later)


def test_softmax_high_temperature_approaches_erm() -> None:
    """tau -> infinity recovers uniform weights, i.e. plain ERM."""
    dro = GroupDRO.create(
        group_count=GROUPS, eta=0.0, device=DEVICE,
        rule="softmax", temperature=1e6,
    )
    weights = _step(dro, _persistent_difficulty())
    assert torch.allclose(
        weights, torch.full((GROUPS,), 1.0 / GROUPS), atol=1e-5
    )


def test_softmax_low_temperature_approaches_worst_group() -> None:
    """tau -> 0 puts all mass on the worst group."""
    losses = _persistent_difficulty()
    dro = GroupDRO.create(
        group_count=GROUPS, eta=0.0, device=DEVICE,
        rule="softmax", temperature=1e-3,
    )
    weights = _step(dro, losses)
    assert weights.argmax().item() == int(np.argmax(losses))
    assert weights.max().item() > 0.999


def test_cumulative_eta_zero_is_group_balanced_erm() -> None:
    """The ERM control arm's basis: eta=0 holds the weights uniform and does not drift.

    THIS ASSERTED `torch.equal` UNTIL 2026-09-06 AND THAT WAS TOO STRONG, in a way that
    is worth writing down because the too-strong version passed for months.

    At eta = 0 the update is `w *= exp(0 * L)`, and `exp(0)` is exactly 1.0 for any
    finite loss, so the multiply is lossless and the losses are irrelevant. The one and
    only source of error is the renormalising `w /= w.sum()`. In float32 `1/11` is
    `0x3dba2e8c`, and eleven of them add to:

        torch 2.7.1      0x3f800000   exactly 1.0
        torch 2.5.1      0x3f800001   one ULP above 1.0
        Python's sum()   0x3f800000

    So on the stack this study's runs were produced under, the divide shifts every
    weight down by one ULP, `0x3dba2e8c` to `0x3dba2e8b`. Bitwise uniformity was never a
    property of the ALGORITHM. It was a property of one torch release's reduction
    happening to land on 1.0, and the test was reading that as science.

    What IS a property of the algorithm, and what this asserts instead:

      1. the deviation stays within a few units in the last place of 1/G, and
      2. it does not GROW, which is the whole distinction between rounding and drift.

    The second is the one that matters. This rule's real failure mode is accumulation:
    at eta > 0 the same multiply-and-renormalise drives all mass onto one source, which
    `test_cumulative_rule_collapses_under_persistent_difficulty` pins. A test that only
    bounded the deviation would pass on a slow drift toward collapse. Comparing an early
    window against a late one will not.

    Measured on torch 2.5.1: 1 ULP at steps 1, 2, 50 and 500, and 2 ULP at step 5. It
    oscillates inside a bounded fixed point and never grows.

    The numerics are deliberately NOT fixed. Summing in float64 would make the divide
    exact and version-independent, and it would also mean the released code no longer
    reproduces the arm this paper reports. The release pins torch 2.5.1 instead, which
    makes that pin load-bearing rather than tidy.
    """
    dro = GroupDRO.create(
        group_count=GROUPS, eta=0.0, device=DEVICE, rule="cumulative"
    )
    uniform = torch.full((GROUPS,), 1.0 / GROUPS)
    # One ULP of 1/G in float32. Computed rather than typed, so the bound follows the
    # dtype instead of a constant somebody would have to remember to change.
    ulp = float(torch.nextafter(uniform[0], torch.tensor(1.0)) - uniform[0])
    deviations = []
    for _ in range(500):
        weights = _step(dro, _persistent_difficulty())
        deviations.append(float((weights.double() - uniform.double()).abs().max()))

    worst = max(deviations)
    assert worst <= 4 * ulp, (
        f"weights drifted {worst:.3e} from uniform, over {4 * ulp:.3e}, which is four "
        f"units in the last place of 1/{GROUPS}"
    )
    early, late = max(deviations[:10]), max(deviations[-10:])
    assert late <= early, (
        f"the deviation grew from {early:.3e} in the first ten steps to {late:.3e} in "
        f"the last ten, so this is accumulation and not rounding"
    )


def test_the_uniformity_check_can_tell_rounding_from_the_slowest_drift() -> None:
    """The mirror of the test above, so its bound is known to discriminate.

    A bound is worth what it rejects. The previous `torch.equal` rejected everything,
    including one unit in the last place of ordinary float32 rounding, which is why it
    failed on the very stack the results were produced on. A bound loose enough to
    accept that could in principle accept a slow drift toward the collapse this rule is
    known for, and a test nobody has tried to defeat does not tell you which it is.

    So this drives the same two assertions with the smallest positive step size worth
    trying. At eta = 1e-6 the deviation after ten steps is 5.7e-07, far too small to
    notice by eye and far too small for any sanity check written in round numbers, and
    it reaches 2.8e-05 by step 500. Both assertions catch it independently: it is three
    orders of magnitude over the four-ULP bound, and it grows fiftyfold between the
    first ten steps and the last ten.
    """
    uniform = torch.full((GROUPS,), 1.0 / GROUPS)
    ulp = float(torch.nextafter(uniform[0], torch.tensor(1.0)) - uniform[0])
    dro = GroupDRO.create(
        group_count=GROUPS, eta=1e-6, device=DEVICE, rule="cumulative"
    )
    deviations = []
    for _ in range(500):
        weights = _step(dro, _persistent_difficulty())
        deviations.append(float((weights.double() - uniform.double()).abs().max()))

    assert max(deviations) > 4 * ulp, (
        "the four-ULP bound accepted a drift it must reject, so it no longer "
        "distinguishes rounding from accumulation"
    )
    assert max(deviations[-10:]) > max(deviations[:10]), (
        "the growth check did not see a drift that grows fiftyfold, so it is not "
        "measuring accumulation"
    )


def test_default_rule_is_the_original_one() -> None:
    """Existing configs must be unaffected by the new option."""
    dro = GroupDRO.create(group_count=GROUPS, eta=0.01, device=DEVICE)
    assert dro.rule == "cumulative"


def test_unsupported_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported group_dro_rule"):
        GroupDRO.create(
            group_count=GROUPS, eta=0.0, device=DEVICE, rule="chi_squared"
        )
    with pytest.raises(ValueError, match="positive temperature"):
        GroupDRO.create(
            group_count=GROUPS, eta=0.0, device=DEVICE,
            rule="softmax", temperature=0.0,
        )


def test_aggregated_loss_matches_the_weights_used() -> None:
    """The returned scalar must be the weighted mean the weights describe."""
    losses = _persistent_difficulty()
    for rule, kwargs in (
        ("cumulative", {"eta": 0.01}),
        ("softmax", {"eta": 0.0, "temperature": 0.5}),
    ):
        dro = GroupDRO.create(
            group_count=GROUPS, device=DEVICE, rule=rule, **kwargs
        )
        tensor = torch.tensor(losses, dtype=torch.float32)
        groups = torch.arange(GROUPS)
        aggregated, observed = dro.aggregate(tensor, groups)
        expected = float((dro.weights.cpu() * tensor).sum())
        assert aggregated.item() == pytest.approx(expected, abs=1e-6)
        assert observed[int(np.argmax(losses))] == pytest.approx(max(losses))
