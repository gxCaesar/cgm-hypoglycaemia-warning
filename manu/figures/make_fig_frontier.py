'Make fig frontier.'

















from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "code" / "scripts" / "figures")
)
import caption_words as _caption_words  # noqa: E402
import glycofig as G  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

FROZEN = 0.5
FAPD_MAX = 2.0
# stem -> (family label, palette key, dash)
ARMS = {
    "erm_lstm_seed17": ("LSTM, ERM", "blue", "-"),
    "erm_tcn_seed17": ("TCN, ERM", "rose", "-"),
    "erm_patchtst_seed17": ("Patch transformer", "teal", "-"),
    "tls_lstm_seed17": ("Tail-loss shaping", None, "-"),
    "dro_tau_0p5_lstm_seed17": (r"GroupDRO $\tau$ 0.5", "olive", "-"),
    "dro_tau_1p0_lstm_seed17": (r"GroupDRO $\tau$ 1.0", "olive", "--"),
    "dro_tau_2p0_lstm_seed17": (r"GroupDRO $\tau$ 2.0", "olive", (0, (1.2, 1.2))),
    "erm_nomemory_glycoshift_seed17": ("Memory transf., seed 17", "amber", "-"),
    "erm_nomemory_glycoshift_seed23": ("Memory transf., seed 23", "amber", "--"),
}
SEED_DASH = ("-", "--", "-.", (0, (1.2, 1.2)))


def frontier(payload: dict, horizon: str) -> tuple[list[float], list[float]]:
    points = payload["horizons"][horizon]["selection_budget_frontier"]["frontier"]
    inside = [p for p in points if p["achieved_fapd"] <= FAPD_MAX]
    # The floor is 100 because the observed minimum across thirteen arms is 165, not
    # because 100 is a round number: a sweep runs to 36 alarms per participant-day and
    # only about six per cent of its thresholds land in the plotted range. A first
    # version guessed 500 and refused every arm. The guard is still worth having --
    # a curve drawn through a handful of retained points looks entirely normal.
    if len(inside) < 100:
        raise SystemExit(f"only {len(inside)} frontier points below {FAPD_MAX} FA/day")
    return ([p["achieved_fapd"] for p in inside],
            [p["best_feasible_event_sensitivity"] for p in inside])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operational-dir", required=True, type=Path)
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height-mm", type=float, default=76.0)
    parser.add_argument("--caption-words", type=int,
                        default=_caption_words.count("fig_frontier"))
    arguments = parser.parse_args(argv)

    arms = {
        stem: json.loads((arguments.operational_dir / f"{stem}.json").read_text())
        for stem in ARMS
    }
    seeds = {
        path.stem: json.loads(path.read_text())
        for path in sorted(arguments.seed_dir.glob("erm_tcn_seed*.json"))
    }
    if len(seeds) != 4:
        raise SystemExit(f"expected four seeds of one configuration, found {len(seeds)}")

    G.set_style()
    figure, axes = plt.subplots(
        1, 3, figsize=(G.COL2_MM * G.MM, arguments.height_mm * G.MM))
    figure.subplots_adjust(left=0.062, right=0.988, top=0.900, bottom=0.230,
                           wspace=0.30)

    for axis, horizon in zip(axes[:2], ("30", "60"), strict=True):
        for stem, (label, hue, dash) in ARMS.items():
            x, y = frontier(arms[stem], horizon)
            axis.plot(x, y, color=G.NOMINAL[hue] if hue else G.NEUTRAL,
                      linestyle=dash, linewidth=0.8, zorder=3,
                      label=label if horizon == "30" else None)
        axis.set_title(f"{horizon} minutes", loc="left", fontsize=6.0, pad=3)

    for stem, dash in zip(sorted(seeds), SEED_DASH, strict=True):
        x, y = frontier(seeds[stem], "30")
        axes[2].plot(x, y, color=G.NOMINAL["rose"], linestyle=dash, linewidth=0.8,
                     zorder=3, label=f"seed {stem.rsplit('seed', 1)[1]}")
    axes[2].set_title("One configuration, four seeds", loc="left", fontsize=6.0, pad=3)

    for axis in axes:
        axis.axvline(FROZEN, color=G.INK, lw=0.6, ls=(0, (2.4, 1.6)), zorder=2)
        axis.set_xlim(0.0, FAPD_MAX)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Achieved false alarms per participant-day")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Best feasible level 2 event sensitivity")
    axes[0].legend(loc="lower right", fontsize=5.0, handlelength=2.2,
                   borderaxespad=0.25, labelspacing=0.22)
    axes[2].legend(loc="lower right", fontsize=5.0, handlelength=2.2,
                   borderaxespad=0.25, labelspacing=0.22)

    for axis, letter in zip(axes, "abc", strict=True):
        G.panel_label(axis, letter, dx=-0.150, dy=1.130)

    G.save(figure, str(arguments.output), caption_words=arguments.caption_words)

    # The caption says curves cross inside the plotted range; check it rather than
    # asserting it. Compare every pair at the two ends of the usable interval.
    at = {}
    for stem in ARMS:
        x, y = frontier(arms[stem], "30")
        low = max(v for u, v in zip(x, y, strict=True) if u <= 0.25)
        high = max(v for u, v in zip(x, y, strict=True) if u <= FAPD_MAX)
        at[stem] = (low, high)
    crossings = sum(
        1
        for a in ARMS for b in ARMS
        if a < b and (at[a][0] - at[b][0]) * (at[a][1] - at[b][1]) < 0
    )
    if crossings == 0:
        raise SystemExit("the caption says several curves cross; none does")

    # And it says the seed spread is narrowest at the frozen budget. The first draft
    # said the opposite, which rendering made obvious and measurement settled.
    def spread(budget: float) -> float:
        values = []
        for stem in seeds:
            x, y = frontier(seeds[stem], "30")
            values.append(y[max(bisect.bisect_right(x, budget) - 1, 0)])
        return max(values) - min(values)

    here = spread(FROZEN)
    wider = [b for b in (0.125, 0.25, 1.0, 1.5, FAPD_MAX) if spread(b) <= here]
    if wider:
        raise SystemExit(
            f"the caption says the seed spread is narrowest at the frozen budget "
            f"({here:.4f}); it is no wider at {wider}"
        )
    print(f"wrote {arguments.output}  {len(ARMS)} arms and {len(seeds)} seeds, "
          f"{crossings} of {len(ARMS) * (len(ARMS) - 1) // 2} arm pairs reorder "
          f"between 0.25 and {FAPD_MAX:g} alarms per participant-day; seed spread "
          f"{here:.4f} at the frozen budget against {spread(1.5):.4f} at 1.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
