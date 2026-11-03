"""What forecast accuracy would have to be worth, priced rather than argued about.

WHY THIS FIGURE EXISTS. The control's advantage is that it reads the present exactly, so
the first thing a reader asks is why not read the future more accurately instead. The
sweep that answers it was run, gated and never cited: ten cells scoring the frozen
endpoint through a glucose forecast at controlled accuracies.

THE ABSCISSA IS mg/dL, NOT lambda. Lambda is the knob; achieved error is the quantity a
reader can compare against a published forecaster. Both were recorded, and plotting the
knob would have made the panel uninterpretable outside this project.

WHY THE TWO SWEEPS ARE BOTH DRAWN. They are matched on achieved error at every shared
setting, so the gap between them at one abscissa is what the error's STRUCTURE costs,
holding its size fixed. That is the only way to say the trained forecast is worse than
noise of the same magnitude without the sentence being a comparison of two different
things.

Geometry, fonts, palette and the zero-baseline rule are enforced by glycofig.save.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "code" / "scripts" / "figures")
)
import caption_words as _caption_words  # noqa: E402
import glycofig as G  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ARMS = (
    ("shrunk_toward_truth", "Shrunk toward the truth", "rose", "-", "s"),
    ("unbiased_noise", "Unbiased error", "blue", "--", "o"),
)
HORIZONS = ("30", "60")


def series(payload: dict, key: str, horizon: str) -> tuple[list[float], list[float]]:
    rows = sorted(payload[key], key=lambda r: r["achieved_rmse_mg_dl"][horizon])
    return ([r["achieved_rmse_mg_dl"][horizon] for r in rows],
            [r["primary"] for r in rows])


def _cell(directory: Path, stem: str, horizon: str) -> dict:
    path = directory / f"{stem}.json"
    if not path.is_file():
        raise SystemExit(f"{path.name} is missing; the sweep panels read the per-cell "
                         "results rather than the assembled summary")
    return json.loads(path.read_text())[horizon]["global"]


def _sweep_panels(axes, payload, cells_dir: Path, published: float, control: float):
    """Six panels the per-cell results carry and the assembled summary drops.

    What a controlled change in forecast accuracy does to the alarms charged, the events
    detected and the rate actually achieved; where the strongest published forecaster
    this manuscript could verify sits on the same axis; whether the cohorts move
    together; and how a second architecture's forecast compares.
    """
    ax_c, ax_e, ax_f, ax_g, ax_h, ax_j = axes
    blend = sorted(payload["shrunk_toward_truth"],
                   key=lambda r: r["achieved_rmse_mg_dl"]["30"])
    crossings = payload["crossing_of_control"]

    # c: where a published forecaster would land on this axis.
    marks = [
        ("this study's own forecast", blend[-1]["achieved_rmse_mg_dl"]["30"], "rose"),
        ("shrunk crossing", crossings["shrunk_toward_truth"]["mg_dl"], "olive"),
        ("unbiased crossing", crossings["unbiased_noise"]["mg_dl"], "amber"),
        ("strongest published", published, "blue"),
    ]
    for index, (_label, value, hue) in enumerate(marks):
        ax_c.plot([value], [index], marker="o", markersize=3.0, color=G.NOMINAL[hue],
                  zorder=3)
        ax_c.plot([0.0, value], [index, index], color=G.NOMINAL[hue], lw=0.7,
                  solid_capstyle="butt", zorder=2)
        ax_c.annotate(f"{value:.2f}", (value, index), fontsize=5.0,
                      textcoords="offset points", xytext=(3.0, 0.0), va="center",
                      color=G.INK, zorder=4)
    ax_c.set_yticks(range(len(marks)))
    ax_c.set_yticklabels([lab for lab, *_ in marks], fontsize=5.0)
    ax_c.set_ylim(-0.6, len(marks) - 0.4)
    ax_c.set_xlim(0.0, 30.0)
    ax_c.set_xlabel("30-minute forecast error (mg/dL)")

    # e, f, g: what the sweep does to the alarm ledger, not only to the endpoint.
    for axis, field, ylabel, scale in (
        (ax_e, "alarm_count", "Alarms charged (thousands)", 1e-3),
        (ax_f, "detected_event_count", "Events detected", 1.0),
        (ax_g, "fapd", "Achieved false alarms per\nparticipant-day", 1.0),
    ):
        for horizon, hue, mark in (("30", "rose", "s"), ("60", "blue", "o")):
            xs, ys = [], []
            for row in blend:
                block = _cell(cells_dir, row["cell"], horizon)
                xs.append(row["achieved_rmse_mg_dl"][horizon])
                ys.append(float(block[field]) * scale)
            pairs = sorted(zip(xs, ys, strict=True))
            axis.plot([a for a, _b in pairs], [b for _a, b in pairs],
                      color=G.NOMINAL[hue], marker=mark, markersize=2.4,
                      linewidth=0.8, zorder=3, label=f"{horizon} min")
        axis.set_xlabel("Achieved forecast error (mg/dL)")
        axis.set_ylabel(ylabel)
    ax_g.axhline(0.5, color=G.INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=2)
    ax_e.legend(loc="upper left", fontsize=5.0, handlelength=1.3,
                borderaxespad=0.2, labelspacing=0.2)

    # h: and whether the cohorts move together as the forecast improves.
    worst, best = blend[-1], blend[0]
    per_worst = _cell(cells_dir, worst["cell"], "30")["per_source"]
    per_best = _cell(cells_dir, best["cell"], "30")["per_source"]
    shared = sorted(set(per_worst) & set(per_best))
    for source in shared:
        ax_h.plot([0, 1], [float(per_worst[source]["sensitivity"]),
                           float(per_best[source]["sensitivity"])],
                  color=G.NEUTRAL, lw=0.6, marker="o", markersize=1.8, zorder=3)
    ax_h.set_xlim(-0.2, 1.2)
    ax_h.set_xticks([0, 1])
    ax_h.set_xticklabels([f"{worst['achieved_rmse_mg_dl']['30']:.0f}",
                          f"{best['achieved_rmse_mg_dl']['30']:.0f}"], fontsize=5.0)
    ax_h.set_ylim(0.0, 1.05)
    ax_h.set_xlabel("Forecast error (mg/dL)")
    ax_h.set_ylabel("Sensitivity by cohort")

    # j: a second architecture's forecast, against the arm the sweep is built on.
    bars = [("LSTM", float(payload["own_forecast_primary"]), "blue", ""),
            ("TCN", float(payload["second_architecture_forecast_primary"]),
             "amber", "///")]
    for index, (_label, value, hue, hatch) in enumerate(bars):
        ax_j.bar([index], [value], width=0.6, color=G.NOMINAL[hue],
                 hatch=hatch or None, edgecolor=G.INK, linewidth=0.35, zorder=3)
    ax_j.axhline(control, color=G.INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=4)
    ax_j.set_xticks(range(len(bars)))
    ax_j.set_xticklabels([lab for lab, *_ in bars], fontsize=5.0, rotation=38,
                         ha="right")
    ax_j.set_ylim(0.0, 1.0)
    ax_j.set_ylabel("Primary endpoint")
    if published >= crossings["shrunk_toward_truth"]["mg_dl"]:
        raise SystemExit(
            f"the caption says the published forecaster sits below the shrunk crossing; "
            f"it is {published} against {crossings['shrunk_toward_truth']['mg_dl']:.2f}"
        )
    return len(shared)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--cells-dir", required=True, type=Path,
                        help="the per-cell result files behind the assembled sweep")
    parser.add_argument("--published-rmse", type=float, default=6.82,
                        help="the strongest published 30-minute forecast error the "
                             "manuscript cites; passed in rather than hardcoded")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height-mm", type=float, default=185.0)
    parser.add_argument("--caption-words", type=int,
                        default=_caption_words.count("fig_forecast"))
    arguments = parser.parse_args(argv)

    payload = json.loads(arguments.input.read_text())
    control = float(payload["control_primary"])
    if not payload.get("matched_on_achieved_rmse"):
        raise SystemExit("the two sweeps are not matched on achieved error")

    G.set_style()
    figure = plt.figure(figsize=(G.COL2_MM * G.MM, arguments.height_mm * G.MM))
    # Rows are laid out from one gap constant rather than three hand-set numbers.
    # The gaps here were 0.09 to 0.13 of the canvas, which on a 185 mm figure is a
    # blank band of 17 to 24 mm between rows, and only about 10 mm of it is doing
    # any work: the tick labels and axis label of the row above, plus clearance for
    # the panel letter of the row below. The rest was dead. Setting one gap and
    # solving for the row height gives that space back to the panels.
    _bottom, _top, _gap = 0.078, 0.956, 0.070
    _tall = (_top - _bottom - 2 * _gap) / 3
    _r1, _r2, _r3 = _bottom + 2 * (_tall + _gap), _bottom + _tall + _gap, _bottom
    ax_a = figure.add_axes([0.100, _r1, 0.230, _tall])
    ax_b = figure.add_axes([0.440, _r1, 0.215, _tall])
    ax_c = figure.add_axes([0.775, _r1, 0.200, _tall])
    # The first layout left 30 mm of white between the second and third rows.
    ax_d = figure.add_axes([0.100, _r2, 0.230, _tall])
    ax_e = figure.add_axes([0.455, _r2, 0.190, _tall])
    ax_f = figure.add_axes([0.775, _r2, 0.200, _tall])
    ax_g = figure.add_axes([0.100, _r3, 0.165, _tall])
    ax_h = figure.add_axes([0.360, _r3, 0.165, _tall])
    ax_i = figure.add_axes([0.620, _r3, 0.165, _tall])
    ax_j = figure.add_axes([0.870, _r3, 0.105, _tall])

    # ---- a, b: the dose-response at each horizon -----------------------------------
    for axis, horizon in zip((ax_a, ax_b), HORIZONS, strict=True):
        axis.axhline(control, color=G.INK, lw=0.6, ls=(0, (2.4, 1.6)), zorder=2)
        for key, label, hue, dash, mark in ARMS:
            x, y = series(payload, key, horizon)
            axis.plot(x, y, color=G.NOMINAL[hue], linestyle=dash, marker=mark,
                      markersize=2.8, linewidth=0.9, zorder=3,
                      label=label if horizon == "30" else None)
        if horizon == "30":
            for key, _label, hue, _dash, _mark in ARMS:
                cross = payload["crossing_of_control"][key]
                axis.plot([cross["mg_dl"]], [control], marker="x", markersize=4.0,
                          markeredgewidth=1.0, color=G.NOMINAL[hue], zorder=5)
                # The two crossings are 2.8 mg/dL apart and their labels collided.
                # 2.8 mg/dL apart on this axis, so the two labels are pushed apart in
                # both directions and away from their own markers.
                dx, dy, align = ((7.0, -6.0, "left") if key == "shrunk_toward_truth"
                                 else (-7.0, -12.0, "right"))
                axis.annotate(
                    f"{cross['mg_dl']:.1f}", (cross["mg_dl"], control), fontsize=5.0,
                    textcoords="offset points", xytext=(dx, dy), ha=align,
                    color=G.NOMINAL[hue], zorder=5)
        axis.set_title(f"{horizon}-minute RMSE", loc="left", fontsize=6.0, pad=3)
        axis.set_xlabel("Achieved RMSE (mg/dL)")
        axis.set_ylim(0.0, 1.0)
    ax_a.set_ylabel("Primary sensitivity\n(mean over horizons)")
    ax_a.annotate("control", (max(series(payload, "shrunk_toward_truth", "30")[0]),
                              control), fontsize=5.0, color=G.INK,
                  textcoords="offset points", xytext=(-2.0, -7.5), ha="right")
    ax_a.legend(loc="upper right", handlelength=2.0, borderaxespad=0.3,
                labelspacing=0.26)

    # ---- c: at a matched error, what the structure costs ----------------------------
    shared = set(payload["shared_settings"])
    by_lambda = {
        key: {r["lambda"]: r for r in payload[key]} for key, *_ in ARMS
    }
    points = sorted(
        (by_lambda["shrunk_toward_truth"][lam]["achieved_rmse_mg_dl"]["30"],
         by_lambda["unbiased_noise"][lam]["primary"]
         - by_lambda["shrunk_toward_truth"][lam]["primary"])
        for lam in shared
    )
    ax_d.axhline(0.0, color=G.INK, lw=0.6, zorder=2)
    ax_d.plot([p[0] for p in points], [p[1] for p in points],
              color=G.NOMINAL["olive"], linestyle="-.", marker="^", markersize=2.8,
              linewidth=0.9, zorder=3)
    ax_d.set_xlabel("Achieved forecast error (mg/dL)")
    ax_d.set_ylabel("Unbiased minus shrunk")
    ax_d.annotate("noise is better", (points[-1][0], points[-1][1]), fontsize=5.0,
                  color=G.MUTED, textcoords="offset points", xytext=(-3.0, -7.0),
                  ha="right")

    # ---- d: why, in one diagnostic --------------------------------------------------
    diagnostic = payload["diagnostic"]
    rows = (("corr_prediction_with_current", "The forecast", "rose", "s"),
            ("corr_truth_with_current", "The observed future", "blue", "o"))
    for index, (field, label, hue, mark) in enumerate(rows):
        for offset, horizon in zip((-0.15, 0.15), HORIZONS, strict=True):
            value = diagnostic[horizon][field]
            ax_i.plot([index + offset], [value], marker=mark, markersize=3.4,
                      markerfacecolor=G.NOMINAL[hue] if horizon == "30" else "white",
                      markeredgecolor=G.NOMINAL[hue], markeredgewidth=0.9,
                      linestyle="none", zorder=3, label=label if horizon == "30" else None)
            ax_i.annotate(f"{value:.3f}", (index + offset, value), fontsize=5.0,
                          textcoords="offset points", xytext=(0, 4.0), ha="center",
                          color=G.INK, zorder=4)
    ax_i.set_xticks(range(len(rows)))
    ax_i.set_xticklabels(["Forecast", "Truth"], fontsize=5.4)
    ax_i.set_xlim(-0.55, len(rows) - 0.45)
    ax_i.set_ylim(0.65, 1.0)
    ax_i.set_ylabel("Correlation with current glucose")
    ax_i.set_xlabel("Filled, 30 minutes; open, 60")

    order = (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i, ax_j)
    _sweep_panels((ax_c, ax_e, ax_f, ax_g, ax_h, ax_j), payload, arguments.cells_dir,
                  arguments.published_rmse, control)
    for axis in order:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
    for axis, letter in zip(order, "abcdefghij", strict=True):
        box = axis.get_position()
        figure.text(box.x0 - 0.058, box.y1 + 0.022, f"{letter})", fontsize=7.0,
                    fontweight="bold", color=G.INK, va="top")

    G.save(figure, str(arguments.output), caption_words=arguments.caption_words)

    # The caption says the forecast resembles the present more than the future does.
    for horizon in HORIZONS:
        block = diagnostic[horizon]
        if not block["corr_prediction_with_current"] > block["corr_truth_with_current"]:
            raise SystemExit(
                f"h={horizon}: the forecast is not more like the present than the truth "
                "is, so the over-persistence sentence is wrong"
            )
    cs = payload["crossing_of_control"]
    print(f"wrote {arguments.output}  crossings "
          f"{cs['shrunk_toward_truth']['mg_dl']:.2f} and "
          f"{cs['unbiased_noise']['mg_dl']:.2f} mg/dL, both "
          f"{cs['unbiased_noise']['basis'].split()[0]}; the arm's own forecast sits at "
          f"{payload['shrunk_toward_truth'][0]['achieved_rmse_mg_dl']['30']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
