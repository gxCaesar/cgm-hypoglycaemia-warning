"""Aggregate figure rendering functions."""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'code/scripts/figures'))
import glycofig as G
SOURCES = ('AZT1D', 'BrisT1D', 'CTR3', 'DCLP3', 'DCLP5', 'Flair', 'IOBP2', 'PEDAP', 'ReplaceBG', 'Shanghai T1DM', 'T1D-UOM')
CONTRACTS = {'revision_contract': '2026-11-04-fixed-frame', 'ascertainment_contract': '2026-11-09-complete-confirmation'}
SOURCE_FILES = ('source-influence.json', 'status.json', 'estimands.csv', 'joint-coverage.csv', 'source-contributions.csv', 'leave-one-source-out.csv')
GRID_FILES = ('export-note.json', 'primary-inference.json', 'condition-rows.csv', 'source-rows.csv')
CAPTION = "Supplementary Figure 5 | Development source influence. a) Each source's interaction. The interaction is the increase from the 54 to 70 mg/dL cut in the LSTM-minus-current-CGM difference in fixed-frame participant-macro sensitivity. Source-specific interactions average the 30- and 60-minute horizons and five LSTM seeds, with one shared current-CGM control. Their mean is the full equal-source interaction. b) The interaction after deleting each named source and reweighting the remaining ten sources equally. The dashed line marks the full eleven-source interaction. These are descriptive reaggregations without retraining or confidence intervals. All policies use the nominal calibration budget of 0.5 false alarms per patient-day; achieved burden can differ."

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def number(value, location: str) -> float:
    require(value is not None, f'{location}: undefined number')
    require(not isinstance(value, bool), f'{location}: boolean is not a number')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{location}: invalid number') from exc
    require(math.isfinite(result), f'{location}: non-finite or undefined number')
    return result

def near(value, expected: float, location: str) -> None:
    require(math.isclose(number(value, location), expected, rel_tol=1e-09, abs_tol=1e-10), f'{location}: arithmetic mismatch')

def keyed(rows: list[dict], key: str, location: str) -> dict:
    require(len(rows) == len(SOURCES), f'{location}: requires eleven sources')
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows) and set(result) == set(SOURCES), f'{location}: missing, duplicate or unexpected source')
    return result

def draw(data: dict):
    G.set_style()
    fig = plt.figure(figsize=(180 * G.MM, 90 * G.MM))
    axes = [fig.add_axes([0.145, 0.18, 0.37, 0.72]), fig.add_axes([0.6, 0.18, 0.375, 0.72])]
    y = np.arange(len(SOURCES))
    interactions = np.array([r['source_interaction'] for r in data['rows']])
    deleted = np.array([r['delete_one_source_interaction'] for r in data['rows']])
    for index, ax in enumerate(axes):
        ax.set_ylim(len(SOURCES) - 0.5, -0.5)
        ax.set_yticks(y, SOURCES if index == 0 else [''] * len(SOURCES))
        ax.tick_params(axis='y', length=0, pad=4)
        ax.spines['left'].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E7E9EC', linewidth=0.4)
        ax.text(-0.055, 1.095, f'{chr(ord('a') + index)})', transform=ax.transAxes, fontsize=7, fontweight='bold', color=G.INK, va='top', ha='left')
    left, right = axes
    left.axvline(0, color=G.MUTED, linewidth=0.6)
    left.hlines(y, 0, interactions, color=G.NEUTRAL, linewidth=0.8)
    for positive, color, marker in ((True, G.NOMINAL['blue'], 'o'), (False, G.NOMINAL['rose'], 'D')):
        selected = interactions >= 0 if positive else interactions < 0
        left.scatter(interactions[selected], y[selected], color=color, marker=marker, s=15, zorder=3)
    lo, hi = (min(0, interactions.min()), max(0, interactions.max()))
    padding = max(hi - lo, 0.01) * 0.08
    left.set_xlim(lo - padding, hi + padding)
    left.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 4, 5, 10]))
    left.set_title('Source-specific interaction', loc='left', pad=9)
    left.set_xlabel('Source-specific interaction')
    right.hlines(y, data['full_contrast'], deleted, color=G.NEUTRAL, linewidth=0.8)
    right.axvline(data['full_contrast'], color=G.INK, linestyle='--', linewidth=0.8)
    right.scatter(deleted, y, color=G.NOMINAL['amber'], marker='s', s=15, zorder=3)
    lo, hi = (min(deleted.min(), data['full_contrast']), max(deleted.max(), data['full_contrast']))
    padding = max(hi - lo, 0.01) * 0.12
    right.set_xlim(lo - padding, hi + padding)
    right.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 4, 5, 10]))
    right.set_title('Delete-one-source contrast', loc='left', pad=9)
    right.set_xlabel('Interaction after excluding source')
    right.text(data['full_contrast'] + 0.002, 1.01, f'Full {data['full_contrast']:.3f}', transform=right.get_xaxis_transform(), fontsize=5.5, va='bottom')
    for ax in axes:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.text(0.145, 0.035, 'Equal horizon/seed means; descriptive reaggregation, no retraining or CI', fontsize=5.5, color=G.MUTED)
    G.assert_compliant(fig, 'Supplementary Figure 5', len(CAPTION.split()))
    return fig
