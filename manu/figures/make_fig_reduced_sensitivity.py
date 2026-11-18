"""Aggregate figure rendering functions."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'code/scripts/figures'))
sys.path.insert(0, str(PROJECT / 'code/src'))
import glycofig as G
SEEDS, HORIZONS, CUTS = ((17, 23, 42, 71, 101), (30, 60), (54, 70))
SENSITIVITY = 'equal_source_participant_macro_fixed_sensitivity'
BURDEN = 'equal_source_participant_macro_unmatched_alarms_per_patient_day'
CASES = (('full-primary', 'observed_only', 'full_sources', 'primary'), ('full-observed', 'observed_only', 'full_sources', 'observed_only'), ('common-primary', 'observed_only', 'common_support', 'primary'), ('common-observed', 'observed_only', 'common_support', 'observed_only'), ('span-primary', 'elapsed_span', 'full_sources', 'primary'), ('span-elapsed', 'elapsed_span', 'full_sources', 'elapsed_span'))
LABELS = {'primary': 'Primary', 'observed_only': 'Observed only', 'elapsed_span': 'Elapsed span'}
ROW_Y = (0.8, 1.8, 3.5, 4.5, 6.2, 7.2)
HEIGHT_MM = 175
CAPTION = "Event-definition sensitivity depends on source support and completeness frame. a,b Base event counts for the two horizons under each row's own catalogue and source set. c,d LSTM-minus-current-glucose-control differences in source-macro fixed sensitivity at gates 54 and 70 mg/dL. e,f Corresponding differences in achieved source-macro unmatched alarms per patient-day. Small marks show five fitted model seeds and larger marks their mean, without confidence intervals. Undefined sensitivity is explicitly labelled. Common support retains sources with positive events in both catalogues and both horizons, without intersecting participants or event onsets. The primary frame requires future completeness through 60/70 minutes for the 30/60-minute horizons. Both catalogues in the elapsed-span comparison instead use the shared 60/75-minute frame. Event counts are invariant to model seed and glucose gate."

def require(condition, message):
    if not condition:
        raise ValueError(message)

def make_figure(data):
    G.set_style()
    fig = plt.figure(figsize=(180 * G.MM, HEIGHT_MM * G.MM))
    axes, panels = ([], {})
    colours = {'primary': G.NOMINAL['blue'], 'observed_only': G.NOMINAL['rose'], 'elapsed_span': G.NOMINAL['teal']}
    count_index = {(r['case'], r['horizon_minutes']): r for r in data['counts']}
    largest_count = max((row['base_event_count'] for row in data['counts']))
    mean_records = []
    for metric in (SENSITIVITY, BURDEN):
        for case in data['cases']:
            for horizon in HORIZONS:
                for cut in CUTS:
                    records = [r for r in data['values'] if (r['metric'], r['case'], r['horizon_minutes'], r['glucose_cut']) == (metric, case['case'], horizon, cut)]
                    require(len(records) == 5 and {r['model_seed'] for r in records} == set(SEEDS), 'five required seed marks are missing')
                    values = [r['difference'] for r in records]
                    require(all((v is None for v in values)) or all((v is not None for v in values)), 'partial seed coverage cannot be plotted')
                    mean_records.append({'metric': metric, 'case': case['case'], 'horizon_minutes': horizon, 'glucose_cut': cut, 'mean': sum(values) / 5 if values[0] is not None else None})
    for row in range(3):
        metric = (None, SENSITIVITY, BURDEN)[row]
        scale = 100 if metric == SENSITIVITY else 1
        displayed = [r['difference'] * scale for r in data['values'] if r['metric'] == metric and r['difference'] is not None]
        low, high = (min([0, *displayed]), max([0, *displayed])) if displayed else (0, 1)
        pad = max((high - low) * 0.12, 0.02 if metric == BURDEN else 1.0)
        for col, horizon in enumerate(HORIZONS):
            ax = fig.add_axes([0.205 if col == 0 else 0.625, (0.715, 0.405, 0.095)[row], 0.35, 0.225])
            axes.append(ax)
            letter = chr(97 + row * 2 + col)
            G.panel_label(ax, letter, dx=-0.14, dy=1.12)
            title = ('Event counts', 'Sensitivity difference', 'Burden difference')[row]
            ax.set_title(f'{title}, {horizon} min', loc='left', pad=8, fontsize=6.5)
            panel_records, undefined = ([], [])
            for index, case in enumerate(data['cases']):
                y, colour = (ROW_Y[index], colours[case['catalogue']])
                if metric is None:
                    record = count_index[case['case'], horizon]
                    value = record['base_event_count']
                    hatch = {'primary': None, 'observed_only': '///', 'elapsed_span': 'xx'}[case['catalogue']]
                    ax.barh(y, value, height=0.57, color=colour, edgecolor=G.INK, linewidth=0.35, hatch=hatch)
                    ax.text(value + largest_count * 0.025, y, f'{value:,}', va='center', fontsize=5.7)
                    panel_records.append(record)
                else:
                    for j, cut in enumerate(CUTS):
                        records = [r for r in data['values'] if (r['case'], r['horizon_minutes'], r['glucose_cut'], r['metric']) == (case['case'], horizon, cut, metric)]
                        panel_records.extend(records)
                        if records[0]['difference'] is None:
                            if j == 0:
                                missing = len(records[0]['undefined_sources'])
                                ax.text(0.52, y, f'Undefined ({missing} source{('s' if missing != 1 else '')})', transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=5.5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.5})
                            undefined.append({'case': case['case'], 'glucose_cut': cut, 'sources': records[0]['undefined_sources']})
                            continue
                        values = np.asarray([r['difference'] for r in records]) * scale
                        centre = y + (-0.17 if j == 0 else 0.17)
                        marker = 'o' if j == 0 else 's'
                        ax.scatter(values, centre + np.linspace(-0.075, 0.075, 5), s=7.5, marker=marker, facecolors=colour if j == 0 else 'white', edgecolors=colour, linewidths=0.4, alpha=0.55)
                        ax.scatter(float(values.mean()), centre, s=24, marker=marker, facecolors=colour if j == 0 else 'white', edgecolors=colour, linewidths=0.75, zorder=4)
            if metric is None:
                ax.set_xlim(0, largest_count * 1.22)
                ax.set_xlabel('Base events')
            else:
                ax.axvline(0, linestyle=':', color=G.MUTED, linewidth=0.6, zorder=0)
                ax.set_xlim(low - pad, high + pad)
                ax.set_xlabel('Δ sensitivity (percentage points)' if metric == SENSITIVITY else 'Δ unmatched alarms per patient-day')
            limits = ax.get_xlim()
            ticks = matplotlib.ticker.MaxNLocator(nbins=4).tick_values(*limits)
            ax.set_xticks([tick for tick in ticks if limits[0] <= tick <= limits[1]])
            ax.set_ylim(7.9, -0.6)
            ax.set_yticks(ROW_Y, [case['label'] for case in data['cases']] if col == 0 else [''] * 6)
            ax.tick_params(axis='y', length=0, labelsize=5.7, pad=3)
            ax.spines['left'].set_visible(False)
            for divider in (2.25, 4.95):
                ax.axhline(divider, color='#D8DEE4', linewidth=0.4, zorder=0)
            if col == 0:
                headings = (f'{data['cases'][0]['source_count']} sources · Primary frame', f'{data['cases'][2]['source_count']} sources · Common support', f'{data['cases'][4]['source_count']} sources · Shared 60/75 min')
                for y, text in zip((-0.08, 2.62, 5.32), headings, strict=True):
                    ax.text(-0.56, y, text, transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=5.3, weight='bold')
            panels[letter] = {'quantity': 'base_event_count' if metric is None else metric, 'horizon_minutes': horizon, 'display_scale': scale, 'values': panel_records, 'undefined_rows': undefined, 'inference': 'descriptive five-seed points and mean; no confidence intervals' if metric else 'catalogue counts, no onset intersection'}
    handles = [Line2D([], [], color=G.INK, marker=marker, linestyle='none', markerfacecolor=G.INK if cut == 54 else 'white', label=f'Gate {cut} mg/dL') for cut, marker in ((54, 'o'), (70, 's'))]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.62, 0.035), ncol=2, fontsize=5.5, handletextpad=0.4)
    fig.text(0.01, 0.049, 'LSTM − current-glucose control', fontsize=5.3)
    fig.text(0.01, 0.029, 'Five seeds and larger mean · No CI', fontsize=5.3)
    fig.text(0.01, 0.009, 'Future completeness · Primary 60/70 min · Shared 60/75 min for 30/60-min horizons', fontsize=5.2)
    G.assert_compliant(fig, 'reduced Figure 7', len(CAPTION.split()))
    fig.canvas.draw()
    renderer, width, height = (fig.canvas.get_renderer(), *fig.canvas.get_width_height())
    overflow = [text.get_text() for text in fig.findobj(matplotlib.text.Text) if text.get_visible() and text.get_text().strip() and (text.get_window_extent(renderer).x0 < -0.5 or text.get_window_extent(renderer).y0 < -0.5 or text.get_window_extent(renderer).x1 > width + 0.5 or (text.get_window_extent(renderer).y1 > height + 0.5))]
    require(not overflow, f'figure text extends outside the final canvas: {overflow}')
    receipt = {'width_mm': 180, 'height_mm': HEIGHT_MM, 'font_family': 'Arial', 'delivery_mode': 'composite', 'annotation_profile': 'minimal', 'coverage': data['coverage'], 'cases': data['cases'], 'panels': panels, 'fit_means': mean_records, 'defined_fit_means': sum((row['mean'] is not None for row in mean_records)), 'undefined_fit_means': sum((row['mean'] is None for row in mean_records)), 'caption_draft': CAPTION, 'text_overflow_count': 0}
    return (fig, receipt)
