"""Aggregate figure rendering functions."""
from __future__ import annotations
import argparse
import csv
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
import glycofig as G
ARMS = ('persistence', 'linear', 'summary_logistic', 'xgboost', 'lstm', 'tcn')
SEEDS = (17, 23, 42, 71, 101)
HEADS = ('hypo_30', 'hypo_60', 'hyper_30', 'hyper_60')
FORECAST_ARMS = ('persistence', 'linear', 'lstm', 'tcn')
IDENTITIES = tuple(((a, s) for a in ARMS for s in (SEEDS if a in ('lstm', 'tcn') else (None,))))
LABELS = {'persistence': 'Current CGM', 'linear': 'Linear', 'summary_logistic': 'Summary logistic', 'xgboost': 'XGBoost', 'lstm': 'LSTM', 'tcn': 'TCN'}
COLOURS = {'persistence': G.NOMINAL['rose'], 'linear': G.NEUTRAL, 'summary_logistic': G.NOMINAL['amber'], 'xgboost': G.NOMINAL['teal'], 'lstm': G.NOMINAL['blue'], 'tcn': G.NOMINAL['olive']}
MARKERS = dict(zip(ARMS, ('o', 's', '^', 'D', 'v', 'P'), strict=True))
STYLES = dict(zip(ARMS, ('-', '--', ':', '-.', '-', '--'), strict=True))
CONTRACT = '2026-11-04-fixed-frame'
SENSITIVITY = 'equal_source_participant_macro_fixed_sensitivity'
BURDEN = 'equal_source_participant_macro_unmatched_alarms_per_patient_day'
PARITY_FIELDS = (SENSITIVITY, BURDEN, 'ppv', 'base_event_count', 'participant_count', 'patient_days', 'observation_minutes', 'matched_alarm_count', 'unmatched_alarm_count', 'unevaluable_alarm_count')
ROLES = ('selection', 'calibration')
DENOMINATOR_FIELDS = ('source_count', 'participant_count', 'patient_days', 'observation_minutes', 'base_event_count', 'base_evaluable_decision_count', 'event_positive_participant_count')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def finite(value, label, *, undefined=False):
    if value is None or (isinstance(value, (int, float)) and (not math.isfinite(value))):
        require(undefined, f'undefined/non-finite {label}')
        return None
    require(type(value) in (float, int) and math.isfinite(value), f'invalid {label}')
    return float(value)

def near(a, b, label):
    require(math.isclose(float(a), float(b), rel_tol=1e-09, abs_tol=1e-10), f'aggregate mismatch: {label}')

def diagnostic_metrics(diagnostic):
    if diagnostic['metrics'] is None:
        require(diagnostic['n'] == 0, 'native null probability metrics require zero retained windows')
        return {'n': 0, 'prevalence': None, 'auprc': None, 'auroc': None, 'brier': None, 'ece': None}
    return diagnostic['metrics']

def record(data, identity, quantity, value, *, horizon, n, series, pointer, **extra):
    return {'arm': identity[0], 'model_seed': identity[1], 'partition': 'selection', 'horizon_minutes': horizon, 'quantity': quantity, 'series': str(series), 'value': finite(value, quantity, undefined=True), 'n': n, 'source': str(data['paths'][identity]), 'json_pointer': pointer, **extra}

def fixed_rows(data, metric, horizon, *, budget='0.5', series=None):
    rows = []
    for identity in IDENTITIES:
        frame = data['reports'][identity]['partitions']['selection']['fixed_frame'][str(horizon)][budget]
        n, n_unit = (frame['base_event_count'], 'base_events')
        if metric.startswith('matched_only'):
            n, n_unit = (frame['matched_alarm_count'], 'matched_alarms')
        elif metric == BURDEN:
            n, n_unit = (frame['patient_days'], 'participant_calendar_days')
        elif metric == SENSITIVITY:
            n, n_unit = (frame['event_positive_participant_count'], 'event_positive_participants_within_11_sources')
        elif metric.endswith('_alarm_count'):
            n, n_unit = (frame['participant_count'], 'base_participants')
        rows.append(record(data, identity, metric, frame[metric], horizon=horizon, n=n, n_unit=n_unit, series=series or budget, pointer=f'/partitions/selection/fixed_frame/{horizon}/{budget}/{metric}', nominal_fapd=float(budget), participant_count=frame['participant_count'], patient_days=frame['patient_days'], observation_minutes=frame['observation_minutes'], base_event_count=frame['base_event_count'], matched_alarm_count=frame['matched_alarm_count']))
    return rows

def probability_rows(data, metric, head, *, series):
    rows = []
    for identity in IDENTITIES:
        diag = data['reports'][identity]['partitions']['selection']['probability_diagnostics'][head]
        metrics = diagnostic_metrics(diag)
        pointer = f'/partitions/selection/probability_diagnostics/{head}/metrics'
        rows.append(record(data, identity, metric, metrics[metric], horizon=int(head.split('_')[1]), n=diag['n'], series=series, pointer=pointer if diag['metrics'] is None else f'{pointer}/{metric}', head=head, n_unit='windows', prevalence=metrics['prevalence'], score_status='temperature_scaled_probability', temperature_fit_partition='calibration'))
    return rows

def title(ax, letter, text):
    G.panel_label(ax, letter, dx=-0.13, dy=1.09)
    ax.set_title(text, loc='left', fontsize=6.5, pad=5)

def point_panel(ax, rows, series, *, horizontal=False, arms=ARMS):
    offsets = np.linspace(-0.16, 0.16, len(series)) if len(series) > 1 else [0]
    for i, arm in enumerate(arms):
        for j, name in enumerate(series):
            group = [r for r in rows if r['arm'] == arm and r['series'] == str(name)]
            require(len(group) == (5 if arm in ('lstm', 'tcn') else 1), 'plot silently lost an actual fit')
            values = [r['value'] for r in group if r['value'] is not None]
            centre = i + offsets[j]
            fill = COLOURS[arm] if j == 0 else 'white'
            if len(group) > 1 and values:
                locations = centre + np.linspace(-0.045, 0.045, len(values))
                ax.scatter(values if horizontal else locations, locations if horizontal else values, s=8, marker=MARKERS[arm], color=COLOURS[arm], alpha=0.42, linewidths=0.3, clip_on=False)
            if len(values) == len(group):
                mean = float(np.mean(values))
                ax.scatter(mean if horizontal else centre, centre if horizontal else mean, s=23, marker=MARKERS[arm], facecolors=fill, edgecolors=COLOURS[arm], linewidths=0.75, zorder=4, clip_on=False)
            else:
                ax.annotate('NA', (0, centre) if horizontal else (centre, 0), fontsize=5)
    labels = [LABELS[a] for a in arms]
    if horizontal:
        ax.set_yticks(range(len(arms)), labels)
        ax.set_ylim(len(arms) - 0.5, -0.5)
    else:
        ax.set_xticks(range(len(arms)), labels, rotation=38, ha='right')
        ax.set_xlim(-0.5, len(arms) - 0.5)
        ax.set_ylim(bottom=0)
    ax.grid(axis='x' if horizontal else 'y', color='#dddddd', linewidth=0.35, zorder=0)

def series_legend(ax, labels):
    handles = [Line2D([], [], marker='o', color=G.INK, linestyle='none', markersize=3, markerfacecolor=G.INK if i == 0 else 'white', label=label) for i, label in enumerate(labels)]
    ax.legend(handles=handles, loc='upper right', ncol=len(labels), handletextpad=0.3, columnspacing=0.8, fontsize=5)

def arm_legend(fig):
    fig.legend(handles=[Line2D([], [], color=COLOURS[a], marker=MARKERS[a], linestyle=STYLES[a], label=LABELS[a]) for a in ARMS], loc='upper center', ncol=6, fontsize=5.5, handlelength=1.8, columnspacing=1.1, bbox_to_anchor=(0.53, 0.995))

def save(fig, out, name, panels, receipt):
    for panel in panels.values():
        groups = {}
        for row in panel['rows']:
            groups.setdefault((row['arm'], row['series'], row['quantity']), []).append(row['value'])
        panel['plotted_fit_means'] = [{'arm': k[0], 'series': k[1], 'quantity': k[2], 'fit_count': len(v), 'mean': float(np.mean(v)) if all((x is not None for x in v)) else None} for k, v in groups.items()]
    fig.canvas.draw()
    with matplotlib.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0}):
        fig.savefig(out / f'{name}.png', dpi=220)
    receipt[name] = {'width_mm': 180, 'height_mm': fig.get_figheight() / G.MM, 'panels': panels, 'encoding': 'large marks are arithmetic means of fit-level metrics; small deep-model marks are five seeds; no CIs'}
    G.save(fig, str(out / f'{name}.pdf'), caption_words=200)

def figure5(data, out, receipt):
    fig, axs = plt.subplots(4, 2, figsize=(180 * G.MM, 178 * G.MM), gridspec_kw={'height_ratios': [1.15, 1, 1, 1.05]})
    fig.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.09, hspace=0.8, wspace=0.42)
    arm_legend(fig)
    panels = {}
    for column, horizon in enumerate((30, 60)):
        ax = axs[0, column]
        rows = []
        for lead in (5, 10, 15, 20):
            rows.extend(fixed_rows(data, f'coverage_{lead}_event_fraction', horizon, series=str(lead)))
        for arm in ARMS:
            means = []
            for lead in (5, 10, 15, 20):
                group = [r['value'] * 100 for r in rows if r['arm'] == arm and r['series'] == str(lead)]
                means.append(float(np.mean(group)))
                if len(group) == 5:
                    ax.scatter(np.full(5, lead), group, s=5, color=COLOURS[arm], marker=MARKERS[arm], alpha=0.35)
            ax.plot((5, 10, 15, 20), means, color=COLOURS[arm], marker=MARKERS[arm], linestyle=STYLES[arm])
        ax.set(xticks=(5, 10, 15, 20), ylim=(0, 100), xlabel='Minimum lead (min)', ylabel='Base events covered (%)')
        letter = chr(97 + column)
        title(ax, letter, f'Event coverage, {horizon} min')
        panels[letter] = {'quantity': 'event-pooled coverage of fixed base events', 'display_scale': 100, 'rows': rows}
        ax = axs[1, column]
        rows = fixed_rows(data, 'matched_only_lead_minutes_mean', horizon, series='Mean')
        rows += fixed_rows(data, 'matched_only_lead_minutes_median', horizon, series='Median')
        point_panel(ax, rows, ('Mean', 'Median'), horizontal=True)
        ax.set_xlabel('Matched-only lead per fit (min)')
        series_legend(ax, ('Mean', 'Median'))
        letter = chr(99 + column)
        title(ax, letter, f'Alarm timing, {horizon} min')
        panels[letter] = {'quantity': 'mean and median within each actual fit, never pooled seed quantiles', 'rows': rows}
        ax = axs[2, column]
        rows = []
        for identity in IDENTITIES:
            for category in ('matched', 'unmatched'):
                alarm = data['alarms'][*identity, horizon, category]
                rows.append(record(data, identity, 'global_alarm_glucose_median', alarm['median'], horizon=horizon, n=alarm['alarm_count'], n_unit=f'{category}_alarms', series=category, pointer=alarm['source_json_pointer'], source=str(data['alarm_source']), source_kind='accepted_alarm_glucose_summary', alarm_category=category))
        point_panel(ax, rows, ('matched', 'unmatched'), horizontal=True)
        ax.set_xlabel('Median CGM per fit (mg/dL)')
        series_legend(ax, ('Matched', 'Unmatched'))
        letter = chr(101 + column)
        title(ax, letter, f'CGM at alarms, {horizon} min')
        panels[letter] = {'quantity': 'global median within actual condition; seed means are not pooled medians/IQRs', 'rows': rows}
        ax = axs[3, column]
        rows = []
        for name in ('matched', 'unmatched', 'unevaluable'):
            rows += fixed_rows(data, f'{name}_alarm_count', horizon, series=name)
        left = np.zeros(6)
        for name, colour, hatch in zip(('matched', 'unmatched', 'unevaluable'), ('#555555', '#bbbbbb', 'white'), ('', '///', 'xx'), strict=True):
            values = [np.mean([r['value'] for r in rows if r['arm'] == a and r['series'] == name]) for a in ARMS]
            ax.barh(range(6), values, left=left, height=0.64, color=colour, hatch=hatch, linewidth=0.35, edgecolor=G.INK, label=name.capitalize())
            left += values
        for i, arm in enumerate(ARMS):
            if arm in ('lstm', 'tcn'):
                totals = [data['reports'][arm, seed]['partitions']['selection']['fixed_frame'][str(horizon)]['0.5']['emitted_alarm_count'] for seed in SEEDS]
                ax.scatter(totals, i + np.linspace(-0.08, 0.08, 5), s=9, marker=MARKERS[arm], color=COLOURS[arm], edgecolor='white', linewidths=0.25, zorder=5)
        ax.set(yticks=range(6), yticklabels=[LABELS[a] for a in ARMS], ylim=(5.5, -0.5), xlabel='Mean emitted alarm count')
        ax.set_xlim(left=0)
        letter = chr(103 + column)
        title(ax, letter, f'Alarm accounting, {horizon} min')
        panels[letter] = {'quantity': 'stacked mean counts across five deep seeds or one actual control', 'rows': rows + fixed_rows(data, 'emitted_alarm_count', horizon, series='emitted')}
    handles, labels = axs[3, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.55, 0.034), ncol=3, fontsize=5, handlelength=1.0, columnspacing=1.0)
    fig.text(0.12, 0.012, 'Selection · CGM ≥54 mg/dL · nominal 0.5 FA/day. Large marks: fit means; small deep-model marks: five seeds.', fontsize=5.5)
    save(fig, out, 'fig5-corrected-candidate', panels, receipt)

def figure6(data, out, receipt):
    fig, axs = plt.subplots(2, 3, figsize=(180 * G.MM, 140 * G.MM), gridspec_kw={'width_ratios': [1.08, 1.08, 0.92]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.16, hspace=0.68, wspace=0.5)
    fig.legend(handles=[Line2D([], [], marker='o', color=G.INK, linestyle='none', markersize=3, markerfacecolor=G.INK if i == 0 else 'white', label=f'{h} min') for i, h in enumerate((30, 60))], loc='upper center', ncol=2, fontsize=5.5)
    panels = {}
    for ax, metric, letter in ((axs[0, 0], 'auprc', 'a'), (axs[0, 1], 'auroc', 'b'), (axs[1, 0], 'brier', 'd'), (axs[1, 1], 'ece', 'e')):
        rows = sum((probability_rows(data, metric, f'hypo_{h}', series=str(h)) for h in (30, 60)), [])
        point_panel(ax, rows, ('30', '60'))
        title(ax, letter, f'Level-2 window {('Brier' if metric == 'brier' else metric.upper())}')
        panels[letter] = {'quantity': 'calibration-fitted temperature-scaled probability diagnostic evaluated on selection windows', 'rows': rows}
    for row, metric, letter in ((0, 'rmse', 'c'), (1, 'mae', 'f')):
        rows = []
        for identity in IDENTITIES:
            if identity[0] not in FORECAST_ARMS:
                continue
            for horizon in (30, 60):
                metrics = data['reports'][identity]['partitions']['selection']['continuous_forecast'][str(horizon)]
                rows.append(record(data, identity, metric, metrics[metric], horizon=horizon, n=metrics['n'], series=str(horizon), pointer=f'/partitions/selection/continuous_forecast/{horizon}/{metric}', n_unit='windows'))
        point_panel(axs[row, 2], rows, ('30', '60'), arms=FORECAST_ARMS)
        axs[row, 2].set_ylabel('mg/dL')
        title(axs[row, 2], letter, f'Forecast {metric.upper()}')
        panels[letter] = {'quantity': 'native continuous glucose forecast error; no logistic/XGBoost forecast', 'rows': rows}
    part = data['reports']['persistence', None]['partitions']['selection']
    probability_counts = ' / '.join((f'{part['probability_diagnostics'][f'hypo_{h}']['n']:,}' for h in (30, 60)))
    forecast_counts = ' / '.join((f'{part['continuous_forecast'][str(h)]['n']:,.0f}' for h in (30, 60)))
    fig.text(0.075, 0.045, f'Selection · 30 / 60 min: Level-2 n={probability_counts}; forecast n={forecast_counts}. Filled: 30 min; open: 60 min.', fontsize=5.5)
    fig.text(0.075, 0.025, 'Calibration-fitted temperatures · Selection diagnostics · Large marks: fit means; small deep-model marks: five seeds.', fontsize=5.5)
    save(fig, out, 'fig6-corrected-candidate', panels, receipt)

def supplement1(data, out, receipt):
    fig, axs = plt.subplots(2, 2, figsize=(180 * G.MM, 133 * G.MM))
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.13, hspace=0.58, wspace=0.35)
    arm_legend(fig)
    panels = {}
    for row, metric in enumerate((SENSITIVITY, BURDEN)):
        for col, horizon in enumerate((30, 60)):
            ax = axs[row, col]
            rows = sum((fixed_rows(data, metric, horizon, budget=b) for b in ('0.5', '1.0')), [])
            for arm in ARMS:
                means = []
                for budget in (0.5, 1.0):
                    group = [r['value'] for r in rows if r['arm'] == arm and r['nominal_fapd'] == budget]
                    means.append(float(np.mean(group)))
                    if len(group) == 5:
                        ax.scatter(np.full(5, budget), group, s=7, marker=MARKERS[arm], color=COLOURS[arm], alpha=0.4)
                ax.plot((0.5, 1.0), means, marker=MARKERS[arm], color=COLOURS[arm], linestyle=STYLES[arm])
            ax.set_xticks((0.5, 1.0), ('0.5', '1.0'))
            ax.set_xlim(0.44, 1.06)
            ax.set_xlabel('Nominal calibration FA/day')
            ax.set_ylabel('Fixed sensitivity' if row == 0 else 'Achieved unmatched alarms/day')
            if row == 0:
                ax.set_ylim(0, 1)
            letter = chr(97 + row * 2 + col)
            title(ax, letter, f'{('Sensitivity' if row == 0 else 'Achieved burden')}, {horizon} min')
            panels[letter] = {'quantity': 'selection response at the two actual frozen calibration policies, not a dense sweep', 'rows': rows}
    population = data['population']['selection']
    counts = {field: ' / '.join((f'{population[str(h)][field]:,}' for h in (30, 60))) for field in ('participant_count', 'patient_days', 'base_event_count')}
    fig.text(0.1, 0.035, f'30 / 60 min · participants {counts['participant_count']} · participant-calendar days {counts['patient_days']} · base events {counts['base_event_count']} · CGM ≥54 mg/dL.', fontsize=5.5)
    save(fig, out, 'supp1-corrected-candidate', panels, receipt)

def supplement3(data, out, receipt):
    fig, axs = plt.subplots(4, 2, figsize=(180 * G.MM, 182 * G.MM), gridspec_kw={'width_ratios': [1.04, 0.96]})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.09, hspace=1.02, wspace=0.35)
    fig.legend(handles=[Line2D([], [], marker='o', color=G.INK, linestyle='none', markersize=3, markerfacecolor=G.INK if i == 0 else 'white', label=label) for i, label in enumerate(('AUPRC / Brier', 'AUROC / ECE'))], loc='upper center', ncol=2, fontsize=5.5)
    panels = {}
    for row, head in enumerate(HEADS):
        diags = [data['reports'][i]['partitions']['selection']['probability_diagnostics'][head] for i in IDENTITIES]
        ns, prevalences = ({d['n'] for d in diags}, {diagnostic_metrics(d)['prevalence'] for d in diags})
        require(len(ns) == len(prevalences) == 1, 'head n/prevalence varies across fits; a single annotation would hide it')
        n, prevalence = (next(iter(ns)), next(iter(prevalences)))
        head_label = ('Level-2' if head.startswith('hypo') else 'Hyperglycaemia') + f', {head.split('_')[1]} min'
        for col, metrics in enumerate((('auprc', 'auroc'), ('brier', 'ece'))):
            ax = axs[row, col]
            rows = sum((probability_rows(data, m, head, series=m) for m in metrics), [])
            point_panel(ax, rows, metrics)
            letter = chr(97 + row * 2 + col)
            prevalence_label = 'undefined' if prevalence is None else f'{prevalence:.3%}'
            title(ax, letter, head_label if col == 0 else f'n={n:,}; prevalence={prevalence_label}')
            panels[letter] = {'quantity': 'temperature-scaled probability diagnostic; native scalar summaries evaluated on selection', 'head': head, 'rows': rows}
    fig.text(0.085, 0.018, 'Calibration-fitted temperatures · Selection diagnostics · Large marks: fit means; small deep-model marks: five seeds.', fontsize=5.5)
    save(fig, out, 'supp3-corrected-candidate', panels, receipt)
