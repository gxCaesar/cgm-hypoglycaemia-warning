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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'code/scripts/figures'))
import glycofig as G
COHORTS = {'stress_hupa_ucm': 'HUPA-UCM', 'external_loop': 'Loop Observational Study'}
LOOP_INVENTORY = {'full_record_count': 843, 'full_group_count': 842, 'verified_pair_record_count': 2, 'record_point_weights': 'original_distinct_record_weights'}
SEEDS, HORIZONS, CUTS, LEADS = ((17, 23, 42, 71, 101), (30, 60), (54, 70), (10, 15, 20))
FITS = (('persistence', 17), *(('lstm', seed) for seed in SEEDS))
EXCLUDED = 'exclude_verified_pair_records'
ANALYSES = (('stress_hupa_ucm', 'primary'), ('external_loop', 'primary'), ('external_loop', EXCLUDED))
FILES = ('external-figure-inputs.json', 'policy-rows.csv', 'joint-fixed-match-coverage.json', 'joint-fixed-match-coverage.csv', 'export-note.json', 'status.json')
METRICS = {'fixed_sensitivity': 'fixed_sensitivity', 'unmatched_alarms_per_patient_day': 'participant_macro_unmatched_alarms_per_patient_day', 'delivered_alarms_per_patient_day': 'participant_macro_delivered_alarms_per_patient_day'}
ESTIMANDS = ('equal_source_participant_macro', 'participant_pooled', 'event_pooled')
POPULATION = ('participant_count', 'event_positive_participant_count', 'base_event_count', 'patient_days', 'observation_minutes')
CONTRASTS = {}
for _cut in CUTS:
    for _arm in ('persistence', 'lstm'):
        CONTRASTS[f'{_arm}_at_{_cut}'] = {(_arm, _cut): 1}
    CONTRASTS[f'lstm_minus_persistence_at_{_cut}'] = {('lstm', _cut): 1, ('persistence', _cut): -1}
INTERACTION = 'lstm_minus_persistence_contrast_change_70_minus_54'
CONTRASTS[INTERACTION] = {('lstm', 70): 1, ('persistence', 70): -1, ('lstm', 54): -1, ('persistence', 54): 1}
POLICY_ROWS = [(h, cut, arm) for h in HORIZONS for cut in CUTS for arm in ('persistence', 'lstm')]
PAIR_ROWS = [(name, horizon) for name in ('lstm_minus_persistence_at_54', 'lstm_minus_persistence_at_70', INTERACTION) for horizon in ('30', '60', 'equal_horizon_mean')]
COLOURS = {'persistence': G.NOMINAL['rose'], 'lstm': G.NOMINAL['blue']}
MARKERS = {'persistence': 'o', 'lstm': 'v'}
CAPTION = "Frozen-policy replay in previously examined HUPA-UCM and Loop cohorts. a,b Participant-macro fixed sensitivity and c,d achieved unmatched alarms per patient-day at 30/60 minutes and gates 54/70 mg/dL. Large points and whiskers are native estimates and 95% paired percentile intervals. LSTM estimates average five training seeds; current glucose uses one shared fit. e,f Direct paired LSTM-minus-current-glucose sensitivity differences at each gate and their 70-minus-54 change, shown for each horizon and their equal mean. Loop diamonds exclude both verified overlapping records without refitting and use that analysis's own intervals. Loop primary intervals resample correlated-record groups; excluded-record and HUPA intervals resample participants. Native sparse or undefined intervals remain absent. g,h Descriptive fixed-match coverage with alarm glucose at least 70 mg/dL and minimum leads of 10/15/20 minutes, retaining all base events. LSTM cells average five fit-specific participant-macro rates. These cells have no CI. All policies use the nominal development-calibration budget of 0.5 false alarms per patient-day. Achieved burden can differ; neither equal achieved burden nor untouched confirmation is claimed."

def require(condition, message):
    if not condition:
        raise ValueError(message)

def number(value, label, *, null=False):
    if value is None:
        require(null, f'{label}: undefined value')
        return None
    require(type(value) in (int, float) and math.isfinite(value), f'{label}: non-finite or invalid number')
    return float(value)

def near(actual, expected, label):
    value = number(actual, label, null=True)
    require(value is None and expected is None or (value is not None and expected is not None and math.isclose(value, expected, rel_tol=1e-09, abs_tol=1e-10)), f'{label}: aggregate arithmetic mismatch or replaced undefined value')

def mean(values):
    return None if any((v is None for v in values)) else math.fsum(values) / len(values)

def index_rows(rows, keys, expected, label):
    require(len(rows) == len(expected), f'{label}: incomplete row coverage')
    result = {}
    for row in rows:
        key = tuple((row[field] for field in keys))
        require(key in expected and key not in result, f'{label}: duplicate or unexpected condition')
        result[key] = row
    require(set(result) == expected, f'{label}: missing seed/cut/horizon')
    return result

def validate_interval(value, expected, payload, population, horizon, terms, metric):
    near(value['estimate'], expected, 'native inference point')
    low, high = (value['ci_low'], value['ci_high'])
    require((low is None) == (high is None), 'native CI requires both endpoints')
    undefined_draws = value.get('undefined_replicate_count', 0)
    require(type(undefined_draws) is int and 0 <= undefined_draws <= 1000, 'invalid undefined-draw count')
    sensitivity = metric == 'fixed_sensitivity'
    support = [p.get('event_positive_group_count', p['event_positive_participant_count']) if payload['resampling_unit'] == 'source_scoped_correlated_participant_group' else p['event_positive_participant_count'] for p in population]
    sparse = sensitivity and (min(support) < 2 if horizon is None else support[HORIZONS.index(horizon)] < 2)
    undefined_base = payload['interval_status'] == 'undefined_base_sensitivity'
    if low is not None:
        lo, hi = (number(low, 'CI low'), number(high, 'CI high'))
        require(expected is not None and lo <= hi and (not undefined_draws) and (not sparse) and (not undefined_base), 'CI added to sparse/undefined result or reversed')
        if sensitivity:
            require(sum((min(v, 0) for v in terms.values())) - 1e-10 <= lo <= hi <= sum((max(v, 0) for v in terms.values())) + 1e-10, 'sensitivity CI has wrong units or impossible endpoints')
    elif expected is not None:
        require(sparse or undefined_base or undefined_draws > 0, 'supported native interval is missing without a native reason')

def validate_analysis(native, partition, analysis, policies, counts):
    population = [counts[partition, analysis, h] for h in HORIZONS]
    require(native['counts'] == [{k: v for k, v in row.items() if k not in ('partition', 'analysis')} for row in population], 'native horizon population differs')
    require(set(native['metrics']) == set(METRICS), 'native metric set differs')
    native_rows = [{k: v for k, v in p.items() if k not in ('partition', 'analysis')} for key, p in policies.items() if key[:2] == (partition, analysis)]
    expected_native = {tuple((p[k] for k in ('arm', 'training_seed', 'glucose_cut', 'horizon_minutes'))): p for p in native_rows}
    actual_native = index_rows(native['descriptive_policies'], ('arm', 'training_seed', 'glucose_cut', 'horizon_minutes'), set(expected_native), 'native descriptive policies')
    require(actual_native == expected_native, 'native and exported policy points differ')
    grouped = partition == 'external_loop' and analysis == 'primary'
    group_mapping = partition == 'external_loop'
    for metric, payload in native['metrics'].items():
        require(payload['model_seeds'] == list(SEEDS) and payload['horizons'] == list(HORIZONS) and (payload['source_order'] == [COHORTS[partition]]) and (payload['bootstrap_seed'] == 20260810) and (payload['aggregation'] == 'equal_source_then_equal_horizon_and_model_seed') and (set(payload['contrasts']) == set(CONTRASTS)), 'native inference scope or seven-contrast coverage differs')
        unit = 'source_scoped_correlated_participant_group' if grouped else 'source_scoped_participant'
        require(payload['resampling_unit'] == unit, 'wrong native resampling unit')
        support = [p.get('event_positive_group_count') if grouped else p['event_positive_participant_count'] for p in population]
        undefined = metric == 'fixed_sensitivity' and (not all(support))
        sparse = metric == 'fixed_sensitivity' and all(support) and (min(support) < 2)
        expected_status = 'undefined_base_sensitivity' if undefined else ('descriptive_sparse_event_positive_groups' if group_mapping else 'descriptive_sparse_event_positive_participants') if sparse else 'group_percentile_bootstrap' if group_mapping else 'participant_percentile_bootstrap'
        allowed = {expected_status}
        if metric != 'fixed_sensitivity':
            allowed.add('undefined_horizon_population_draws')
        require(payload['interval_status'] in allowed and payload['replicates'] == (0 if undefined else 1000), 'native sparse/undefined interval status differs')
        for name, terms in CONTRASTS.items():
            result = payload['contrasts'][name]
            require(result['coefficients'] == [{'arm': arm, 'glucose_cut': cut, 'coefficient': coefficient} for (arm, cut), coefficient in terms.items()] and set(result['per_horizon']) == {'30', '60'} and (set(result['per_model_seed']) == {str(seed) for seed in SEEDS}), 'paired coefficients or horizon/seed coverage differs')
            points = []
            for horizon in HORIZONS:
                seed_points = []
                for seed in SEEDS:
                    values = [policies[partition, analysis, arm, 17 if arm == 'persistence' else seed, cut, horizon][METRICS[metric]] for arm, cut in terms]
                    seed_points.append(None if any((v is None for v in values)) else math.fsum((v * c for v, c in zip(values, terms.values(), strict=True))))
                points.append(seed_points)
                validate_interval(result['per_horizon'][str(horizon)], mean(seed_points), payload, population, horizon, terms, metric)
            validate_interval(result, mean([v for row in points for v in row]), payload, population, None, terms, metric)
            for index, seed in enumerate(SEEDS):
                near(result['per_model_seed'][str(seed)], mean([points[0][index], points[1][index]]), 'native seed point')

def analysis_report(data, cohort, analysis):
    native = data['native'][cohort]
    return native if analysis == 'primary' else native['named_sensitivities'][analysis]

def point(data, cohort, analysis, metric, contrast, horizon):
    payload = analysis_report(data, cohort, analysis)['metrics'][metric]
    result = payload['contrasts'][contrast]
    value = result if horizon == 'equal_horizon_mean' else result['per_horizon'][str(horizon)]
    status = 'undefined' if value['estimate'] is None else 'native_ci' if value['ci_low'] is not None else 'sparse_no_ci' if payload['interval_status'].startswith('descriptive_sparse') else 'native_ci_unavailable'
    return {'cohort': cohort, 'analysis': analysis, 'metric': metric, 'contrast': contrast, 'horizon': str(horizon), 'estimate': value['estimate'], 'ci_low': value['ci_low'], 'ci_high': value['ci_high'], 'display_status': status, 'interval_status': payload['interval_status'], 'undefined_replicate_count': value.get('undefined_replicate_count', 0), 'resampling_unit': payload['resampling_unit'], 'coefficients': result['coefficients'], 'source_pointer': f'/native_inference/{cohort}/' + (f'named_sensitivities/{EXCLUDED}/' if analysis != 'primary' else '') + f'metrics/{metric}/contrasts/{contrast}' + ('' if horizon == 'equal_horizon_mean' else f'/per_horizon/{horizon}')}

def mark(ax, record, y, *, scale=1, colour=G.INK, marker='o', opened=False):
    value = record['estimate']
    if value is None:
        ax.text(0.5, y, 'Undefined', transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=5.0, color=colour)
        return
    if record['ci_low'] is not None:
        lo, hi = (record['ci_low'] * scale, record['ci_high'] * scale)
        ax.hlines(y, lo, hi, color=colour, linewidth=0.65)
        ax.vlines([lo, hi], y - 0.07, y + 0.07, color=colour, linewidth=0.5)
    ax.plot(value * scale, y, marker=marker, linestyle='none', markersize=2.6, markeredgewidth=0.6, markeredgecolor=colour, markerfacecolor='white' if opened or record['ci_low'] is None else colour)
    if record['ci_low'] is None:
        text = 'Sparse; no CI' if record['display_status'] == 'sparse_no_ci' else 'No CI'
        ax.text(0.98, y + 0.18, text, transform=ax.get_yaxis_transform(), ha='right', va='top', fontsize=5.0, color=G.MUTED)

def make_figure(data, *, synthetic=False):
    G.set_style()
    fig = plt.figure(figsize=(180 * G.MM, 185 * G.MM))
    axes, panels = ([], {})
    positions = [(0.765, 0.17), (0.535, 0.17), (0.295, 0.18), (0.095, 0.14)]
    row_labels = [f'{h}/{cut}  {('Current glucose' if arm == 'persistence' else 'LSTM')}' for h, cut, arm in POLICY_ROWS]
    contrast_labels = {'lstm_minus_persistence_at_54': '54 gate', 'lstm_minus_persistence_at_70': '70 gate', INTERACTION: '70−54 change'}
    pair_labels = [f'{contrast_labels[name]} · {(h + ' min' if h != 'equal_horizon_mean' else 'Mean')}' for name, h in PAIR_ROWS]
    for col, cohort in enumerate(COHORTS):
        fig.text((0.355, 0.775)[col], 0.983, 'HUPA-UCM' if col == 0 else 'Loop', fontsize=7, fontweight='bold', ha='center', va='top')
        population = [data['counts'][cohort, 'primary', h] for h in HORIZONS]
        summaries = [f'{h}: N={p['participant_count']}, n+={p['event_positive_participant_count']}, E={p['base_event_count']}' + (f', G={p['group_count']}' if cohort == 'external_loop' else '') for h, p in zip(HORIZONS, population, strict=True)]
        fig.text((0.18, 0.6)[col], 0.966, ' | '.join(summaries), fontsize=5.0, va='top')
    for row, (bottom, height) in enumerate(positions):
        for col, cohort in enumerate(COHORTS):
            ax = fig.add_axes([(0.18, 0.6)[col], bottom, 0.35, height])
            axes.append(ax)
            letter = chr(97 + row * 2 + col)
            G.panel_label(ax, letter, dx=-0.09, dy=1.1)
            title = ('Fixed sensitivity', 'Achieved unmatched burden', 'Paired sensitivity contrasts', 'Joint glucose / lead coverage')[row]
            ax.set_title(title, loc='left', fontsize=6.2, pad=6)
            records = []
            if row in (0, 1):
                metric = 'fixed_sensitivity' if row == 0 else 'unmatched_alarms_per_patient_day'
                for y, (h, cut, arm) in enumerate(POLICY_ROWS):
                    record = point(data, cohort, 'primary', metric, f'{arm}_at_{cut}', h)
                    record['fit_points'] = [{'training_seed': seed, 'value': data['policies'][cohort, 'primary', arm, seed, cut, h][METRICS[metric]]} for seed in ((17,) if arm == 'persistence' else SEEDS)]
                    mark(ax, record, y, colour=COLOURS[arm], marker=MARKERS[arm])
                    records.append(record)
                ax.set_yticks(range(8), row_labels if col == 0 else [''] * 8)
                ax.set_ylim(7.65, -0.65)
                if row == 0:
                    ax.set_xlim(-0.03, 1.03)
                    ax.set_xticks([0, 0.5, 1])
                else:
                    ax.set_xlabel('Unmatched alarms per patient-day', fontsize=5.5)
            elif row == 2:
                analyses = ('primary', EXCLUDED) if cohort == 'external_loop' else ('primary',)
                for y, (contrast, horizon) in enumerate(PAIR_ROWS):
                    for analysis in analyses:
                        record = point(data, cohort, analysis, 'fixed_sensitivity', contrast, horizon)
                        offset = (-0.19 if analysis == 'primary' else 0.19) if len(analyses) == 2 else 0
                        mark(ax, record, y + offset, scale=100, colour=G.NOMINAL['blue'] if analysis == 'primary' else G.NOMINAL['amber'], marker='o' if analysis == 'primary' else 'D', opened=analysis != 'primary')
                        records.append(record)
                ax.set_yticks(range(9), pair_labels if col == 0 else [''] * 9)
                ax.set_ylim(8.7, -0.7)
                ax.axvline(0, color=G.MUTED, linewidth=0.5, linestyle=':')
                ax.set_xlabel('Difference (percentage points)', fontsize=5.5)
            else:
                matrix = np.empty((8, 3))
                for y, (h, cut, arm) in enumerate(POLICY_ROWS):
                    for x, lead in enumerate(LEADS):
                        values = [data['coverage'][cohort, 'primary', arm, seed, cut, h, lead] for seed in ((17,) if arm == 'persistence' else SEEDS)]
                        estimate = mean([v['equal_source_participant_macro']['estimate'] for v in values])
                        matrix[y, x] = np.nan if estimate is None else estimate
                        records.append({'cohort': cohort, 'analysis': 'primary', 'arm': arm, 'horizon_minutes': h, 'glucose_cut': cut, 'minimum_lead_minutes': lead, 'estimate': estimate, 'display_percent': None if estimate is None else estimate * 100, 'fit_values': values, 'uncertainty': 'descriptive_no_ci'})
                cmap = LinearSegmentedColormap.from_list('glyco_coverage', ['white', G.NOMINAL['blue'], G.INK])
                cmap.set_bad('#EEEEEE')
                image = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', vmin=0, vmax=1, cmap=cmap, interpolation='nearest')
                for y in range(8):
                    for x in range(3):
                        value = matrix[y, x]
                        ax.text(x, y, 'Undefined' if not np.isfinite(value) else f'{value * 100:.1f}%', ha='center', va='center', fontsize=5.2, color='white' if np.isfinite(value) and value > 0.6 else G.INK)
                ax.set_yticks(range(8), row_labels if col == 0 else [''] * 8)
                ax.set_xticks(range(3), [str(lead) for lead in LEADS])
                ax.set_xlabel('Minimum lead (min)', fontsize=5.5)
                if col == 1:
                    cax = fig.add_axes([0.967, bottom, 0.008, height])
                    bar = fig.colorbar(image, cax=cax, ticks=[0, 0.5, 1])
                    bar.ax.set_yticklabels(['0', '50', '100'])
                    bar.ax.tick_params(labelsize=5, length=1, pad=1)
            ax.tick_params(axis='y', labelsize=5.3, length=0, pad=3)
            if row < 3:
                ax.spines['left'].set_visible(False)
                ax.grid(axis='y', color='#E8EAED', linewidth=0.35)
                ax.set_axisbelow(True)
                if row > 0:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            panels[letter] = {'cohort': cohort, 'values': records, 'display_scale': 100 if row in (2, 3) else 1}
    for pair, scale in (((axes[2], axes[3]), 1), ((axes[4], axes[5]), 100)):
        records = [r for ax in pair for r in panels[chr(97 + axes.index(ax))]['values']]
        values = [v * scale for r in records for v in (r['estimate'], r['ci_low'], r['ci_high']) if v is not None]
        lo, hi = (min([0, *values]), max([0, *values]))
        padding = max(hi - lo, 0.05 * scale) * 0.08
        for ax in pair:
            ax.set_xlim(lo - padding, hi + padding)
    handles = [Line2D([], [], color=G.NOMINAL['blue'], marker='o', linestyle='none', markersize=3, label='Loop primary (group-resampled)'), Line2D([], [], color=G.NOMINAL['amber'], marker='D', markerfacecolor='white', linestyle='none', markersize=3, label='Loop pair records excluded (own CI)')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.175, 0.02), ncol=2, fontsize=5.2, handletextpad=0.3, columnspacing=1.2)
    fig.text(0.018, 0.041, 'Rows: horizon / gate (mg/dL) · Hollow circles/triangles: no CI · Coverage: descriptive %', fontsize=5.0, color=G.MUTED)
    fig.text(0.018, 0.008, 'SYNTHETIC FIXTURE · layout validation only' if synthetic else 'Previously examined cohorts · Frozen policies · Nominal calibration budget 0.5', fontsize=5.0, color=G.MUTED)
    G.assert_compliant(fig, 'reduced external Figure 8', len(CAPTION.split()))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    overflow = [text.get_text() for text in fig.findobj(matplotlib.text.Text) if text.get_visible() and text.get_text().strip() and (text.get_window_extent(renderer).x0 < -0.5 or text.get_window_extent(renderer).y0 < -0.5 or text.get_window_extent(renderer).x1 > width + 0.5 or (text.get_window_extent(renderer).y1 > height + 0.5))]
    require(not overflow, f'text outside final canvas: {overflow}')
    receipt = {'status': 'candidate_external_aggregate_figure', 'scientific_acceptance': False, 'synthetic_fixture': synthetic, 'width_mm': 180, 'height_mm': 185, 'panel_labels': [f'{chr(97 + i)})' for i in range(8)], 'panels': panels, 'horizon_populations': list(data['counts'].values()), 'supporting_policies': list(data['policies'].values()), 'supporting_joint_coverage': list(data['coverage'].values()), 'native_inference': data['native'], 'text_overflow_count': 0, 'caption_draft': CAPTION}
    return (fig, receipt)
