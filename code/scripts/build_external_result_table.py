"""Render the external table from the two native paired-inference reports."""
import argparse
import json
from pathlib import Path
import statistics


def table(hupa, loop):
    lines = [r'\begin{tabular}{llccc}', r'\toprule',
             r'Cohort & Policy & Horizon (min) & Sensitivity (95\% CI) & Unmatched/day (95\% CI) \\', r'\midrule']
    for cohort, report in [('Loop', loop), ('HUPA-UCM', hupa)]:
        assert report['status'] == 'complete' and report['external_already_exposed'] is True
        for arm, label in [('persistence', 'Current glucose'), ('lstm', 'LSTM'), ('composition', 'Rank composition')]:
            for horizon in ('30', '60', 'Mean'):
                cells = []
                for metric in ('fixed_sensitivity', 'unmatched_alarms_per_patient_day'):
                    point = report['metrics'][metric]['contrasts'][arm + '_at_54']
                    if horizon != 'Mean':
                        point = point['per_horizon'][horizon]
                    cells.append(f"{point['estimate']:.3f} ({point['ci_low']:.3f}--{point['ci_high']:.3f})")
                lines.append(' & '.join([cohort, label, horizon, *cells]) + r' \\')
        if cohort == 'Loop':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def timing_table(hupa, loop):
    lines = [r'\begin{table}[!htbp]', r'\centering', r'\small',
             r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{llccccc}', r'\toprule',
             r'Cohort & Policy & Horizon & PPV & Coverage $\geq$10 & Coverage $\geq$20 & Median lead \\',
             r' & & (min) & & (min) & (min) & (min) \\', r'\midrule']
    fields = ('ppv', 'coverage_10_event_fraction', 'coverage_20_event_fraction', 'matched_only_lead_minutes_median')
    for cohort, report in [('Loop', loop), ('HUPA-UCM', hupa)]:
        for arm, label in [('persistence', 'Current glucose'), ('lstm', 'LSTM'), ('composition', 'Rank composition')]:
            for horizon in (30, 60):
                rows = [r for r in report['descriptive_policies'] if r['arm'] == arm and r['horizon_minutes'] == horizon]
                assert len(rows) == (1 if arm == 'persistence' else 5)
                values = [statistics.mean(r[k] for r in rows) for k in fields]
                lines.append(' & '.join([cohort, label, str(horizon), *[f'{v:.3f}' for v in values[:3]], f'{values[3]:.1f}']) + r' \\')
        if cohort == 'Loop':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}',
        r'\caption{Descriptive external warning timing and positive predictive value under frozen development policies. Alarm positive predictive value (PPV) is the fraction of evaluable delivered alarms matched one-to-one to a strictly future base event. Coverage is the fraction of all base events detected at least ten or twenty minutes before onset, including missed events in the denominator. Lead time is calculated only among matched events. Long short-term memory (LSTM) and composition entries average the five fit-specific ratios or medians, whereas the current-glucose control contributes one value. A mean of medians is not the median of pooled alarms. These event-pooled coverage ratios differ from the participant-mean sensitivities in the main table. The base catalogues contain 58,596/59,400 events in Loop and 265/266 in HUPA-UCM at 30/60 minutes. No timing confidence interval or comparison restricted to commonly detected events is implied. All policies use glucose eligibility of at least 54~mg/dL and the nominal development calibration budget of 0.5 unmatched alarms per participant-calendar day.}',
        r'\label{tab:external_timing}', r'\end{table}']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hupa', required=True, type=Path)
    parser.add_argument('--loop', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timing-output', type=Path)
    args = parser.parse_args()
    hupa, loop = json.loads(args.hupa.read_text()), json.loads(args.loop.read_text())
    args.output.write_text(table(hupa, loop))
    if args.timing_output:
        args.timing_output.write_text(timing_table(hupa, loop))
