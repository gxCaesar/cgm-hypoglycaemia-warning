# CGM hypoglycaemia warning

Research code for retrospective, source-separated warning of clinically significant
hypoglycaemia 30 and 60 minutes ahead. The package includes the scoring definition,
trained model families, calibration, score composition, analysis tools, scientific
configuration templates and synthetic tests. It is not a clinical deployment system.

## Environment and synthetic example

Use Python 3.10, 3.11 or 3.12. The reported experiments used Python 3.10.20 and
PyTorch 2.5.1 with a CUDA 12.1 build. The requirements below pin the recorded
numerical package versions; select the corresponding PyTorch build for GPU
reproduction. `code/env.yml` instead provides broader development ranges.
Install the recorded versions in an isolated environment:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r code/requirements-lock.txt
.venv/bin/pip install --no-deps -e ./code
.venv/bin/python -m pytest -c pytest.ini -q
PYTHONPATH=code/src .venv/bin/python -m glycoshift.train --config code/configs/smoke.yaml
```

The synthetic command writes a small demonstration run to the output directory in
`smoke.yaml`. Synthetic metrics test software behavior and are not study results.
No real data, participant-level outputs, model weights or trained checkpoints are included.

## Data access and scientific settings

The main comparisons use sensor glucose, its observation mask and sine/cosine
encodings of the recorded clock. Action records and source-availability features
are excluded. Feature scaling uses training participants only. Prediction anchors
must be observed and uncensored; retrospective outcome completeness does not
advance or reset the online alarm state.

The warning target is CGM-defined level 2 hypoglycaemia, using three consecutive
five-minute readings below 54 mg/dL. It does not identify level 3 events requiring
assistance. Neural targets, loss masks, class weights, positive-example sampling,
checkpoint selection and calibration use the same outcome. Current-glucose and
linear controls retain their training-fitted scalar risk mappings. The main
eligibility comparison changes current glucose from at least 54 to at least
70 mg/dL while retaining the base event catalogue and participant exposure.

Obtain the consolidated CGM release through the access procedures described by
[MetaboNet](https://doi.org/10.1177/19322968261441637) and comply with its
applicable licence terms. The external cohort has its own
access conditions. Keep development and external inputs in separate paths. Never use
external labels for model, checkpoint, threshold or analysis selection.

Aggregate numerical results and plot inputs are provided in the companion
[Supplementary Data](https://github.com/gxCaesar/cgm-hypoglycaemia-warning/releases/download/v0.3.0/cgm-hypoglycaemia-warning-supporting-data.zip) under CC BY 4.0. The underlying clinical data retain
their original access conditions and licence terms. Participant records,
temporal traces and fitted weights are not redistributed.

The `code/configs/*.template.yaml` files preserve scientific parameters, including
source membership, participant split seed and fractions, context lengths, labels,
training settings and evaluation settings. The split implementation is in
`code/src/glycoshift/split.py`; it assigns participants within source before windows.
This release does not expose participant assignments or local input identity bindings.
The corresponding template fields are unset. They are deliberate prerequisites, and
the unchanged validation code rejects these templates for real-data execution until
a licensed, locally validated input configuration is available. No command here
disables that validation or constructs replacement scientific bindings.

Once a valid local configuration and data have been prepared, the matched
level-2 workflow is:

```bash
PYTHONPATH=code/src .venv/bin/python -m glycoshift.revision_prepare \
  --config local.yaml --normalized data/development-normalized \
  --legacy-windows data/participant-window-shards --output outputs/replay
PYTHONPATH=code/src .venv/bin/python -m glycoshift.revision_runner \
  --config local.yaml --prepared outputs/replay --architecture lstm \
  --seed 17 --output outputs/lstm-17
PYTHONPATH=code/src .venv/bin/python -m glycoshift.revision_baselines \
  --config local.yaml --prepared outputs/replay --baseline persistence \
  --seed 17 --output outputs/persistence
```

These paths are examples, not supplied data or fitted models. Preparation binds
the original source-scoped participant assignments and retained windows without
resplitting participants. Neural comparisons use all five model seeds
17, 23, 42, 71 and 101; the shared persistence control is fitted once. The same
runner accepts `--architecture tcn` with its corresponding architecture
configuration. Other classical options are `linear`, `summary_logistic` and
`xgboost`. They use all training rows retained by each task's actual mask.

This is a release of the scientific implementation and parameter templates, not a
claim that the complete real-data experiment is runnable from the archive alone.
Creating valid local input bindings remains an explicit prerequisite. It requires the
licensed source data and the study's data preparation conditions. The synthetic
example and function-level tests run without those inputs.

## Analyses and figures

Summarise the development population from the complete `revision_prepare` output
before fitting models:

```bash
PYTHONPATH=code/src .venv/bin/python code/scripts/summarise_revision_population.py \
  --prepared outputs/replay --output outputs/population
PYTHONPATH=code/src .venv/bin/python code/scripts/build_revision_population_tables.py \
  --aggregate outputs/population --output outputs/population-tables
```

The report directory must be absent or empty. The first command reads the prepared
event catalogues, exposure and task masks and writes source/partition/horizon
aggregates:

| Output | Meaning |
| --- | --- |
| `input-counts.csv` | Participants and base-evaluable, observed-anchor and excluded interpolated-anchor decision counts by source and partition. |
| `endpoint-counts.csv` | Base participants, event-positive participants, distinct base events, calendar days, observation minutes and equivalent 24-hour exposure at each horizon. |
| `training-window-counts.csv` | Windows retained by the level 2 supervision mask and their positive/negative labels before neural training subsampling. |
| `population-summary.json` | Definitions, partition totals, endpoint totals and reporting coverage. Totals sum sources rather than averaging their sensitivities. |

Positive windows can precede the same event and are not distinct event counts.
Calendar days and observation-minutes divided by 1,440 are separate exposure
quantities. The report retains participants without events and preserves the
prepared event catalogue and exposure across current-glucose eligibility rules.
It exports no participant identifiers, timestamps or raw CGM values.

Keep the complete prepared bundle, including its participant-timeline manifest.
Without that manifest the first command omits the training-window report. The
second command requires complete timeline counts for all eleven development
sources and all three partitions. It writes `tab_cgm_cohort.tex`, `tab_split.tex`
and `tab_endpoint_population.tex`, together with `cohort-figure-counts.json`.
The JSON supplies per-source participant/calendar-day counts and separate 30- and
60-minute event counts for figure construction; its `severe_hypo_events` field
contains the 60-minute base count. These are aggregate input counts, not fitted
model performance or rendered figures. The generated tables describe the study's
fixed development population, excluding external cohorts.

`revision_analysis` evaluates the fixed event/exposure population across glucose
cuts, with both calibration refitting and a frozen-threshold gate-only analysis.
Its primary contrast requires the complete five-seed LSTM set and one shared
persistence fit. `revision_composition` fits right-continuous calibration ECDFs
for the fixed LSTM–persistence pair and averages their ranks. The combined score
is not an event probability. `revision_event_sensitivity` changes only event
definitions while preserving emitted alarm sequences and base exposure.

`revision_external` and `revision_external_composition` require a development-fixed
recipe and source-access bindings. They apply the saved models, score mappings
and alarm policies without external fitting. The external cohorts were previously
examined, so these results provide conditional validation. The ledger-only
`revision_external_inference` uses shared participant resampling across horizons
and model seeds; it does not read model predictions. These modules document their
configuration schemas in their module docstrings and expose `--help`.

The older entry points under `code/scripts/` implement additional analyses with
their own input and endpoint definitions. They are not interchangeable with the
fixed-frame workflow. Preserve exact decision keys, calibration roles, event
denominators and alarm policies for every comparison.

`manu/figures/make_fig_*.py` reads aggregate analysis outputs and writes PDF figures.
`code/scripts/figures/` contains their plotting helpers and table builders. Each
entry point documents its input arguments with `--help`. The inputs under
`manu/sections/` and `manu/supplement.tex` contain only figure environments and
their captions, which are used to calculate plot geometry. The study's generated analysis
outputs are not bundled, so reproducing a result figure first requires generating
the corresponding analysis inputs from licensed data. Figure 1's editable native
PowerPoint is distributed in the manuscript handoff's editable-figure folder.

Install the plotting dependency before using those builders:

```bash
.venv/bin/pip install -r requirements-figures.txt
```

The experiment records do not establish an exact plotting-library version; this
additional requirement is a compatibility range rather than a claimed recorded pin.
The gradient-boosting comparator used XGBoost 3.2.0. Its optional `baselines`
extra pins that version in `code/pyproject.toml`; the comparator raises an error
when XGBoost is unavailable. Install it before running that arm:

```bash
.venv/bin/pip install xgboost==3.2.0
```

The supplied synthetic suite does not establish reproduction of the real-data arm.

## Licence and citation

The software uses the MIT licence. This licence does not relicense the source cohorts.
Use `CITATION.cff` for the software citation and cite the paper and original datasets
when reporting derived results. The source repository is
https://github.com/gxCaesar/cgm-hypoglycaemia-warning.

The accompanying manuscript is by Delong Yang, Nan Li, Yu Fu, Yideng Jiang and
Guanxing Chen. The manuscript is entitled Glucose eligibility shapes comparisons of hypoglycaemia warning models across cohorts. Software
implementation and computational analysis are attributed to Guanxing Chen; the
paper provides the full author contributions.

## Funding

This work was supported by the National Science and Technology Major Project of the
Ministry of Science and Technology of China (No. 2024ZD0531200) and the National
Natural Science Foundation of China (Nos. 82302842 and 82560451).

## Recreate the supplied result figures

Extract the companion Supplementary Data archive separately. Its
`data/plot-inputs/` directory contains numerical inputs for the original rendering
functions, including reported intervals and explicitly undefined quantities.
These inputs contain no participant records or private acceptance documents.
With the plotting requirements installed, run from the extracted software root:

```bash
python3 manu/figures/build_corrected_figures.py --plot-inputs /path/to/supplementary-data/data/plot-inputs --output-dir /path/to/new-figure-directory
```

This recreates Figures 3-8 and Supplementary Figures 1-4 with the supplied geometry.
Figures 1 and 2 retain their supplied source/export route. The output directory
must be new. Rendering does not refit a model, rematch events, recompute intervals
or establish independent real-data reproduction.
