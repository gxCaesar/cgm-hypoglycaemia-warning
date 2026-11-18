# CGM hypoglycaemia warning

Scientific implementation for *Glucose eligibility shapes comparisons of hypoglycaemia
warning models across cohorts*. The study compares six warning methods at 30 and 60
minutes across eleven development sources. The primary contrast is the change from
54 to 70 mg/dL in the LSTM-minus-current-glucose sensitivity difference. The software
also supports the fixed outer holdout, event-definition and source-influence analyses,
and development-fixed LSTM/control evaluation in Loop and HUPA-UCM.

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

The companion Supplementary Data supplies the numerical results and aggregate
plotting inputs for this manuscript. It is distributed under CC BY 4.0. Earlier
repository releases and archived data are separate versions and do not contain
these revised results. Clinical records, individual trajectories and fitted weights
are excluded and retain their original access conditions.

## Licensed-data preparation and analysis

Configuration templates preserve scientific settings while leaving local licensed-input
identities unset. Real-data validation remains fail closed until those inputs are
available. Participant splits are assigned within source before overlapping windows.
No supplied command constructs substitute identities or disables data validation.

With a valid local configuration, prepare and fit the original participant split:

```bash
PYTHONPATH=code/src python -m glycoshift.revision_prepare --config local.yaml --normalized data/development-normalized --legacy-windows data/participant-window-shards --output outputs/replay
PYTHONPATH=code/src python -m glycoshift.revision_runner --config local.yaml --prepared outputs/replay --architecture lstm --seed 17 --output outputs/lstm-17
PYTHONPATH=code/src python -m glycoshift.revision_baselines --config local.yaml --prepared outputs/replay --baseline persistence --seed 17 --output outputs/persistence
```

Repeat neural fitting at seeds 17, 23, 42, 71 and 101 for LSTM and temporal convolutional
network configurations. The four classical arms are `persistence`, `linear`,
`summary_logistic` and `xgboost`. Their feature, target-mask and training rules are
implemented in `revision_baselines`. XGBoost requires its pinned optional dependency
in `code/pyproject.toml`. The fixed holdout uses outer fold 0, LSTM seed 17 and its
shared persistence comparator. Its points are descriptive and have no generalization CI.

The analysis modules expose input schemas in their module documentation and `--help`:

| Module | Output and scope |
| --- | --- |
| `revision_analysis` | Six-cut, two-horizon fixed-frame replay and the five-seed paired primary contrast. |
| `revision_source_influence` | Descriptive source interactions, delete-one-source reaggregation, named weighting schemes and joint fixed-match coverage. |
| `revision_round2_sensitivity_runner` | Observed-only and elapsed-span catalogue comparisons with their own event denominators and completeness frames. |
| `revision_holdout_summary` | The fixed fold-0 two-method holdout summary without a generalization interval. |
| `revision_external` | Saved-model evaluation using development-fixed policies without external fitting. |
| `revision_external_inference` | Native paired inference with shared resamples across methods, cuts, horizons and seeds. |

Loop primary inference retains correlated-record groups and the original record-level
point weights. Excluding both verified overlapping records is a separately named
sensitivity analysis. External cohorts were previously examined. Their results are
conditional validation. Nominal calibration burden does not imply equal achieved burden.
Undefined source terms remain undefined, including the full eleven-source observed-only
sensitivity. Model seeds describe optimization variation rather than independent participants.

These interfaces require reader-prepared scientific inputs and their complete local
bindings. The archive does not by itself reproduce the private-data training runs.
Synthetic examples test software behavior and do not establish scientific performance.

## Recreate the result figures

Install `requirements-figures.txt` in the same environment. Extract the companion
Supplementary Data separately and run from this software directory:

```bash
python manu/figures/build_reduced_figures.py --plot-inputs /path/to/supplementary-data/data/plot-inputs --output-dir /path/to/new-figures
```

The command recreates Figures 3–8 and Supplementary Figures 1–5 using the original
rendering functions and supplied aggregate arrays, including native intervals and
explicitly undefined values. It writes PDF figures and plotted values into a new
directory. It performs no fitting, event matching or resampling. Historical score
composition is outside this manuscript's display workflow.

Figure 1 is supplied as editable PowerPoint in the figure-source handoff. Figure 2's
Python source and aggregate cohort counts are included. Its individual illustrative
traces are not redistributed, so recreating that entire figure requires licensed
source data and the documented trace-extraction procedure. The supplied PDF remains
the manuscript's figure. Aggregate cohort profiles alone cannot recreate its trace panels.

## Licence and citation

The software uses the MIT licence. Aggregate Supplementary Data uses CC BY 4.0.
Neither licence changes the source cohorts' terms. Use `CITATION.cff`, the associated
paper and the original dataset citations when reporting derived results. The source
repository is https://github.com/gxCaesar/cgm-hypoglycaemia-warning.
