# CGM hypoglycaemia warning

Research code for retrospective, source-separated warning of clinically significant
hypoglycaemia 30 and 60 minutes ahead. The package includes the scoring definition,
trained model families, calibration, score composition, analysis tools, scientific
configuration templates and synthetic tests. It is not a clinical deployment system.

## Environment and synthetic example

Use Python 3.10, 3.11 or 3.12. The reported experiments used Python 3.10.20 and
PyTorch 2.5.1. Install the recorded versions in an isolated environment:

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

Obtain the consolidated CGM release through the access procedures described by
[MetaboNet](https://doi.org/10.1177/19322968261441637) and comply with its
applicable licence terms. The external cohort has its own
access conditions. Keep development and external inputs in separate paths. Never use
external labels for model, checkpoint, threshold or analysis selection.

The `code/configs/*.template.yaml` files preserve scientific parameters, including
source membership, participant split seed and fractions, context lengths, labels,
training settings and evaluation settings. The split implementation is in
`code/src/glycoshift/split.py`; it assigns participants within source before windows.
This release does not expose participant assignments or local input identity bindings.
The corresponding template fields are unset. They are deliberate prerequisites, and
the unchanged validation code rejects these templates for real-data execution until
a licensed, locally validated input configuration is available. No command here
disables that validation or constructs replacement scientific bindings.

Once a valid local configuration and data have been prepared, the scientific entry
points are:

```bash
PYTHONPATH=code/src .venv/bin/python -m glycoshift.prepare --help
PYTHONPATH=code/src .venv/bin/python -m glycoshift.run_seed_suite --help
PYTHONPATH=code/src .venv/bin/python -m glycoshift.evaluate --help
```

This is a release of the scientific implementation and parameter templates, not a
claim that the complete real-data experiment is runnable from the archive alone.
Creating valid local input bindings remains an explicit prerequisite. It requires the
licensed source data and the study's data preparation conditions. The synthetic
example and function-level tests run without those inputs.

## Analyses and figures

`code/scripts/` contains analyses of reader-generated prediction and metric outputs.
For example, `registered_participant_bootstrap.py --help` documents the participant
resampling inputs and `composition_frozen_evaluation.py --help` documents the
calibration-only composition evaluation. Preserve prediction join keys, calibration
roles and the frozen alarm policy when reproducing a comparison.

`manu/figures/make_fig_*.py` reads aggregate analysis outputs and writes PDF figures.
`code/scripts/figures/` contains their plotting helpers and table builders. Each
entry point documents its input arguments with `--help`. The caption inputs under
`manu/sections/` and `manu/supplement.tex` are included because figure geometry uses
the caption associated with each plotted result. The study's generated analysis
outputs are not bundled, so reproducing a result figure first requires generating
the corresponding analysis inputs from licensed data. Figure 1's editable native
PowerPoint is distributed in the manuscript handoff's editable-figure folder.

Install the plotting dependency before using those builders:

```bash
.venv/bin/pip install -r requirements-figures.txt
```

The experiment records do not establish an exact plotting-library version; this
additional requirement is a compatibility range rather than a claimed recorded pin.
The optional gradient-boosting baseline also requires the `baselines` extra declared
in `code/pyproject.toml`; it raises an error when that dependency is unavailable.
The supplied synthetic suite does not establish reproduction of that real-data arm.

## Licence and citation

The software uses the MIT licence. This licence does not relicense the source cohorts.
Use `CITATION.cff` for the software citation and cite the paper and original datasets
when reporting derived results. The source repository is
https://github.com/gxCaesar/cgm-hypoglycaemia-warning.

The accompanying manuscript is *Dominance of the eligibility rule over model choice
in warning of clinically significant hypoglycaemia*, by Delong Yang, Nan Li, Yu Fu,
Yideng Jiang and Guanxing Chen. The manuscript has not yet been published. Software
implementation and computational analysis are attributed to Guanxing Chen; the
paper provides the full author contributions.

## Funding

This work was supported by the National Science and Technology Major Project of the
Ministry of Science and Technology of China (No. 2024ZD0531200) and the National
Natural Science Foundation of China (Nos. 82302842 and 82560451).
