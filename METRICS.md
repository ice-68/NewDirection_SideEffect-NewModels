# Stage 2 Evaluation Contract

Frozen: 2026-10-27

This file is the Stage 2 definition gate. Results produced under a different
protocol must be reported as exploratory and cannot replace the frozen results.

## Data And Splits

- Data set: CC200, T=78, 877 subjects, 20 sites.
- Outer validation: leave-one-site-out (LOSO), repeated for seeds 0-4.
- Inner validation: 20% of the training sites only. The held-out site never
  contributes to model fitting, stopping, feature selection, or threshold
  selection.
- B1 primary result is strictly inductive: the held-out site is absent from the
  training and validation graph. Each test subject is scored without edges to
  other held-out test subjects.
- B2 standardisation is fitted on the training sites only.
- The primary analysis uses all 20 sites, including CMU. CMU is also reported
  separately because it has only five subjects.

## Primary Metric

The primary metric is **site-macro AUC**.

For each seed:

1. Compute AUC separately within each held-out site.
2. Average the 20 site AUC values with equal site weight.

Across seeds, report the mean and sample standard deviation (ddof=1) of the
seed-level site-macro AUC. Predictions from different sites are not pooled to
compute the primary AUC.

## Secondary Metrics

The following metrics must be reported for every model:

- PR-AUC, using average precision
- balanced accuracy
- sensitivity
- specificity
- accuracy
- positive-class F1
- Cohen's kappa

Binary labels are assigned with a fixed threshold of 0.5. Thresholds are not
tuned on the held-out site.

Every site, seed, and model combination must retain its predictions and
confusion-matrix counts. Reports must include each fold/seed result and the
mean +/- SD across seeds.

## Reporting Rules

- Do not report a best fold or best seed as the main result.
- Do not report isolated p-values. Later inferential analyses must include an
  effect size, 95% confidence interval, and a stated multiple-comparison
  correction unit.
- The initial Stage 2 comparison is descriptive. Uncertainty intervals will be
  added in the subsequent site-level evaluation step.
- Any change to this contract requires a new dated section, a new experiment
  version, and a full rerun of the affected models.

## Stage 3 Uncertainty Addendum

Defined: 2026-11-13

This addendum applies only to post-processing of the frozen Stage 2 outputs.
The primary metric definition and the formal model results are unchanged.

- Uncertainty target: the equal-site-weighted mean of seed-level site-macro
  AUC across seeds 0-4.
- Resampling unit: held-out site. Each of the 20 sites is sampled with
  replacement to form one bootstrap data set.
- Seed preservation: when a site is selected, all five seed-specific AUC values
  for that site are selected together. This preserves the repeated-measures
  structure across seeds.
- Replicate statistic: arithmetic mean of all selected site-by-seed AUC values.
  Equivalently, compute the site-macro AUC for each seed and average over the
  five seeds.
- Number of replicates: 10,000.
- Bootstrap seed: 20261113.
- Interval: percentile interval using the empirical 2.5th and 97.5th
  percentiles of the replicate statistics.
- Complete site-by-seed matrices are required. Duplicate cells, missing cells,
  or non-finite AUC values invalidate the analysis.

The interval captures between-site heterogeneity and repeated-seed variation.
It does not re-estimate models, resample subjects within sites, or represent
uncertainty for a new subject from an already sampled site.

The companion site-by-model matrix reports the mean and sample standard
deviation of AUC across seeds for every site and model. These summaries are
descriptive and do not replace the frozen primary metric.

## Required Outputs

Each experiment directory must contain:

- `predictions.csv`
- `fold_selection.csv`
- `training_history.csv` when applicable
- `site_seed_metrics.csv`
- `seed_metrics.csv`
- `site_summary.csv`
- `summary.json`
- `report.md`

`summary.json` must record seeds, protocol, software versions, Git commit, and
content hashes for the cache and fold files.
