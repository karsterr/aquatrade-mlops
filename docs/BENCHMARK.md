# Reproducible forecast comparison

This illustrative run uses a named complete series, not an independent accuracy
benchmark or a representative sample of all commodities. It neither validates data
provenance nor establishes future performance. The series was selected after data
inspection; no model hyperparameter search was performed.

```bash
python -m scripts.benchmark --commodity 'Pacific salmon, frozen' \
  --metric export_vlu --horizon 12 --evaluation-months 24 \
  --output /tmp/aquatrade-benchmark.json
```

Use Python 3.12 and the pinned development requirements from the README. The command
transforms the bundled raw CSV, selects World observations, and writes only the
specified report. It does not write the registry, experiments or DuckDB. Missing
values fail explicitly. Other series may not meet the completeness requirement.

The checked-in [example report](benchmark-example.json) records the input SHA-256,
versions, dates, origins, all candidate scores and future predictions. Input:
328 monthly observations, January 1989–April 2016. Two nonoverlapping 12-month blocks
follow the April 2014 and April 2015 observed origins; 24 predictions are scored per
candidate. Scalers and estimators are fitted using only each origin's history.

| Candidate | MAE (value units) | RMSE (value units) | WAPE |
| --- | ---: | ---: | ---: |
| GradientBoosting | 10,355,200 | 16,750,854 | 23.28% |
| RandomForest | 12,076,065 | 19,082,701 | 27.15% |
| SeasonalNaive | 12,310,049 | 20,228,577 | 27.68% |
| RidgeRegression | 14,630,546 | 22,564,588 | 32.90% |
| LinearTrend | 16,498,424 | 23,976,060 | 37.10% |
| MovingAverage | 35,008,816 | 61,450,454 | 78.72% |
| Naive | 36,630,883 | 64,565,531 | 82.37% |

These are rounded presentation values; ranking uses full precision. The dataset's
value-unit label is U.S.$; source valuation methodology remains unknown. Gradient
boosting has lower validation error than these baselines on this specific series and
configuration. This observation cannot support a project-wide superiority claim.
The same two blocks select the winner, so its reported score has selection bias.
No future forecast accuracy or interval coverage has been measured.

For a defensible next benchmark, freeze eligible series and preprocessing rules,
reserve outer test origins before selection, compare every candidate on identical
origins and horizons, report per-series and per-horizon errors, and disclose missing
series and failures. Use disjoint calibration origins before evaluating interval
coverage. Preserve the source hash, dependency constraints and origin metadata.
