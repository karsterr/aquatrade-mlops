# Engineering audit

## A. Executive summary

The layered architecture was retained. The main correction replaces teacher-forced
holdout regression with expanding-window recursive forecasts on identical origins
for regressors and baselines. ETL now retains missing classification keys. Local
persistence, failure visibility, CLI validation, dependency reproducibility and
synthetic regression coverage were strengthened. README claims now match the local
prototype's actual behavior.

## B. Critical findings

1. **Evaluation integrity:** ML holdout features were constructed from the complete
   observed series, exposing prior holdout actuals to later predictions. Baselines
   instead forecast the whole holdout from one origin. The claimed rolling-origin
   comparison was absent. Recursive evaluation now matches future inference.
2. **Silent data loss:** pivoting dropped raw rows with nullable HS codes. There are
   97,992 such raw records; corrected aggregation retains 43,237 resulting facts.
3. **Hidden missingness:** repository interpolation/backfill used later price
   observations, and zero-filling treated absent reporting as observed zeros.
   Missing observations are now preserved and rejected for forecasting.
4. **Misleading uncertainty/metrics:** heuristic RMSE bands were labeled 95% CI;
   epsilon MAPE/WAPE changed metric definitions and R² returned misleading zero for
   undefined cases. Bands are labeled heuristic and metrics have explicit definitions.
5. **Lifecycle reliability:** fixed v1.0.0 overwrote runs; corrupt JSON was silently
   treated as empty; candidate and pipeline exceptions were swallowed. Versions are
   unique, duplicate explicit versions fail, JSON errors propagate, and unexpected
   forecasting failures are visible.
6. **Persistence/security:** forecast writes replaced prior forecasts despite an
   append contract. Metric and table identifier interpolation lacked validation.
   Forecasts now append; metrics are allowlisted, table identifiers validated, and
   partner limits parameterized. Joblib still requires trusted artifacts.
7. **Presentation/reproducibility:** no test workflow, unbounded dependencies,
   unsupported production/DDD/interval/test claims and unresolved data rights.
   No tracked credential files were identified by filename inspection; this is not
   a comprehensive secret-history or dependency-vulnerability audit.

## C. Changes implemented

- `src/use_cases/forecast_service.py`: fresh cloned models per origin, identical full
  horizons, recursive feedback, finite/nonnegative validation, explicit origin
  metadata, seed 42, training-only Ridge scaling, precise scoring and heuristic bands.
- `src/use_cases/etl_pipeline.py`, domain validation/models: retain nullable grouping
  keys and missing sums, reject negative/infinite measurements and nonfinite domain
  values; derived prices no longer conflict with silently clipped base measures.
- Repository and DuckDB client: preserve missing observations, append forecast
  history, validate identifiers and metric selection, atomic single-table replacement.
- Registry/tracker and new `src/mlops/storage.py`: unique versions, collision-resistant
  artifact names, portable new paths, atomic JSON replacement, explicit corruption
  errors, full transformed-content SHA-256, nested-run rejection and initial run
  records. No tracked historic artifacts or the raw data were deleted.
- Orchestration: skip only explicitly insufficient history, propagate unexpected
  failures, persist complete candidate/origin reports, and link reports to active runs.
- Monitoring: constant-baseline PSI can detect shifts; finite filtering and explicit
  insufficient sample handling replace false stability for empty samples.
- `main.py`, `src/config.py`: positive CLI counts and no directory creation on import.
- Tests, pinned runtime/dev requirements and constraints, Ruff config and minimal
  GitHub Actions workflow; repository Python code formatted consistently.
- `README.md`, `docs/DATA.md`, `docs/BENCHMARK.md`, benchmark script and measured
  example report document implementation, limitations and reproducibility.

## D. Validation

The system Python initially had no pytest. An initial exploratory environment used
Python 3.15 and unconstrained dependencies; these are not the supported release
configuration. A separate Python 3.12.15 environment was installed and validated
against pinned dependencies. Do not interpret initial exploratory runs as validation
of the original Git revision: correctness edits had already begun.

Commands used for final validation:

```bash
.venv-audit/bin/python -m pytest -q
.venv-audit/bin/ruff check .
.venv-audit/bin/ruff format --check .
git diff --check
.venv-audit/bin/python -m scripts.benchmark --output docs/benchmark-example.json
```

Final local result: **57 passed, 0 failed, 0 skipped, 0 errors** on Python 3.12.15.
Ruff lint, format checks and `git diff --check` passed.
Remote test CI has not yet been run on this branch at the time of this report.
Tests use synthetic fixtures and temporary artifact storage. A real-data full-pipeline
smoke run used a temporary base directory and did not modify tracked lifecycle stores.
It loaded all 283,178 raw rows and persisted 125,453 facts, then **failed explicitly**
on missing targets in the default top-ranked forecasting series. This is a known
data-readiness limitation, not a passing end-to-end validation. The synthetic
end-to-end test includes four forecasted series and model registrations.

## E. Forecasting evaluation

See `docs/BENCHMARK.md` for the measured single-series example and limitations.
Nonoverlapping full H-step horizons avoid scoring one target repeatedly. New origins
may use observations from prior completed blocks, as expanding-window availability
requires. Within a block, only predictions feed future features. Selection uses MAE,
equivalent to WAPE ranking on a common positive denominator. All-zero actuals still
permit MAE selection. Undefined percentage/R² values remain NaN in memory and null in
JSON. Scores are model-selection validation, not an untouched test set. Historical
tracked scores were produced by the superseded method and should not be compared.

## F. Dataset and licensing

The original 30,292,480-byte CSV is unchanged. Its observed contents, missingness,
hash, dictionary and ETL assumptions are in `docs/DATA.md`. No verified publisher,
original URL, collection methodology or redistribution terms were established.
The owner must resolve these before asserting rights. No software LICENSE exists;
MIT and Apache-2.0 are possible owner choices, not applied licenses.

## G. Remaining technical debt

- **P0 before asserting redistribution rights:** establish original dataset source,
  classification revision, filtering recipe and license. Owner action required.
- **P1 before accuracy claims:** freeze series selection, reserve an outer holdout,
  report horizon-level performance and validate any interval coverage separately.
- **P1 before serving models:** version feature/preprocessing contracts and history
  with artifacts; build a tested reload-to-recursive-forecast interface. Current
  persistence supports model inspection/reload, not a standalone serving deployment.
- **P1 before concurrent or unattended runs:** add locking/transactions or reconciliation
  across artifacts, metadata and indexes. Atomic files do not make a run transactional;
  initial records identify interrupted runs but do not recover them automatically.
- **P1 data semantics:** resolve missing reports versus zeros, duplicated measures,
  aggregate versus partner double-counting and commodity code/description changes.
  Existing full-history top-N and global summaries can include aggregate/partner
  overlap. Forecasts default to World only; summary totals are not global trade estimates.
- **P2 quality/configuration:** invalid dates are dropped; unknown attributes/units
  need a verified data contract. Quality warnings do not halt persistence. Empty raw
  data and partial missing features need richer diagnostics. Several configurable
  table/candidate settings are legacy fields not fully wired into implementations.
- **P2 observability:** persist candidate reports with run lineage into DuckDB, add
  environment/code revision metadata and full-dataset provenance hashes, and replace
  the fixed-2010 pooled PSI comparison with justified feature-specific reference windows.
- **P2 maintenance:** audit dependency advisories separately, automate constraints
  refresh, support additional Python versions only after testing. No new cloud
  infrastructure is justified by the current scope.

## H. Git and metadata

Initial worktree was clean on main at `5c3886b`; origin/main matched after a safe fetch.
Changes are isolated to `improvement/mlops-quality-audit`. Main is not pushed directly;
no history rewriting, artifact deletion or automatic merging is performed.

Prepared repository description:

> Monthly aquaculture trade ETL and recursive forecasting with DuckDB, baseline comparisons, and local model tracking.

Relevant topics: python, machine-learning, mlops, time-series-forecasting, duckdb,
data-engineering, aquaculture, pandas, scikit-learn. The description was applied and machine-learning/mlops topics were added; existing
topics were preserved. Git commit, push, PR and CI outcomes are reported with the
review handoff.

## I. Next cycle

First resolve data rights and reporting semantics. Then freeze a representative
series cohort with independent test origins, package inference contracts/history,
calibrate uncertainty separately, and improve interruption recovery. Keep storage
local until a concrete operational requirement justifies additional infrastructure.
