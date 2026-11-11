# Dataset status and dictionary

The tracked input is `data/raw/Aquaculture_Trade_Filtered.csv`. No source citation,
download URL, publisher statement, collection methodology, or redistribution license
was found in the repository or its dataset history. An exact filename web search
also returned no source. Column labels alone do not establish provenance or rights.
The file has been preserved. The owner must supply the original source, access date,
filtering recipe, classification revision, and explicit redistribution terms before
claiming attribution or a dataset license. Software licensing is separate.

## Verified file contents

Profiled during the engineering audit on 2026-11-12:

- 30,292,480 bytes; 283,178 rows; ten columns.
- SHA-256: `007fd42b240e013280c1a336220f46b9e4ba443482c01fe738be92028b29d7f0`.
- Observation dates: January 1989 through April 2016, 328 distinct months.
- 176 geography labels and 130 commodity descriptions. These are observed labels,
  not proof of comprehensive geographic or aquaculture coverage.
- Attribute labels indicate US imports and exports, not all bilateral global trade.
- Units in the file: `KG` and `U.S.$`. Inflation, valuation basis, exchange-rate
  treatment, and whether quantities are net weight are unverified.
- 97,992 missing HS codes; no missing values in the other columns, exact duplicate
  rows, or negative amounts. This does not establish completeness of monthly series.

| Column | Observed meaning / processing |
| --- | --- |
| SOURCE_ID | Source identifier; institutional mapping unknown |
| HS_CODE | Commodity classification label stored as numeric; revision unknown; nullable |
| COMMODITY_DESC | Commodity description used as a series key |
| GEOGRAPHY_CODE | Numeric partner or aggregate identifier; codebook unknown |
| GEOGRAPHY_DESC | Partner/aggregate label, including World; do not sum World with partners |
| ATTRIBUTE_DESC | US Import/Export, QTY/VLU; pivoted to four measures |
| UNIT_DESC | KG or U.S.$; expected quantity/value interpretation |
| YEAR_ID | Observation year |
| TIMEPERIOD_ID | Month number 1–12 |
| AMOUNT | Trade measure; missing remains missing, negative/infinite values rejected |

## Transformation assumptions

ETL constructs month-start dates, drops unparseable dates, and sums records sharing
month, geography, HS code, commodity and attribute. Nullable HS keys are retained.
Repeated measures are summed; genuine duplicates must be investigated upstream.
Unit prices are summed value divided by summed quantity, undefined at zero quantity.
Balances retain the existing convention of treating absent import/export sides as
zero. That convention is unsuitable as proof of complete reporting.

The corrected ETL produces 125,453 facts, including 43,237 with missing HS codes.
Missing measures remain substantial: import quantity 61,905; import value 59,430;
export quantity 49,948; export value 47,351. The existing IQR profiler flags 4,762
unit-price outliers; these are warnings, not verified errors. Its `is_valid` flag
checks emptiness and negative measures, not dataset fitness for forecasting.

Repository series extraction inserts missing calendar months as missing observations.
It does not interpolate prices or equate absent quantities with observed zeros.
Forecasting rejects gaps, missing/nonfinite/negative targets, and duplicate dates.
No implicit imputation uses future observations. Resolve reporting semantics before
adding a causal, origin-specific imputation policy.

CI uses only synthetic fixtures in `tests/conftest.py`; no raw file or credentials
are needed. Historical experiment and model metadata remain in Git for preservation,
but their old scores came from the superseded evaluation procedure and are not valid
benchmarks. Some reference local artifacts that are not tracked.
