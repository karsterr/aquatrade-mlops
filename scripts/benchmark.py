"""Reproduce one explicitly selected series comparison without writing pipeline stores.

Run from repository root: python -m scripts.benchmark --output /tmp/benchmark.json
"""

import argparse
import hashlib
import platform
from pathlib import Path

import pandas as pd
import sklearn

from main import positive_int
from src.mlops.storage import write_json
from src.use_cases.etl_pipeline import ETLService
from src.use_cases.forecast_service import ForecastService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-path", type=Path, default=Path("data/raw/Aquaculture_Trade_Filtered.csv")
    )
    parser.add_argument("--commodity", default="Pacific salmon, frozen")
    parser.add_argument(
        "--metric",
        choices=["export_vlu", "import_vlu", "export_qty", "import_qty"],
        default="export_vlu",
    )
    parser.add_argument("--horizon", type=positive_int, default=12)
    parser.add_argument("--evaluation-months", type=positive_int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    facts = ETLService().transform(pd.read_csv(args.raw_path))
    selected = facts[(facts.COMMODITY_DESC == args.commodity) & (facts.GEOGRAPHY_DESC == "World")]
    series = selected[["date", args.metric]].rename(columns={args.metric: "value"})
    result, _ = ForecastService(test_split_months=args.evaluation_months).run_tournament(
        series, f"{args.commodity}__{args.metric}__World", args.commodity, args.metric, args.horizon
    )
    payload = {
        "purpose": "illustrative selection-validation comparison, not independent test accuracy",
        "dataset_sha256": hashlib.sha256(args.raw_path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "observations": len(series),
        "start": series.date.min().isoformat(),
        "end": series.date.max().isoformat(),
        "result": result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print(f"Wrote {args.output}; champion: {result.champion_model}")


if __name__ == "__main__":
    main()
