# AquaTrade MLOps 🐟-📊

An MLOps pipeline and analytical system designed for processing, modeling, and forecasting aquaculture trade data.

---

## 📌 Project Overview

**AquaTrade MLOps** processes trade statistics (imports, exports, quantities, values) for aquaculture commodities. Built using **Domain-Driven Design (DDD)** principles, the system decouples infrastructure dependencies, application logic, and domain rules to prepare data for downstream machine learning and forecasting models.

---

## 🏗️ Architecture & Project Structure

The project follows a clean, layered DDD architecture:

```text
aquatrade-mlops/
├── data/
│   ├── raw/                 # Raw datasets (e.g., Aquaculture_Trade_Filtered.csv)
│   └── processed/           # Processed database storage (aquaculture.duckdb)
├── src/
│   ├── application/         # Core application logic and ETL pipelines
│   │   ├── etl_service.py       # Data cleaning, pivoting, metric calculations
│   │   ├── data_inspector.py    # Data profiling & inspection utility
│   │   └── forecast_service.py  # (Roadmap) Time-series forecasting services
│   ├── domain/              # Business domain entities & models
│   │   └── models.py            # Domain models (placeholder)
│   └── infrastructure/      # Technical infrastructure & storage drivers
│       ├── data_loaders/
│       │   └── csv_loader.py    # CSV file loading driver
│       └── db/
│           └── duckdb_client.py # DuckDB connection & query manager
├── main.py                  # Pipeline execution entrypoint
└── README.md                # Project documentation
```

---

## 🚀 Current Features & Pipeline Workflow

1. **Data Ingestion**: `CSVDataLoader` loads raw CSV files from `data/raw/`.
2. **ETL & Transformation**:
   - Formats `YEAR_ID` and `TIMEPERIOD_ID` into standardized datetime objects (`date`).
   - Pivots metric attributes (`US Import QTY/VLU`, `US Export QTY/VLU`).
   - Computes calculated trade metrics (`import_unit_price`, `export_unit_price`).
3. **Data Inspection**: `DataInspector` profiles DataFrame dimensions, data types, and missing values.
4. **Analytical Storage**: `DuckDBClient` persists the transformed data into a high-performance analytical local database (`data/processed/aquaculture.duckdb`) under the `fact_trade` table.

---

## 🛠️ Requirements & Setup

### Prerequisites
- Python 3.9+
- Python packages: `pandas`, `duckdb`, `numpy`

### Running the Pipeline

Execute the main pipeline script:

```bash
python main.py
```

Upon execution, `main.py` will:
1. Load raw trade data from `data/raw/Aquaculture_Trade_Filtered.csv`.
2. Execute transformations and calculate unit prices.
3. Profile and output dataset summary stats to the console.
4. Store transformed facts in `data/processed/aquaculture.duckdb`.
5. Run a test query against `fact_trade` and display row count.

---

## 🔮 Roadmap

- [ ] **Domain Models**: Implement domain entities and validation schemas in `src/domain/models.py`.
- [ ] **Forecasting Service**: Build ML-based time-series forecasting in `src/application/forecast_service.py`.
- [ ] **MLOps Tools**: Integrate experiment tracking (MLflow), data versioning (DVC), and pipeline orchestration.
- [ ] **Automated Testing**: Add unit tests for ETL transformations and data loaders.
