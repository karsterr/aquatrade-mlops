from src.infrastructure.data_loaders.csv_loader import CSVDataLoader
from src.adapters.presenters.data_inspector import DataInspector
from src.infrastructure.db.duckdb_client import DuckDBClient
from src.use_cases.etl_pipeline import ETLService

def main():
    path = "data/raw/Aquaculture_Trade_Filtered.csv"
    loader = CSVDataLoader(path)
    df = loader.load_raw_data()

    etl = ETLService()
    df_transformed = etl.transform(df)

    inspector = DataInspector()
    inspector.inspect(df_transformed)

    db_client = DuckDBClient()
    db_client.save_dataframe(df_transformed, "fact_trade")

    count_df = db_client.execute_query("SELECT COUNT(*) AS total_rows FROM fact_trade")
    print("\nDuckDB 'fact_trade' Tablosundaki Toplam Satir Sayisi:")
    print(count_df)

if __name__ == "__main__":
    main()
