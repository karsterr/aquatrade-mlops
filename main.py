from src.infrastructure.data_loaders.csv_loader import CSVDataLoader
from src.application.data_inspector import DataInspector
from src.infrastructure.db.duckdb_client import DuckDBClient
from src.application.etl_service import ETLService

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
    print("\nDuckDB 'fact_trade' Tablosundaki Toplam Satır Sayısı:")
    print(count_df)

if __name__ == "__main__":
    main()
