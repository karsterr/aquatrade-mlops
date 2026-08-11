import os
import duckdb
import pandas as pd

class DuckDBClient:
    def __init__(self, db_path: str = "data/processed/aquaculture.duckdb"):
        self.db_path = db_path
        
        # Klasör yoksa oluştur (data/processed dizinini garantiye almak için)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def get_connection(self):
        """Veritabanına bir bağlantı açar ve döndürür."""
        return duckdb.connect(self.db_path)

    def save_dataframe(self, df: pd.DataFrame, table_name: str = "fact_trade") -> None:
        """Pandas DataFrame'ini DuckDB içerisinde SQL tablosuna kaydeder."""
        conn = self.get_connection()
        
        # DuckDB, conn.execute içinde Python değişkenlerini (df) doğrudan görebilir.
        # Tablo yoksa oluşturur, varsa üzerine yazar (CREATE OR REPLACE TABLE)
        query = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df"
        
        conn.execute(query)
        conn.close()
        print(f"[OK] DataFrame basariyla '{table_name}' tablosuna kaydedildi ({len(df)} satir).")

    def execute_query(self, query: str) -> pd.DataFrame:
        """Verilen SQL sorgusunu çalıştırır ve sonucu Pandas DataFrame olarak döner."""
        conn = self.get_connection()
        result_df = conn.execute(query).df()
        conn.close()
        return result_df
