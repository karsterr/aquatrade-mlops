class DataInspector:
    def inspect(self, df):
        if df is None:
            print("Empty dataframe")
            return
        print("===== BOYUT =====")
        print(df.shape)
        print("===== VERI TIPLERI =====")
        print(df.dtypes)
        print("===== EKSIK DEGERLER =====")
        print(df.isnull().sum())
        print("===== ILK 5 SATIR =====")
        print(df.head())
