import pandas as pd
import numpy as np

class ETLService:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        df_clean['TIMEPERIOD_ID'] = df_clean['TIMEPERIOD_ID'].astype(str).str.zfill(2)
        df_clean['YEAR_ID'] = df_clean['YEAR_ID'].astype(str)

        date_str = df_clean['YEAR_ID'] + '-' + df_clean['TIMEPERIOD_ID'] + '-01'
        df_clean['date'] = pd.to_datetime(date_str)

        df_pivot = pd.pivot_table(
            df_clean,
            index=['date', 'GEOGRAPHY_CODE', 'GEOGRAPHY_DESC', 'HS_CODE', 'COMMODITY_DESC'],
            columns = 'ATTRIBUTE_DESC',
            values = 'AMOUNT',
            aggfunc = 'sum'
        ).reset_index()

        df_pivot.columns.name = None

        df_pivot = df_pivot.rename(columns={
            'US Import, QTY': 'import_qty',
            'US Import, VLU': 'import_vlu',
            'US Export, QTY': 'export_qty',
            'US Export, VLU': 'export_vlu'
        })

        df_pivot['import_unit_price'] = df_pivot['import_vlu'] / df_pivot['import_qty'].replace(0, np.nan)
        df_pivot['export_unit_price'] = df_pivot['export_vlu'] / df_pivot['export_qty'].replace(0, np.nan)

        return df_pivot
