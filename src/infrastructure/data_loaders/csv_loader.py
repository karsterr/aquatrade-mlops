import pandas as pd

class CSVDataLoader:
    def __init__(self, path):
        self.path = path

    def load_raw_data(self):
        try:
            df = pd.read_csv(self.path)
            return df
        except FileNotFoundError:
            None
        except Exception as e:
            None
