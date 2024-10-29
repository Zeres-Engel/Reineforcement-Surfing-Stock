import pandas as pd
import numpy as np

class StockData:
    def __init__(self, file_path, ma_windows=[5, 10, 20]):
        self.file_path = file_path
        self.ma_windows = ma_windows
        self.data = self._load_data()
        self._compute_moving_averages()
        
    def _load_data(self):
        df = pd.read_csv(self.file_path, parse_dates=['time'])
        df.sort_values('time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def _compute_moving_averages(self):
        for window in self.ma_windows:
            self.data[f'MA_{window}'] = self.data['close'].rolling(window=window).mean()
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
    
    def get_data(self):
        return self.data
