# dataloader/preprocessing.py
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from scipy import stats

class Preprocessing:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, df, features):
        """Fit the scaler on the training data and transform it."""
        df_scaled = df.copy()
        self.scaler.fit(df_scaled[features])
        df_scaled[features] = self.scaler.transform(df_scaled[features])
        return df_scaled

    def transform(self, df, features):
        """Transform the data using the already fitted scaler."""
        df_scaled = df.copy()
        df_scaled[features] = self.scaler.transform(df_scaled[features])
        return df_scaled

    def boxcox_transform(self, df, columns):
        """Apply Box-Cox transformation to specified columns."""
        df_transformed = df.copy()
        for col in columns:
            # Box-Cox requires positive data
            if (df_transformed[col] <= 0).any():
                min_val = df_transformed[col].min()
                df_transformed[col] = df_transformed[col] - min_val + 1e-6
            df_transformed[col], _ = stats.boxcox(df_transformed[col])
        return df_transformed

    def add_technical_indicators(self, df):
        """Add technical indicators (e.g., H-L, O-C, moving averages)."""
        df_indicators = df.copy()
        df_indicators['H-L'] = df_indicators['high'] - df_indicators['low']
        df_indicators['O-C'] = df_indicators['open'] - df_indicators['close']
        
        ma_windows = [5, 10, 20]
        for window in ma_windows:
            df_indicators[f'MA_{window}'] = df_indicators['close'].rolling(window=window, min_periods=1).mean()
            df_indicators[f'STD_{window}'] = df_indicators['close'].rolling(window=window, min_periods=1).std()
        
        df_indicators.fillna(method='bfill', inplace=True)
        return df_indicators

    def denoise_data(self, df, method='median_filter'):
        """Denoise data using specified method."""
        df_denoised = df.copy()
        numeric_columns = df_denoised.select_dtypes(include=['float64', 'int64']).columns
        
        if method == 'median_filter':
            df_denoised[numeric_columns] = df_denoised[numeric_columns].rolling(window=3, min_periods=1).median()
        elif method == 'mean_filter':
            df_denoised[numeric_columns] = df_denoised[numeric_columns].rolling(window=3, min_periods=1).mean()
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        return df_denoised

    def augment_data(self, df, strategy='default'):
        """Augment data using specified strategy."""
        if strategy == 'default':
            df = self.add_technical_indicators(df)
        elif strategy == 'advanced':
            df = self.boxcox_transform(df, ['open', 'high', 'low', 'close'])
            df = self.add_technical_indicators(df)
        else:
            raise ValueError(f"Unknown augmentation strategy: {strategy}")
        return df
