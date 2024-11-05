import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from scipy import stats

class Preprocessing:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, df, features):
        df_scaled = df.copy()
        self.scaler.fit(df_scaled[features])
        df_scaled[features] = self.scaler.transform(df_scaled[features])
        return df_scaled

    def transform(self, df, features):
        df_scaled = df.copy()
        df_scaled[features] = self.scaler.transform(df_scaled[features])
        return df_scaled

    def boxcox_transform(self, df, columns):
        """Apply Box-Cox transformation to specified columns."""
        for col in columns:
            df[col], _ = stats.boxcox(df[col].clip(lower=1e-10))
        return df

    def add_technical_indicators(self, df):
        """Add technical indicators (e.g., H-L, O-C, moving averages)."""
        df['H-L'] = df['high'] - df['low']
        df['O-C'] = df['open'] - df['close']
        
        ma_windows = [5, 10, 20]
        for window in ma_windows:
            df[f'MA_{window}'] = df['close'].rolling(window=window).mean()
            df[f'STD_{window}'] = df['close'].rolling(window=window).std()
        
        df.fillna(method='bfill', inplace=True)
        return df

    def denoise_data(self, df, method='median_filter'):
        """Denoise data using specified method."""
        df_denoised = df.copy()
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
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

