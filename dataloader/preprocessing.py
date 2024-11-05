# dataloader/preprocessing.py
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pywt
import pandas_ta as ta
import seaborn as sns
import numpy as np 

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
    
    def discrete_wavelet_transform(self, df, feature='close', wavelet='db1', level=4):
        close_prices = df[feature].values

        coeffs = pywt.wavedec(close_prices, wavelet, level=level)

        # Reconstruct signal from approximation coefficients to get transformed close prices
        approx = coeffs[0]
        reconstructed_close_prices = pywt.waverec([approx] + [None] * (len(coeffs) - 1), wavelet)
        reconstructed_close_prices = reconstructed_close_prices[:len(close_prices)]

        # Create a new DataFrame with the DWT-transformed feature
        df_dwt = df.copy()
        df_dwt[f'{feature}_dwt'] = reconstructed_close_prices

        return df_dwt
    
    def Gaussian_noise(self,df,feature, std_range=(0.5, 1)):
        df_gauss = self.df.copy()
        std_dev = np.random.uniform(std_range[0], std_range[1])
        noise = np.random.normal(0, std_dev, size=df[self.feature].shape)
        df[f"{self.feature}_with_noise"] = df[self.feature] + noise
        return df_gauss
    
    def add_gaussian_noise(self, df, feature, std_range=(0.5, 1.0)):
        df_gauss = df.copy()
        std_dev = np.random.uniform(std_range[0], std_range[1])
        noise = np.random.normal(0, std_dev, size=df[feature].shape)
        df_gauss[f"{feature}_with_noise"] = df[feature] + noise
        return df_gauss

    def z_close_normalize(self, df, close):
        res = df.div(close, axis=0)
        return res

    def min_max_normalize(self, pd_series, x_max, x_min):
        return (pd_series - x_min) / (x_max - x_min)

    def normalize(self, stock_df):
        z_close_norm_cols = [
            "open", "high", "low", "macd", "boll_ub", "boll_lb", "close_sma_30", "close_sma_60", "stoch_k"
        ]
        stock_df[z_close_norm_cols] = self.z_close_normalize(stock_df[z_close_norm_cols], stock_df["close"])

        stock_df["rsi_30"] = self.min_max_normalize(stock_df["rsi_30"], 100, 0)
        stock_df["cci_30"] = self.min_max_normalize(stock_df["cci_30"], 100, -100)
        stock_df["adx_30"] = self.min_max_normalize(stock_df["adx_30"], 100, 0)
        stock_df["stoch_k"] = self.min_max_normalize(stock_df["stoch_k"], 100, 0)

        stock_df["close"] = stock_df["close"] / stock_df["close"].shift(1)
        stock_df["volume"] = stock_df["volume"] / stock_df["volume"].shift(1)
        return stock_df

    def add_and_normalize_features(self, stock_df):
        # Calculate indicators using pandas_ta
        stock_df["macd"] = ta.macd(stock_df["close"]).iloc[:, 0]  # MACD line
        bollinger = ta.bbands(stock_df["close"], length=20, std=2)
        stock_df["boll_ub"] = bollinger["BBU_20_2.0"]
        stock_df["boll_lb"] = bollinger["BBL_20_2.0"]
        stock_df["rsi_30"] = ta.rsi(stock_df["close"], length=30)
        stock_df["cci_30"] = ta.cci(stock_df["high"], stock_df["low"], stock_df["close"], length=30)
        adx = ta.adx(stock_df["high"], stock_df["low"], stock_df["close"], length=30)
        stock_df["adx_30"] = adx["ADX_30"]
        stock_df["close_sma_30"] = ta.sma(stock_df["close"], length=30)
        stock_df["close_sma_60"] = ta.sma(stock_df["close"], length=60)

        # Calculate the %K Stochastic Oscillator
        high_14 = stock_df["high"].rolling(window=14).max()
        low_14 = stock_df["low"].rolling(window=14).min()
        stock_df["stoch_k"] = 100 * (stock_df["close"] - low_14) / (high_14 - low_14)

        # Normalize
        stock_df = self.normalize(stock_df)

        # Remove rows with NaN values
        stock_df.dropna(axis=0, how="any", inplace=True)
        return stock_df
    
    def lagged_features(self, df, lag_features, num_lags=3):
        # Có thể cân nhắc, thêm vào faeture giá của trước đó vài ngày, thay vì 1 ngày
        df_with_lags = df.copy() 

        for feature in lag_features:
            for lag in range(1, num_lags + 1):
                df_with_lags[f'{feature}_lag_{lag}'] = df_with_lags[feature].shift(lag)

        # Loại bỏ các hàng đầu tiên chứa NaN do việc tạo độ trễ
        df_with_lags.dropna(inplace=True)

        return df_with_lags


