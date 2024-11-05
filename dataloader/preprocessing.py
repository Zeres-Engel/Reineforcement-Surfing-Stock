# dataloader/preprocessing.py
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
import numpy as np 

class Preprocessing:
    def __init__(self):
        self.scaler = StandardScaler()
        self.windows = [419, 1007, 839, 503, 629, 559, 2516, 1258, 1678, 5033]

    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators with multiple windows"""
        df_indicators = df.copy()
        
        # 1. Price Based Indicators
        df_indicators['H-L'] = df_indicators['high'] - df_indicators['low']
        df_indicators['O-C'] = df_indicators['open'] - df_indicators['close']
        
        # 2. Technical Indicators cho mỗi window
        for window in self.windows:
            try:
                # ADX - Average Directional Index
                
                # C
                # SMA - Simple Moving Average
                df_indicators[f'SMA_{window}'] = ta.sma(df_indicators['close'], length=window)
                
                # EMA - Exponential Moving Average
                df_indicators[f'EMA_{window}'] = ta.ema(df_indicators['close'], length=window)
                
                # Bollinger Bands
                bb_result = ta.bbands(df_indicators['close'], length=window)
                if bb_result is not None:  # Kiểm tra kết quả
                    df_indicators[f'BB_upper_{window}'] = bb_result[f'BBU_{window}_2.0']
                    df_indicators[f'BB_lower_{window}'] = bb_result[f'BBL_{window}_2.0']
                
                # RSI - Relative Strength Index
                df_indicators[f'RSI_{window}'] = ta.rsi(df_indicators['close'], length=window)
                
                # MACD - Moving Average Convergence Divergence
                if window > 26:  # MACD chỉ có ý nghĩa với window đủ lớn
                    macd_result = ta.macd(df_indicators['close'], fast=12, slow=window)
                    if macd_result is not None:  # Kiểm tra kết quả
                        df_indicators[f'MACD_{window}'] = macd_result[f'MACD_12_{window}_9']
                
                # Stochastic Oscillator
                stoch_result = ta.stoch(df_indicators['high'], df_indicators['low'], df_indicators['close'], k=window)
                if stoch_result is not None:  # Kiểm tra kết quả
                    df_indicators[f'STOCH_{window}'] = stoch_result[f'STOCHk_{window}_3_3']
                
            except Exception as e:
                logging.warning(f"Error calculating indicators for window {window}: {str(e)}")
                continue

        return df_indicators

    def denoise_data(self, df, method='median_filter'):
        """Denoise data using median filter"""
        df_denoised = df.copy()
        numeric_columns = df_denoised.select_dtypes(include=['float64', 'int64']).columns
        
        if method == 'median_filter':
            for col in numeric_columns:
                df_denoised[col] = df_denoised[col].rolling(window=5, min_periods=1).median()
        
        return df_denoised


