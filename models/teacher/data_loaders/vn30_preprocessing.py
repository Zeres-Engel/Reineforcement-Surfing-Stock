import numpy as np
import pandas as pd
import os
import talib
from talib.abstract import *


def z_close_normalize(df, close):
    res = df.div(close, axis=0)
    return res


def min_max_normalize(pd_series, x_max, x_min):
    return (pd_series - x_min) / (x_max - x_min)


def normalize(stock_df):
    z_close_norm_cols = [
        "open",
        "high",
        "low",
        "macd",
        "boll_ub",
        "boll_lb",
        "close_sma_30",
        "close_sma_60",
    ]
    stock_df.loc[:, z_close_norm_cols] = z_close_normalize(
        stock_df.loc[:, z_close_norm_cols], stock_df["close"]
    )

    stock_df.loc[:, "rsi_30"] = min_max_normalize(stock_df.loc[:, "rsi_30"], 100, 0)
    stock_df.loc[:, "cci_30"] = min_max_normalize(stock_df.loc[:, "cci_30"], 100, -100)
    stock_df.loc[:, "dx_30"] = min_max_normalize(stock_df.loc[:, "dx_30"], 100, 0)

    stock_df.loc[:, "close"] = stock_df.loc[:, "close"] / stock_df.loc[
        :, "close"
    ].shift(1)
    stock_df.loc[:, "volume"] = stock_df.loc[:, "volume"] / stock_df.loc[
        :, "volume"
    ].shift(1)

    return stock_df


def add_and_normalize_features(stock_df):
    # add indicators
    inputs = {
        "open": stock_df["open"].to_numpy(dtype="float"),
        "high": stock_df["high"].to_numpy(dtype="float"),
        "low": stock_df["low"].to_numpy(dtype="float"),
        "close": stock_df["close"].to_numpy(dtype="float"),
        "volume": stock_df["volume"].to_numpy(dtype="float"),
    }
    stock_df["macd"], _, _ = MACD(inputs)
    stock_df["boll_ub"], _, stock_df["boll_lb"] = BBANDS(inputs)
    stock_df["rsi_30"] = RSI(inputs, timeperiod=30)
    stock_df["cci_30"] = CCI(inputs, timeperiod=30)
    stock_df["dx_30"] = DX(inputs, timeperiod=30)
    stock_df["close_sma_30"] = SMA(inputs, timeperiod=30)
    stock_df["close_sma_60"] = SMA(inputs, timeperiod=60)

    # normalize
    stock_df = normalize(stock_df)

    # remove na rows
    stock_df.dropna(axis=0, how="any", inplace=True)
    return stock_df
