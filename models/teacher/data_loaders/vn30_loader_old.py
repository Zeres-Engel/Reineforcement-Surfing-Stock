import torch
import numpy as np
import pandas as pd
import os
import talib
import datetime
from talib.abstract import *
from utils.logger import get_logger
from utils.utils import last_day_of_month
LOG = get_logger('data_loader')


def z_close_normalize(df, close):
    res = df.div(close, axis=0)
    return res

def range_scaling(pd_series, x_max, x_min):
    return (pd_series - x_min) / (x_max - x_min)

def normalize(stock_df):
    z_close_norm_cols = ['open', 'high', 'low', 'macd', 'boll_ub', 'boll_lb', 'close_sma_30', 'close_sma_60']
    stock_df.loc[:, z_close_norm_cols] = z_close_normalize(stock_df.loc[:, z_close_norm_cols], stock_df['close'])
    
    stock_df.loc[:, 'rsi_30'] = range_scaling(stock_df.loc[:, 'rsi_30'], 100, 0)
    stock_df.loc[:, 'cci_30'] = range_scaling(stock_df.loc[:, 'cci_30'], 100, -100)
    stock_df.loc[:, 'dx_30'] = range_scaling(stock_df.loc[:, 'dx_30'], 100, 0)

    stock_df.loc[:, 'close'] = stock_df.loc[:, 'close'] / stock_df.loc[:, 'close'].shift(1)
    stock_df.loc[:, 'volume'] = stock_df.loc[:, 'volume'] / stock_df.loc[:, 'volume'].shift(1) 

    return stock_df


def add_and_normalize_features(stock_df):
    #add indicators
    inputs = {
        'open': stock_df['open'].to_numpy(dtype='float'), 
        'high': stock_df['high'].to_numpy(dtype='float'), 
        'low': stock_df['low'].to_numpy(dtype='float'), 
        'close': stock_df['close'].to_numpy(dtype='float'), 
        'volume': stock_df['volume'].to_numpy(dtype='float'), 
    }
    stock_df['macd'], _, _ = MACD(inputs)
    stock_df['boll_ub'], _, stock_df['boll_lb'] = BBANDS(inputs)
    stock_df['rsi_30'] = RSI(inputs, timeperiod=30)
    stock_df['cci_30'] = CCI(inputs, timeperiod=30)
    stock_df['dx_30'] = DX(inputs, timeperiod=30)
    stock_df['close_sma_30'] = SMA(inputs, timeperiod=30)
    stock_df['close_sma_60'] = SMA(inputs, timeperiod=60)

    #normalize
    stock_df = normalize(stock_df)

    #remove na rows
    stock_df.dropna(axis=0, how='any', inplace=True)
    return stock_df

class DataLoader:


    def __init__(self, vn30_df_dict, timesteps_dim, n_predictions, split_method, stock, train_datetime_dict, test_datetime_dict):
        self.vn30_df_dict = vn30_df_dict
        self.timesteps_dim = timesteps_dim
        self.n_predictions = n_predictions
        self.split_method = split_method
        self.stock = stock
        self.train_datetime_dict = train_datetime_dict
        self.test_datetime_dict = test_datetime_dict

    @classmethod
    def from_json(cls, cfg):
        LOG.info(f'Loading dataset...')
        thesis_dir = cfg.path
        timesteps_dim = cfg.timesteps_dim
        n_predictions = cfg.n_predictions
        split_method = cfg.split_method
        stock = cfg.stock

        stock_files = os.listdir(thesis_dir)
        vn30_df_dict = {}
        
        for file in stock_files:
            ticket = file[:-4]
            stock_df = pd.read_csv(thesis_dir + file, index_col='date')
            stock_df = add_and_normalize_features(stock_df)

            vn30_df_dict[ticket] = stock_df
            #stock_df.to_csv('./debug/normalized_data/' + f'{ticket}.csv')


        """ print(type(trading_days_lst))
        print((test_day_ind))
        print(vn30_x_df.loc[trading_days_lst[test_day_ind + 1], :]) """
        
        train_datetime_dict = { 'start': cfg.start_train_day,
                                'end': cfg.end_train_day
                            }

        test_datetime_dict = { 'start': cfg.start_test_day,
                                'end': cfg.end_test_day
                            }

        LOG.info(f'Loading finished')
        return cls(vn30_df_dict, timesteps_dim, n_predictions, split_method, stock, train_datetime_dict, test_datetime_dict)
    
    def get_timesteps_dim(self):
        return self.timesteps_dim
    
    def get_tickets_dim(self):
        return len(self.vn30_df_dict.keys())
    
    def get_features_dim(self):
        return self.vn30_df_dict['ACB'].shape[1]
    
    def get_samples_dim(self):
        return self.vn30_df_dict['ACB'].shape[0]
    
    def get_tickets(self):
        return list(self.vn30_df_dict.keys())
    
    def get_split_method(self):
        return self.split_method

    def get_vn30_df_dict(self):
        return self.vn30_df_dict
    
    def get_test_day(self):
        return self.test_day
    
    def get_day_ind(self, dt_obj):
        day_ind = self.get_trading_days().index(dt_obj)
        return day_ind
    
    def get_trading_days(self):
        ticket_sample = 'ACB'
        ticket_sample_df = self.vn30_df_dict[ticket_sample]
        trading_days = pd.unique(ticket_sample_df.index).tolist()
        return trading_days
    
    def get_n_predictions(self):
        return self.n_predictions
    
    def get_days_ind_train_test_split(self):
        train_start_day_ind = 0
        train_end_day_ind = self.get_day_ind(self.train_datetime_dict['end'])
        test_start_day_ind = self.get_day_ind(self.test_datetime_dict['start'])
        test_end_day_ind = self.get_day_ind(self.test_datetime_dict['end'])
        #print(f'{train_start_day_ind} {train_end_day_ind} {test_start_day_ind} {test_end_day_ind}')
        return train_start_day_ind, train_end_day_ind, test_start_day_ind, test_end_day_ind

    def get_stock(self):
        return self.stock