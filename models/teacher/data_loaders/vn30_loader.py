import numpy as np
import pandas as pd
import os
import talib
from talib.abstract import *
from utils.logger import get_logger
from utils.utils import last_day_of_month
from .vn30_preprocessing import *

LOG = get_logger("data_loader")


class DataLoader:
    def __init__(
        self,
        vn30_raw_df_dict,
        vn30_df_dict,
        timesteps_dim,
        n_predictions,
        ticket,
        train_datetime_dict,
        validation_date_dict,
        test_datetime_dict,
    ):
        self.vn30_raw_df_dict = vn30_raw_df_dict
        self.vn30_df_dict = vn30_df_dict
        self.timesteps_dim = timesteps_dim
        self.n_predictions = n_predictions
        self.ticket = ticket
        self.train_datetime_dict = train_datetime_dict
        self.validation_date_dict = validation_date_dict
        self.test_datetime_dict = test_datetime_dict

    @classmethod
    def from_json(cls, cfg):
        LOG.info(f"Loading dataset...")
        thesis_dir = cfg.path
        timesteps_dim = cfg.timesteps_dim
        n_predictions = cfg.n_predictions
        ticket = cfg.ticket

        stock_files = os.listdir(thesis_dir)
        vn30_df_dict = {}
        vn30_raw_df_dict = {}


        for file in stock_files:
            ticket_temp = file[:-4]
            raw_stock_df = pd.read_csv(thesis_dir + file, index_col="date")
            stock_df = add_and_normalize_features(raw_stock_df.copy())
            raw_stock_df = raw_stock_df.loc[stock_df.index, :]
            """ print(raw_stock_df)
            print(stock_df)
            exit() """
            vn30_raw_df_dict[ticket_temp] = raw_stock_df

            vn30_df_dict[ticket_temp] = stock_df
            # stock_df.to_csv('./debug/normalized_data/' + f'{ticket}.csv')

        """ print(type(trading_days_lst))
        print((test_day_ind))
        print(vn30_x_df.loc[trading_days_lst[test_day_ind + 1], :]) """

        train_datetime_dict = {"start": cfg.start_train_date, "end": cfg.end_train_date}

        validation_date_dict = {
            "start": cfg.start_validation_date,
            "end": cfg.end_validation_date,
        }

        test_datetime_dict = {"start": cfg.start_test_date, "end": cfg.end_test_date}

        LOG.info(f"Loading finished")
        return cls(
            vn30_raw_df_dict,
            vn30_df_dict,
            timesteps_dim,
            n_predictions,
            ticket,
            train_datetime_dict,
            validation_date_dict,
            test_datetime_dict,
        )

    def get_timesteps_dim(self):
        return self.timesteps_dim

    def get_tickets_dim(self):
        return len(self.vn30_df_dict.keys())

    def get_features_dim(self):
        return self.vn30_df_dict["ACB"].shape[1]

    def get_samples_dim(self):
        return self.vn30_df_dict["ACB"].shape[0]

    def get_tickets(self):
        return list(self.vn30_df_dict.keys())

    def get_vn30_raw_df_dict(self):
        return self.vn30_raw_df_dict

    def get_vn30_df_dict(self):
        return self.vn30_df_dict

    def get_test_day(self):
        return self.test_day

    def get_day_ind(self, dt_obj):
        day_ind = self.get_trading_days().index(dt_obj)
        return day_ind

    def get_trading_days(self):
        ticket_sample = "ACB"
        ticket_sample_df = self.vn30_df_dict[ticket_sample]
        trading_days = pd.unique(ticket_sample_df.index).tolist()
        return trading_days

    def get_n_predictions(self):
        return self.n_predictions

    def get_train_date_inds(self):
        train_start_day_ind = 0
        train_end_day_ind = self.get_day_ind(self.train_datetime_dict["end"])
        return train_start_day_ind, train_end_day_ind

    def get_validation_date_inds(self):
        validation_start_day_ind = self.get_day_ind(self.validation_date_dict["start"])
        validation_end_day_ind = self.get_day_ind(self.validation_date_dict["end"])
        return validation_start_day_ind, validation_end_day_ind

    def get_test_date_inds(self):
        test_start_day_ind = self.get_day_ind(self.test_datetime_dict["start"])
        test_end_day_ind = self.get_day_ind(self.test_datetime_dict["end"])
        return test_start_day_ind, test_end_day_ind

    def get_ticket(self):
        # return 'ACB'
        return self.ticket
