import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import os
import pandas as pd



class BaseLogger:

    def __init__(self):
        pass
    

class TrainLogger(BaseLogger):

    def __init__(self, info_keys, vn30_tickets, trading_days):
        super().__init__()
        self.info_keys = info_keys
        self.vn30_tickets = vn30_tickets
        self.trading_days = trading_days

        self.episodes_accuracy = []
        self.episodes_f1 = []
        self.episodes_idx = []
        
    def init_episode(self):
        self.cur_ep_trues = []
        self.cur_ep_predictions = []
    
    def log_step(self, info):
        self.cur_ep_trues.append(info['true'].tolist())
        self.cur_ep_predictions.append(info['prediction'].tolist())

    def record(self, episode):     
        trues = np.array(self.cur_ep_trues).reshape(-1)
        predictions = np.array(self.cur_ep_predictions).reshape(-1)
        accuracy = accuracy_score(trues, predictions)
        f1 = f1_score(trues, predictions, average='weighted')
        self.episodes_accuracy.append(accuracy)
        self.episodes_f1.append(f1)
        self.episodes_idx.append(int(episode))
        
        cf_mt = confusion_matrix(trues, predictions, normalize='all')
        true_ind_len, prediction_ind_len = cf_mt.shape
        cf_mt_dict = {f'true-{true_ind}_prediction-{prediction_ind}' : cf_mt[true_ind][prediction_ind] for true_ind in range(true_ind_len) for prediction_ind in range(prediction_ind_len)}
    
        return accuracy, f1, cf_mt_dict

    def to_csv(self, save_path):
        csv_files = os.listdir(save_path)
        metrics_csv = 'train_metrics.csv'
        if (metrics_csv in csv_files):
            exist_metrics_df = pd.read_csv(save_path + metrics_csv, index_col='episode')
            metrics_df = pd.DataFrame({'accuracy': self.episodes_accuracy, 'f1': self.episodes_f1, 'episode': self.episodes_idx})
            metrics_df.set_index('episode', inplace=True)
            metrics_df = pd.concat([exist_metrics_df, metrics_df], axis=0)
            metrics_df.to_csv(save_path + metrics_csv, header=True, index=True)

        else:
            metrics_df = pd.DataFrame({'accuracy': self.episodes_accuracy, 'f1': self.episodes_f1, 'episode': self.episodes_idx})
            metrics_df.set_index('episode', inplace=True)
            metrics_df.to_csv(save_path + metrics_csv, header=True, index=True)

        max_idx = metrics_df['f1'].idxmax()
        max_acc, max_f1 = metrics_df.loc[max_idx, 'accuracy'], metrics_df.loc[max_idx, 'f1']
        return max_idx, max_acc, max_f1
    
def zip_final_info(final_info):
    trues = ([item['true'] for item in final_info])
    predictions = ([item['prediction'] for item in final_info])
    return trues, predictions

class TestLogger(BaseLogger):

    def __init__(self, info_keys, vn30_tickets, trading_days):
        super().__init__()
        self.info_keys = info_keys
        self.vn30_tickets = vn30_tickets
        self.trading_days = trading_days

        self.episodes_accuracy = []
        self.episodes_f1 = []
        self.episodes_idx = []
        
    def init_episode(self):
        self.cur_ep_trues = []
        self.cur_ep_predictions = []
    
    def log_step(self, info, terminated):
        if terminated.all():
            trues, predictions = zip_final_info(info['final_info'])
        else:
            trues, predictions = info['true'].tolist(), info['prediction'].tolist()

        self.cur_ep_trues.append(trues)
        self.cur_ep_predictions.append(predictions)

    def record(self, episode):     
        trues = np.array(self.cur_ep_trues).reshape(-1)
        predictions = np.array(self.cur_ep_predictions).reshape(-1)
        accuracy = accuracy_score(trues, predictions)
        f1 = f1_score(trues, predictions, average='weighted')
        self.episodes_accuracy.append(accuracy)
        self.episodes_f1.append(f1)
        self.episodes_idx.append(episode)


        cf_mt = confusion_matrix(trues, predictions, normalize='all')
        true_ind_len, prediction_ind_len = cf_mt.shape
        cf_mt_dict = {f'true-{true_ind}_prediction-{prediction_ind}' : cf_mt[true_ind][prediction_ind] for true_ind in range(true_ind_len) for prediction_ind in range(prediction_ind_len)}
        
        return accuracy, f1, cf_mt_dict

    def to_csv(self, save_path):
        csv_files = os.listdir(save_path)
        metrics_csv = 'test_metrics.csv'
        if (metrics_csv in csv_files):
            exist_metrics_df = pd.read_csv(save_path + metrics_csv, index_col='episode')
            metrics_df = pd.DataFrame({'accuracy': self.episodes_accuracy, 'f1': self.episodes_f1, 'episode': self.episodes_idx})
            metrics_df.set_index('episode', inplace=True)
            metrics_df = pd.concat([exist_metrics_df, metrics_df], axis=0)
            metrics_df.to_csv(save_path + metrics_csv, header=True, index=True)

        else:
            metrics_df = pd.DataFrame({'accuracy': self.episodes_accuracy, 'f1': self.episodes_f1, 'episode': self.episodes_idx})
            metrics_df.set_index('episode', inplace=True)
            metrics_df.to_csv(save_path + metrics_csv, header=True, index=True)

        max_idx = metrics_df['f1'].idxmax()
        max_acc, max_f1 = metrics_df.loc[max_idx, 'accuracy'], metrics_df.loc[max_idx, 'f1']
        return max_idx, max_acc, max_f1