# dataloader/dataset.py
import pandas as pd
import logging
from .preprocessing import Preprocessing

class Dataset(Preprocessing):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._load_raw_data()

    def _load_raw_data(self):
        """Load raw data from CSV file"""
        columns_to_use = ['time'] + self.config['data']['features']
        self.data = pd.read_csv(self.config['data']['data_path'], 
                               usecols=columns_to_use, 
                               parse_dates=["time"])
        
        if 'time' not in self.data.columns:
            raise KeyError("The 'time' column is missing in the dataset.")
        
        self.data['time'] = pd.to_datetime(self.data['time'], dayfirst=True, errors='coerce')
        self.data = self.data.dropna(subset=['time'])

    def _filter_by_date(self, start_date, end_date):
        """Filter data by date range"""
        mask = (self.data['time'] >= pd.to_datetime(start_date)) & \
               (self.data['time'] <= pd.to_datetime(end_date))
        filtered_data = self.data.loc[mask].reset_index(drop=True)
        
        if filtered_data.empty:
            raise ValueError(f"No data found between {start_date} and {end_date}")
        
        return filtered_data

    def prepare_data(self, start_date, end_date, features):
        """Main data preparation pipeline"""
        try:
            filtered_data = self._filter_by_date(start_date, end_date)
            augmented_data = self.add_technical_indicators(filtered_data)
            
            # Chuẩn hóa dữ liệu
            numeric_columns = augmented_data.select_dtypes(include=['float64', 'int64']).columns
            augmented_data[numeric_columns] = self.scaler.fit_transform(augmented_data[numeric_columns])
            
            # Khử nhiễu và chọn features
            augmented_data = self.denoise_data(augmented_data)
            if features is not None:
                augmented_data = self.get_features_data(augmented_data, features)
            
            return augmented_data.fillna(method='ffill').fillna(method='bfill')
            
        except Exception as e:
            logging.error(f"Error in prepare_data: {str(e)}")
            return None

    def get_features_data(self, data, selected_features):
        """Get data with selected features plus 'close'"""
        if 'close' not in data.columns:
            raise KeyError("'close' column is missing in the data.")
        
        selected_features = list(selected_features)
        if 'close' not in selected_features:
            selected_features.append('close')
        
        if not set(selected_features).issubset(data.columns):
            missing = set(selected_features) - set(data.columns)
            raise KeyError(f"Missing features: {missing}")
        
        return data[selected_features].copy()

    def get_all_augmented_features(self, start_date, end_date):
        """Get complete list of features after augmentation"""
        filtered_data = self._filter_by_date(start_date, end_date)
        augmented_data = self.add_technical_indicators(filtered_data)
        return list(augmented_data.select_dtypes(include=['float64', 'int64']).columns), augmented_data
