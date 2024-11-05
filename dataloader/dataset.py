# dataloader/dataset.py
import pandas as pd
import logging
import joblib
from itertools import combinations
from .preprocessing import Preprocessing

class Dataset(Preprocessing):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Load raw data
        self._load_raw_data()

    def _load_raw_data(self):
        """Load raw data from CSV file"""
        columns_to_use = ['time'] + self.config['data']['features']
        self.data = pd.read_csv(self.config['data']['data_path'], 
                                usecols=columns_to_use, 
                                parse_dates=["time"])

        if 'time' not in self.data.columns:
            raise KeyError("The 'time' column is missing in the dataset.")
        
        # Process the time column
        self.data['time'] = pd.to_datetime(self.data['time'], dayfirst=True, errors='coerce')
        if self.data['time'].isnull().any():
            logging.warning("Một số mục thời gian không thể phân tích cú pháp và sẽ bị loại bỏ.")
            self.data = self.data.dropna(subset=['time'])

    def process_data(self, filtered_data, features):
        """Normalize selected features"""
        processed_data = self.fit_transform(filtered_data, features)
        return processed_data

    def _filter_by_date(self, start_date, end_date):
        """Filter data by date range"""
        mask = (self.data['time'] >= pd.to_datetime(start_date)) & \
               (self.data['time'] <= pd.to_datetime(end_date))
        filtered_data = self.data.loc[mask].reset_index(drop=True)
        
        if filtered_data.empty:
            raise ValueError(f"No data found between {start_date} and {end_date}")
        
        return filtered_data

    def augment_features(self, data, strategy='default'):
        """Add technical indicators and transform data"""
        augmented_data = data.copy()
        
        if strategy == 'default':
            augmented_data = self.add_technical_indicators(augmented_data)
        elif strategy == 'advanced':
            price_columns = ['open', 'high', 'low', 'close']
            augmented_data = self.boxcox_transform(augmented_data, price_columns)
            augmented_data = self.add_technical_indicators(augmented_data)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        return augmented_data

    def prepare_data(self, start_date, end_date, features, strategy='default', denoise=False):
        """Prepare data pipeline without feature selection"""
        try:
            # 1. Lọc dữ liệu theo ngày
            filtered_data = self._filter_by_date(start_date, end_date)
            
            # 2. Augment features
            augmented_data = self.augment_features(filtered_data, strategy)
            
            # 3. Chuẩn hóa các tính năng đã chọn
            processed_data = self.process_data(augmented_data, features)
            
            # 4. Denoise dữ liệu nếu cần
            if denoise:
                processed_data = self.denoise_data(processed_data, method='median_filter')
            
            return processed_data
        except ValueError as e:
            logging.error(e)
            return None

    def get_features_data(self, data, selected_features):
        """Get data with selected features plus 'close'"""
        if 'close' not in data.columns:
            raise KeyError("'close' column is missing in the data.")
        
        selected_features = list(selected_features)
        if 'close' not in selected_features:
            selected_features.append('close')  # Retain 'close' for profit calculation
        
        if not set(selected_features).issubset(data.columns):
            missing = set(selected_features) - set(data.columns)
            raise KeyError(f"The following selected features are missing in the data: {missing}")
        
        return data[selected_features].copy()

    def save_scaler(self, path):
        """Save the scaler to a file"""
        joblib.dump(self.scaler, path)
        logging.info(f"Scaler saved to {path}")

    def load_scaler(self, path):
        """Load the scaler from a file"""
        self.scaler = joblib.load(path)
        logging.info(f"Scaler loaded from {path}")

    def get_all_augmented_features(self, start_date, end_date):
        """Get all features after augmentation"""
        filtered_data = self._filter_by_date(start_date, end_date)
        
        augmented_data = self.augment_features(filtered_data, strategy='advanced')
        
        feature_columns = augmented_data.select_dtypes(include=['float64', 'int64']).columns
        
        original_columns = len(filtered_data.columns)
        augmented_columns = len(augmented_data.columns)
        logging.info(f"Original columns: {original_columns}, After augment columns: {augmented_columns}")
        
        return list(feature_columns), augmented_data
