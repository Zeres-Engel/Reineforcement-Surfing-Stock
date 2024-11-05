# dataloader/dataset.py
import pandas as pd
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from .preprocessing import Preprocessing

class Dataset(Preprocessing):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Load raw data
        self._load_raw_data()

    def _load_raw_data(self):
        """Load raw data from csv file"""
        columns_to_use = ['time'] + self.config['data']['features']
        self.data = pd.read_csv(self.config['data']['data_path'], 
                                usecols=columns_to_use, 
                                parse_dates=["time"])

        if 'time' not in self.data.columns:
            raise KeyError("The 'time' column is missing in the dataset.")
        
        # Process the time column
        self.data['time'] = pd.to_datetime(self.data['time'], dayfirst=True, errors='coerce')
        if self.data['time'].isnull().any():
            logging.warning("Some time entries could not be parsed and will be dropped.")
            self.data = self.data.dropna(subset=['time'])

    def process_data(self, start_date, end_date, features):
        """Process raw data: filter by date and normalize"""
        # Filter data by date range
        filtered_data = self._filter_by_date(start_date, end_date)
        
        # Normalize data
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

    def select_features(self, data, target_column='close', n_features=5):
        """
        Select the top n_features based on feature importance using RandomForest.
        """
        # Create target variable
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        data = data.dropna()
        
        X = data.drop(['time', 'close', 'target'], axis=1)
        y = data['target']
        
        if X.empty:
            raise ValueError("No features available for selection.")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        feature_importances = pd.Series(importances, index=X.columns)
        top_features = feature_importances.sort_values(ascending=False).head(n_features).index.tolist()
        
        logging.info(f"Selected top {n_features} features: {top_features}")
        return top_features

    def prepare_data(self, start_date, end_date, features, strategy='default', denoise=False, n_features=5):
        """Complete pipeline to prepare data"""
        processed_data = self.process_data(start_date, end_date, features)
        
        augmented_data = self.augment_features(processed_data, strategy)
        
        if denoise:
            augmented_data = self.denoise_data(augmented_data, method='median_filter')
        
        # Feature Selection
        selected_features = self.select_features(augmented_data, n_features=n_features)
        selected_features.append('close')  # Retain 'close' for profit calculation
        augmented_data = augmented_data[selected_features]
        
        return augmented_data

    def save_scaler(self, path):
        """Save the scaler to a file"""
        joblib.dump(self.scaler, path)
        logging.info(f"Scaler saved to {path}")

    def load_scaler(self, path):
        """Load the scaler from a file"""
        self.scaler = joblib.load(path)
        logging.info(f"Scaler loaded from {path}")
