import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

class Augmentation:
    def __init__(self, df, price_type="open"):
        self.df = df
        self.price_type = price_type  # Biến để chọn cột giá cần thao tác
    
    def augment_with_Gaussian(self, std_range=(0.01, 0.03)):
        df = self.df.copy()
        std_dev = np.random.uniform(std_range[0], std_range[1])
        noise = np.random.normal(0, std_dev, size=df[self.price_type].shape)
        df[f"{self.price_type}_with_noise"] = df[self.price_type] + noise
        return df

    def time_stretch(self, rate=1.5):
        x_original = np.arange(len(self.df))
        x_new = np.linspace(0, len(self.df) - 1, int(len(self.df) * rate))
        interpolator = interp1d(x_original, self.df[self.price_type], kind='linear')
        stretched_prices = interpolator(x_new)
        
        return pd.DataFrame({
            'Date': pd.date_range(start=self.df['time'].min(), periods=len(stretched_prices), freq='D'),
            f'{self.price_type}_Stretched': stretched_prices
        })
    
    def random_cutout(self, missing_days=3):
        data_copy = self.df.copy()
        start_idx = np.random.randint(0, len(self.df) - missing_days)
        data_copy.loc[start_idx:start_idx + missing_days - 1, self.price_type] = np.nan
        return data_copy

    def seasonal_decomposition(self, period=12):
        decomposition = seasonal_decompose(self.df[self.price_type], model="additive", period=period)
        trend_part = decomposition.trend
        seasonal_part = decomposition.seasonal
        residual_part = decomposition.resid
        
        plt.figure(figsize=(12, 8))
        plt.subplot(4, 1, 1)
        plt.plot(self.df[self.price_type], label=f"Giá {self.price_type}")
        plt.legend(loc="upper left")
        plt.subplot(4, 1, 2)
        plt.plot(trend_part, label="Xu hướng", color="orange")
        plt.legend(loc="upper left")
        plt.subplot(4, 1, 3)
        plt.plot(seasonal_part, label="Mùa vụ", color="green")
        plt.legend(loc="upper left")
        plt.subplot(4, 1, 4)
        plt.plot(residual_part, label="Nhiễu", color="red")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
        
        return trend_part, seasonal_part, residual_part

# Đoạn code mẫu để chạy thử
df = pd.read_csv("data/ACB_VCI_2024-06-01_2024-10-28.csv")
augmenter = Augmentation(df, price_type="open")

# Thêm nhiễu Gaussian vào giá open
augmented_df = augmenter.augment_with_Gaussian()
print("Gaussian augment")
print(augmented_df.head())

# Thực hiện co giãn thời gian
stretched_df = augmenter.time_stretch(rate=1.5)
print("Stretched Augment")
print(stretched_df.describe())

# Áp dụng cắt bỏ ngẫu nhiên một phần dữ liệu
cutout_df = augmenter.random_cutout(missing_days=5)
print("Cut out Augment")
print(cutout_df.describe())

# Phân tích mùa vụ
trend, seasonal, residual = augmenter.seasonal_decomposition(period=12)
