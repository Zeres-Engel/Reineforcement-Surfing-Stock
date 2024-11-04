import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
filepath = 'data/FPT.csv'

class StockDataTransform:
    def __init__(self, filepath):
        self.stock_data = pd.read_csv(filepath)
        self.dates = self.stock_data['time']
        self.close_prices = self.stock_data['close'].values
        self.transformed_data = {}

    def discrete_wavelet_transform(self):
        # Apply DWT and get approximation coefficients
        coeffs = pywt.wavedec(self.close_prices, 'db1', level=4)
        approx = coeffs[0]
        detail = coeffs[1:]
        # Store DWT approximation and details
        self.transformed_data['dwt_approx'] = approx
        self.transformed_data['dwt_detail'] = detail
        # Reconstruct signal from approximation coefficients to get transformed close prices
        reconstructed_close_prices = pywt.waverec([approx] + [None] * (len(coeffs) - 1), 'db1')
        reconstructed_close_prices = reconstructed_close_prices[:len(self.close_prices)]
        # Save reconstructed close prices to a new DataFrame and CSV
        df_dwt = pd.DataFrame({'time': self.dates, 'close_dwt': reconstructed_close_prices})
        df_dwt.to_csv('data/FPT_close_price_DWT.csv', index=False)
        return approx, detail, df_dwt
    def plot_transforms(self):
        plt.figure(figsize=(14, 10))
        # Plot original data
        plt.subplot(3, 1, 1)
        plt.plot(self.close_prices, color='blue', label='Original Close Prices')
        plt.title("Original Stock Close Prices")
        plt.legend()


        # Plot DWT Approximation
        plt.subplot(3, 1, 3)
        plt.plot(self.transformed_data['dwt_approx'], color='green', label='DWT Approximation')
        plt.title("DWT Approximation of Stock Close Prices")
        plt.legend()

        plt.tight_layout()
        plt.show()



transformer = StockDataTransform(filepath)

dwt_approx, dwt_detail, df_dwt = transformer.discrete_wavelet_transform()

# Plot the results
transformer.plot_transforms()

print("Saved DWT-transformed close prices to 'FPT_close_price_DWT.csv'")
