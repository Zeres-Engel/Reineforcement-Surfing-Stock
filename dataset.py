from vnstock3 import Vnstock 
import os

folder_path = "data"
Stock_symbol = "ACB"
source = "VCI"
start_date = "2024-06-01"
end_date = "2024-10-28"

stock = Vnstock().stock(symbol=Stock_symbol, source=source) # Định nghĩa biến vnstock lưu thông tin mã chứng khoán & nguồn dữ liệu bạn sử dụng
df = stock.quote.history(start=start_date, end=end_date, interval='1D') # Thiết lập thời gian tải dữ liệu và khung thời gian tra cứu là 1 ngày
csv_path = f"{Stock_symbol}_{source}_{start_date}_{end_date}.csv"
file_path = os.path.join(folder_path,csv_path)
df.to_csv(file_path,index = False)

