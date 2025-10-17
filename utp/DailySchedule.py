import threading
import time

import MySQL
from utp.Schedule import DailySchedule

threads = 300


def get_daily_schedule(stock_code, isin_code):
    DailySchedule(stock_code, isin_code)


# 獲取股票代碼
# codes = MySQL.get_stock('00','3042')
codes = MySQL.get_stock(None)
# 將股票代碼分成10組
chunk_size = len(codes) // threads + (len(codes) % threads > 0)
chunks = [codes[i:i + chunk_size] for i in range(0, len(codes), chunk_size)]

# 用線程處理每一組股票代碼
threads = []
for chunk in chunks:
    for code in chunk:
        thread = threading.Thread(target=get_daily_schedule, args=(code['stock_code'], code['isin_code'],))
        thread.start()
        threads.append(thread)
        time.sleep(0.5)  # 在发送每个请求之间增加 0.5 秒的延迟

# 等待所有線程結束
for thread in threads:
    thread.join()
