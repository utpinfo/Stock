import yfinance as yf
from datetime import datetime, timedelta
#row = yf.Ticker('6176.TW').history(start='2025-10-27', end='2025-10-28')
#print(row.to_string())
print((datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))