from datetime import datetime, timedelta

import yfinance as yf

import MySQL
import YahooStockInfo


# 獲取3年股價/成交量
def DailySchedule(stock_code, isin_code, start_date=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")):
    # start_date = "2023-03-01"
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    data = MySQL.get_price(stock_code, None, 'asc', start_date, end_date)
    msg = f"stock_code:{stock_code}, isin_code:{isin_code}, start_date:{start_date}, end_date:{end_date}"

    if len(data) > 0:
        msg += " (DB已存在)"
        print(msg)
        return
    else:
        msg += " (API調用)"
        print(msg)

    rows = yf.Ticker(isin_code).history(start=start_date, end=end_date)
    for date, row in rows.iterrows():
        price_date = str(date.date())
        # print(f"日期: {price_date}, 收盤價: {round(row['Close'], 2)}, 成交量: {row['Volume']}")
        MySQL.add_price(stock_code, price_date, row['Close'], row['Volume'])
    '''
    for i in range(t):
        year = datetime.now().year - i
        b_date = datetime(year, 1, 1)
        e_date = datetime(year, 12, 31)
        r = FugleUtils.candles(code, b_date, e_date)
        if r:
            data = json.loads(r)
            rows = data['data']
            for row in rows:
                MySQL.add_price(code, row['date'], row['close'], row['volume'])
    '''


def MonthlySchedule(code):
    # r = YahooStockInfo.get_revenue(code)
    rows = YahooStockInfo.get_revenue(code)
    if rows:
        for row in rows:
            MySQL.add_revenue(code, row['date'], row['price'].replace(",", ""))


def QuarterlySchedule(code):
    # r = YahooStockInfo.get_revenue(code)
    rows = YahooStockInfo.get_eps(code)
    if rows:
        for row in rows:
            MySQL.add_eps(code, row['date'], row['price'].replace(",", ""))
