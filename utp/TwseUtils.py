from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utp import MySQL

base_url = 'https://isin.twse.com.tw/isin/C_public.jsp'
# 获取台湾股票列表数据的网址
"""
"https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",  # 上市證券
"https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",  # 上櫃證券
"https://isin.twse.com.tw/isin/C_public.jsp?strMode=5"  # 興櫃證券
"""


# 发送请求并获取数据
def get_stock_list(url):
    r = requests.get(url)
    if r.status_code != 200:
        return "下載異常！"

    # 直接用 pandas.read_html 解析整個 HTML
    tables = pd.read_html(r.text)
    print(tables)
    if len(tables) == 0:
        return "找不到表格！"

    data = tables[0].iloc[2:]  # 從第2列開始（跳過標題）

    for index, row in data.iterrows():
        if "上市認購(售)權證" in row.values:
            break
        c1 = row[0]
        stock_kind = row[3]
        if stock_kind in ("上市", "上櫃"):
            parts = str(row[0]).replace("\u5168", "\u3000").split("\u3000", 1)
            stock_code = parts[0]
            stock_name = parts[1]
            isin_no = row[1]
            MySQL.add_stock(stock_code, stock_name, stock_kind, isin_no)

    return "下載完成！"


if __name__ == "__main__":
    # get_stock_list()
    stock_kind = [2, 4]
    for row in stock_kind:
        url = f"{base_url}?strMode={str(row)}"
        get_stock_list(url)
