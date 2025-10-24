import calendar
import datetime
import requests
from bs4 import BeautifulSoup
from utp import MySQL

"""
奇摩頁面, 獲取EPS(單位盈餘)
奇摩頁面, 獲取REVENUE(營收)
"""

domain = 'https://tw.stock.yahoo.com/quote'

# code = '1513.TW'
code = '6285.TW'


def get_end_of_month(year, month):
    # 获取该月份的总天数
    days_in_month = calendar.monthrange(year, month)[1]
    # 构造日期对象表示月底
    end_of_month = datetime.date(year, month, days_in_month)
    return end_of_month


def get_eps(code):
    try:
        path = 'eps'
        url = f"{domain}/{code}/{path}"
        print(url)
        with requests.get(url) as r:
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                data = soup.find(id='qsp-eps-table')
                result = []
                if data:
                    rows = data.find_all('li')
                    for row in rows:
                        tr = row.find(attrs={'class': 'table-row'})
                        f1 = tr.find(attrs={'class': 'D(f)'}).find('div')
                        f2 = tr.find(attrs={'class': 'Fxg(1)'}).find('span')
                        parts = f1.text.split(' ')
                        year = int(parts[0])
                        month = int(str(parts[1]).replace('Q', '')) * 3
                        date = str(MySQL.get_last_trade_date(code, year, month)[0]['price_date'])
                        if date == 'None':
                            date = str(get_end_of_month(year, month))
                        print(date, '---', f2.text)
                        # hello = {'year': year, 'quarter': quarter, 'price': f2.text}
                        data = {'date': date, 'price': f2.text}
                        result.append(data)
                return result
                # print(result)
    except Exception as e:
        return
        # print("An error occurred:", e)


def get_revenue(code):
    try:
        path = 'revenue'
        url = f"{domain}/{code}/{path}"
        print(f"Fetching URL: {url}")

        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"Failed to fetch data, status code: {r.status_code}")
            return []

        soup = BeautifulSoup(r.text, 'html.parser')
        table_body = soup.find(id='qsp-revenue-table')
        if not table_body:
            print("Revenue table not found")
            return []

        table_body = table_body.find('div', class_='table-body')
        if not table_body:
            print("Table body not found")
            return []

        result = []

        rows = table_body.find_all('li', class_='List(n)')
        for row in rows:
            # 日期
            date_div = row.find('div', class_='D(f)')
            if not date_div:
                continue
            f1 = date_div.find('div', class_='W(65px)')
            if not f1:
                continue

            parts = f1.text.strip().split('/')
            if len(parts) != 2:
                continue
            year, month = int(parts[0]), int(parts[1])

            # 價格/營收
            price_span = row.find('div', class_='Flx(a)')
            if price_span:
                price_li = price_span.find('li', class_='Jc(c)')
                price = price_li.find('span').text.strip() if price_li else None
            else:
                price = None

            # 確認日期
            last_trade = MySQL.get_last_trade_date(code, year, month)
            if last_trade and last_trade[0]['price_date']:
                date = str(last_trade[0]['price_date'])
            else:
                date = str(get_end_of_month(year, month))

            # print(f"{date} --- {price}")

            result.append({'date': date, 'price': price})

        return result

    except Exception as e:
        print(f"Error in get_revenue: {e}")
        return []
# get_revenue('1513')
