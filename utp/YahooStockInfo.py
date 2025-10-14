import calendar
import datetime

import requests
from bs4 import BeautifulSoup

from utp import MySQL

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
        print(url)
        with requests.get(url) as r:
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                data = soup.find(id='qsp-revenue-table').find('div', class_='table-body')
                result = []
                if data:
                    rows = data.find_all('li', class_='List(n)')
                    for row in rows:
                        f1 = row.find('div', class_='D(f)').find('div', class_='W(65px)')
                        f2 = row.find('div', class_='Flx(a)').find('li', class_='Jc(c)').find('span')
                        parts = f1.text.split('/')
                        year = int(parts[0])
                        month = int(parts[1])
                        date = str(MySQL.get_last_trade_date(code, year, month)[0]['price_date'])
                        if date == 'None':
                            date = str(get_end_of_month(year, month))
                        print(date, '---', f2.text)
                        # hello = {'year': year, 'quarter': quarter, 'price': f2.text}
                        data = {'date': date, 'price': f2.text}
                        result.append(data)
                # print(result)
                return result
    except Exception as e:
        return
        # print("An error occurred:", e)

# get_revenue('1513')
