from datetime import datetime, timedelta

import requests

domain = "https://api.fugle.tw"
route = "marketdata/v1.0/stock"
appKey = "NGNmMTg5MjEtZGM0MS00MTNmLTgwMzctMjdmOWU3MDU4OGM1IDAwOTQ1OTQ2LWY5NDgtNDViNi05NDJlLWRhMTRiNjdkNDgzMA"


def quote(code):
    global url
    method = "intraday/quote"
    url = f"{domain}/{route}/{method}/{code}"
    head = {'X-API-KEY': appKey}
    r = requests.get(url, headers=head)
    if r.status_code == 200:
        return r.text
    else:
        errMsg = f"錯誤代碼: {r.status_code}，錯誤股票:{code}"
        # print(errMsg)
        return


def candles(code, b_date=None, e_date=None):
    global url
    dateFormat = "%Y-%m-%d"
    if not e_date:
        e_date = datetime.today() - timedelta(days=1)
    if not b_date:
        b_date = e_date - timedelta(days=1)

    method = "historical/candles"
    url = f"{domain}/{route}/{method}/{code}"
    head = {'X-API-KEY': appKey}
    param = {'from': b_date.strftime(dateFormat), 'to': e_date.strftime(dateFormat), 'timeframe': 'D',
             'fields': 'volume,close'}
    print('股票代碼:', code, ',日期區間:', b_date.strftime(dateFormat), '~', e_date.strftime(dateFormat))
    # print(url)
    r = requests.get(url, headers=head, params=param)
    if r.status_code == 200:
        return r.text
    else:
        errMsg = f"錯誤代碼: {r.status_code}，錯誤股票:{code}"
        print(errMsg)
        return
