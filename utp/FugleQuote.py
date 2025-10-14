import json

import FugleUtils

codes = [{"stockNo": "1513", "stockName": "中興電"}, {"stockNo": "6176", "stockName": "瑞儀"},
         {"stockNo": "6285", "stockName": "啟碁"}]


def getQuote():
    for code in codes:
        r = FugleUtils.quote(code.get('stockNo'))
        if r:
            data = json.loads(r)
            print(f"股票代號:{code.get('stockNo')},最後一筆成交：{data['type']}")


def getCandle():
    for code in codes:
        volumes = 0
        prices = 0
        r = FugleUtils.candles(code.get('stockNo'))
        if r:
            data = json.loads(r)
            data = data['data']
            cnt = len(data)
            index = 0
            for date in data:
                index += 1
                volumes += date['volume']
                prices += date['close']
                if index == 1:
                    volume = round(volumes / 1000)
                    price = prices
            # 股數轉張數
            volumes_all = round(volumes / 1000)
            # 日均量
            volumes_avg = round(volumes_all / cnt)
            prices_all = prices
            prices_avg = round(prices / cnt)
            # T-1

            print(
                f"股票名稱:{code.get('stockName')}, 股票代號:{code.get('stockNo')}, 合計成交量：{volumes_all}張, 昨日成交量：{volume}張, 日均成交量：{volumes_avg}張, 日均價:{prices_avg}, 昨日價：{price}")
            price = round(price / prices_avg - 1, 2)
            volume = round(volume / volumes_avg - 1, 2)
            print(f"價格波動{price}%, 數量波動{volume}%")
            print("-------------------------------------")
            ##計算昨日出現, 成交量 > 日均量

# getCandle()
