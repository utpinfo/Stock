import math
from datetime import datetime

import numpy as np
import pandas as pd  # 开源的数据分析和数据处理库
import seaborn as sns
from matplotlib import pyplot as plt

from utp import MySQL

decimal_place = 2
analyse_days = 90
codes = MySQL.get_stock('90',2610)  # 1513, 2301, 1513,6285,4974
# codes = MySQL.get_stock('10')
sns.set(style="whitegrid")  # 设置Seaborn默认样式

display_matplot = 1  # 折綫圖顯示(1:顯示), 找股模式(0.不顯示)
rec_days = 7  # 檢測近幾個交易日内
rec_volume = 1000  # 檢核量如果過小， 不精準
rec_stocks = []


# 正反趨勢判斷
def is_trend(seq):
    n = len(seq)
    total_increase = 0
    total_decrease = 0
    for i in range(1, n):
        weight = i / (n * (n + 1) / 2) * 2  # 越後面的權重越大
        diff = seq[i] - seq[i - 1]
        if diff > 0:
            total_increase += diff * weight
        elif diff < 0:
            total_decrease += abs(diff) * weight

    if total_increase > total_decrease:
        msg = "遞增趨勢"
        return 1, msg
    elif total_increase < total_decrease:
        msg = "遞減趨勢"
        return -1, msg
    else:
        msg = "沒有明顯趨勢"
        return 0, msg


# 相對強弱指標(RSI值越大，表示在過去一段期間，上漲機率較大；相反的，RSI值越小，表示過去一段期間，下跌機率較大)
# 大於70(過熱) 小於30(過冷)
# (公式: 上升幅度的合計 / 上升幅度的合計 + 下跌幅度的合計)
def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_rsi_divergence(data, window=5):
    rsi = calculate_rsi(data)
    rsi_diff = rsi.diff()
    is_divergence = (rsi_diff > 0) & (rsi_diff.shift(-1) < 0)
    divergence_dates = is_divergence[is_divergence == True].index

    divergence_windows = []
    for date in divergence_dates:
        end_index = data.index.get_loc(date)
        start_index = max(0, end_index - window)
        divergence_windows.append(data.iloc[start_index:end_index + 1])

    return divergence_windows


def plot_stock(stock_code, stock_name, df):
    plt.figure(figsize=(10, 5))  # 視窗大小
    plt.get_current_fig_manager().set_window_title('買賣建議')
    plt.xticks(rotation=90, fontsize=8)  # 设置 x 轴标签垂直显示和字体大小
    plt.xlabel('日期')
    plt.ylabel('價格')
    plt.xlim(len(df) - 60, len(df))
    max_price = df['close'].max() + 10
    plt.ylim(-50, max_price + 50)  # 设置 y 轴的范围，限制最高和最低值

    plt.subplots_adjust(top=0.9)  # 调整顶部边界框位置
    ratio1 = math.ceil(avg_volume / avg_price) * 2
    ratio2 = math.ceil(avg_volume / avg_price) * 4
    ratio3 = math.ceil(avg_pvr / avg_price)

    # 连接鼠标移动事件
    plt.gcf().canvas.mpl_connect('motion_notify_event', lambda event: on_mouse_move(event, df))
    plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 指定默认字体为中文宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    # plt.grid(True)  # 添加网格
    # plt.gca().set_facecolor('lightgrey')  # 修改背景颜色为浅灰色
    # 设置 x 轴范围为最后 30 个标签
    df['price_date'] = df['price_date'].astype(str)
    x = df['price_date'].values

    plt.title(f"{stock_name}({stock_code})\n指標價:{est_price} ,均價:{avg_price}")
    for idx, detail in df.iterrows():
        if df.at[idx, 'ind'] > 0:
            plt.scatter(idx, -45, marker='^', color='blue', s=100)  # 使用marker參數設置標記形狀為正三角形
        elif df.at[idx, 'ind'] < 0:
            plt.scatter(idx, -45, marker='v', color='red', s=100)  # 使用marker參數設置標記形狀為正三角形
    diff_pvr = abs((df['diff_pvr'] / ratio3)).values
    sns.lineplot(x=x, y=diff_pvr, color='blue', label='當日PVR(价格量比率,越大:價格变动与成交量的关联性越强)')
    y_avg = df['avg_price'].values
    sns.lineplot(x=x, y=y_avg, color='yellow', label='平均PVR')
    amp_pvr = df['amp_pvr'].values
    sns.lineplot(x=x, y=amp_pvr, color='red', label='波動PVR')
    volume = (df['volume'] / ratio2).values  # ratio2
    sns.lineplot(x=x, y=volume, color='#ff00ff', label='數量')
    close = df['close'].values
    sns.lineplot(x=x, y=close, color='green', label='價格', linewidth=1)
    p_5_ma = df['5_MA'].values
    p_10_ma = df['10_MA'].values
    p_15_ma = df['15_MA'].values
    sns.lineplot(x=x, y=p_5_ma, color='purple', label='5日均價', linewidth=1)
    sns.lineplot(x=x, y=p_10_ma, color='#00e4ff', label='10日均價', linewidth=1)
    sns.lineplot(x=x, y=p_15_ma, color='#ff9200', label='15日均價', linewidth=1)
    v_5_ma = (df['5_V_MA'] / ratio1).values
    v_10_ma = (df['10_V_MA'] / ratio1).values
    v_15_ma = (df['15_V_MA'] / ratio1).values
    sns.lineplot(x=x, y=v_5_ma, color='#EE82EE', label='5日均量', linewidth=1, linestyle='dashed')
    sns.lineplot(x=x, y=v_10_ma, color='#87CEEB', label='10日均量', linewidth=1, linestyle='dashed')
    sns.lineplot(x=x, y=v_15_ma, color='#1E90FF', label='15日均量', linewidth=1, linestyle='dashed')
    sns.lineplot(x=x, y=df['RSI'] - 100, color='#8B0000', label='RSI (>70(過熱) <30(過冷))', linewidth=1,
                 linestyle='dashed')  # 下移100, 顯示比較明顯
    # print(price_15_ma)
    # plt.tight_layout()  # 调整子图布局
    # begin MACD 藍轉紅(買）
    ax = plt.gca()

    df['date'] = pd.to_datetime(df['price_date'])
    df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    ema1 = df["close"].ewm(span=12, adjust=False).mean() * 5  # 放大10倍顯示
    ema2 = df["close"].ewm(span=26, adjust=False).mean() * 5  # 放大10倍顯示
    DIF = (ema1 - ema2)  # 短期長期差
    DEA = DIF.ewm(span=9, adjust=False).mean()  # 9日移動平均
    macd_point = 2 * (DIF - DEA)
    positive_macd = np.where(macd_point > 0, macd_point, 0)  # 90日的指數移動平均集合
    negative_macd = np.where(macd_point < 0, macd_point, 0)  # 90日的指數移動平均集合
    ax.plot(np.arange(0, len(df)), DIF, label='DIF快綫', linewidth=3, linestyle='dashed')
    ax.plot(np.arange(0, len(df)), DEA, label='DEA慢綫', linewidth=3, linestyle='dashed')
    # print(len(df)) #筆數:90
    # 0開始到90,間距為
    ax.bar(np.arange(0, len(df)), positive_macd, color="red")
    ax.bar(np.arange(0, len(df)), negative_macd, color="blue")
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(20)) #创建了一个 MaxNLocator 对象，用于自动计算主刻度的位置，使得 x 轴上的主刻度数量最多为 20 个
    '''
    def format_date(x, pos=None):
        if x < 0 or x > len(df['date']) - 1:
            return ''
        return df['date'][int(x)]
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    '''
    # end MACD
    plt.legend(loc='upper left', fontsize=8)  # 设置图例字体属性
    plt.show()


# 定义鼠标移动事件处理程序
def on_mouse_move(event, data):
    if not event.inaxes:
        return
    ax = plt.gca()
    # 清理指示线
    for line in ax.lines[13:]:  # 从第4个线条开始是指示线
        line.remove()
    # 清理右上角顯示文字
    for text in ax.texts:
        text.remove()

    idx = int(event.xdata)
    # 右上角顯示文字
    if idx >= 0 and idx < len(data):
        cur_date = data.at[idx, 'price_date']
        diff_pvr = data.at[idx, 'diff_pvr']
        cur_price = data.at[idx, 'close']
        cur_volume = data.at[idx, 'volume']
        # 绘制新的指示线
        ax.axhline(y=event.ydata, color='gray', linestyle='--')
        ax.axvline(x=event.xdata, color='gray', linestyle='--')
        # 上方壓力計算
        press_top_price = 0
        press_top_data = df[(df['ind'] != 0) & (df['close'] > cur_price) & (df['price_date'] < cur_date)]
        if len(press_top_data) > 0:
            press_top_price = press_top_data.iloc[-1]['close']

        press_low_data = df[(df['ind'] != 0) & (df['close'] < cur_price) & (df['price_date'] < cur_date)]
        press_low_price = 0
        if len(press_low_data) > 0:
            press_low_price = press_low_data.iloc[-1]['close']

        msg = f'日期: {cur_date}\nPVR:{diff_pvr}, 價格:{cur_price}, 數量:{cur_volume}\n(壓力:{press_top_price} 支撐:{press_low_price})'
        plt.text(0.95, 0.95, msg, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
                 color='red', fontsize=10)
        plt.title(
            f"{stock_name}({stock_code})\n指標價:{est_price} ,均價:{avg_price} 價格:{cur_price}, 數量:{cur_volume}(壓力:{press_top_price} 支撐:{press_low_price}）")
    plt.draw()


for idx, master in enumerate(codes):
    stock_code = master.get('stock_code')
    stock_name = master.get('stock_name')
    details = MySQL.get_price(stock_code, analyse_days, 'asc')
    if not details:
        continue
    tgt_price = []
    tol_pvr, tol_cnt, tol_price, tol_volume = 0, 0, 0, 0
    volume_after_extra_prv = []

    # data_with_str_dates = [{**item, 'price_date': str(item['price_date'])} for item in details]
    # print(details)
    df = pd.DataFrame(details)  # 二維表格操作（類似 Excel）
    # 一次性初始化所有行
    df['diff_pvr'] = 0
    df['avg_pvr'] = 0
    df['amp_pvr'] = 0
    df['ind'] = 0
    df['volume'] = df['volume'] / 1000
    df['result'] = ""
    for idx, detail in df.iterrows():
        tol_cnt = tol_cnt + 1
        cur_date = str(df.at[idx, 'price_date'])
        cur_price = float(df.at[idx, 'close'])
        cur_volume = df.at[idx, 'volume']
        tol_price = tol_price + cur_price
        tol_volume = tol_volume + cur_volume
        avg_price = round(tol_price / tol_cnt, decimal_place)
        avg_volume = round(tol_volume / tol_cnt, decimal_place)

        # 取得T-1數據
        if idx > 0:
            pre_price = float(df.at[idx - 1, 'close'])
            pre_volume = df.at[idx - 1, 'volume']
            pre_avg_pvr = df.at[idx - 1, 'avg_pvr']
        else:
            pre_price = cur_price
            pre_volume = cur_volume
            pre_avg_pvr = 0

        # 差價
        diff_price = round(cur_price - pre_price, decimal_place)
        # 差量
        diff_volume = round(cur_volume - pre_volume, decimal_place)
        diff_pvr = 0
        if diff_volume != 0:
            # 差異價量比(正數比例越大:刻意拉抬, 負數比例越大:刻意下壓),(比例越小:無波瀾))
            diff_pvr = round(diff_price / (diff_volume / 10000), decimal_place)
        # 合計波動價量比
        tol_pvr = tol_pvr + abs(diff_pvr)
        # 平均波動價量比
        avg_pvr = round(tol_pvr / tol_cnt, decimal_place)
        # 震蕩幅度
        if avg_pvr:
            amp_pvr = round(diff_pvr / avg_pvr, decimal_place)
        else:
            amp_pvr = 0

        # 震動參數,越高越好
        if abs(diff_pvr) > abs(avg_pvr):
            tgt_price.append(cur_price)

        # 詞典新增字段
        df.at[idx, 'amp_pvr'] = diff_volume / 1000000
        df.at[idx, 'avg_price'] = avg_price
        df.at[idx, 'diff_pvr'] = diff_pvr
        df.at[idx, 'avg_pvr'] = avg_pvr
        df.at[idx, 'amp_pvr'] = amp_pvr
        df.at[idx, 'ind'] = 0
        df.at[idx, 'DIF'] = 0
        df.at[idx, 'DEA'] = 0

        df.at[idx, '5_MA'] = round(df['close'].iloc[idx - 4:idx + 1].mean(), decimal_place)  # 计算五日均线(含T日)
        df.at[idx, '10_MA'] = round(df['close'].iloc[idx - 9:idx + 1].mean(), decimal_place)  # 计算十日均线(含T日)
        df.at[idx, '15_MA'] = round(df['close'].iloc[idx - 14:idx + 1].mean(), decimal_place)  # 计算十五日均线(含T日)
        df.at[idx, '5_V_MA'] = round(df['volume'].iloc[idx - 4:idx + 1].mean(), decimal_place)  # 计算五日均线(含T日)
        df.at[idx, '10_V_MA'] = round(df['volume'].iloc[idx - 9:idx + 1].mean(), decimal_place)  # 计算十日均线(含T日)
        df.at[idx, '15_V_MA'] = round(df['volume'].iloc[idx - 14:idx + 1].mean(), decimal_place)  # 计算十五日均线(含T日)
        df.at[idx, '5_E_MA'] = round(abs(df['diff_pvr']).iloc[idx - 4:idx + 1].mean(), decimal_place)  # 计算五日均线(含T日)
        df.at[idx, '10_E_MA'] = round(abs(df['diff_pvr']).iloc[idx - 9:idx + 1].mean(), decimal_place)  # 计算十日均线(含T日)
        df.at[idx, '15_E_MA'] = round(abs(df['diff_pvr']).iloc[idx - 14:idx + 1].mean(), decimal_place)  # 计算十五日均线(含T日)

        # RSI
        # 計算價格變化
        df.at[idx, 'RSI'] = 0
        rsi = calculate_rsi(df[:idx + 1])
        df.at[idx, 'RSI'] = rsi[idx]
        # RSI

        # begin (MACD計算)
        ema1 = df["close"].ewm(span=12, adjust=False).mean() * 5  # 放大10倍顯示
        ema2 = df["close"].ewm(span=26, adjust=False).mean() * 5  # 放大10倍顯示
        DIF = (ema1 - ema2)  # 短期長期差
        DEA = DIF.ewm(span=9, adjust=False).mean()  # 9日移動平均
        MACD = 2 * (DIF - DEA)
        df.at[idx, 'DIF'] = DIF[idx]
        df.at[idx, 'DEA'] = DEA[idx]
        df.at[idx, 'MACD'] = MACD[idx]
        df.loc[idx, 'MACD'] = MACD[idx]  # pandas列表立即追加當前計算的MACD, 含當日
        macd_5_days = df['MACD'].iloc[idx - 4:idx]
        # macd_5_diff = macd_5_days.diff()
        # macd_5_diff = macd_5_diff.iloc[1:]  # 去除第一笔数据
        # macd_5_diff.loc[idx] = df.at[idx,'MACD']  # 將當日計算的MACD, 加入計算
        # print(macd_5_diff)
        # 计算移动平均
        # window_size = 5  # 移动平均窗口大小
        # moving_avg = np.convolve(macd_5_diff, np.ones(window_size) / window_size, mode='valid')
        # 判断移动平均是否递增或递减
        macd_msg = ""
        df.at[idx, 'MACD_DIRECTION'] = 0
        if 1 > df.at[idx, 'MACD'] > -1:
            if idx > 0:
                pre_dif = df.at[idx - 1, 'DIF']
                pre_dea = df.at[idx - 1, 'DEA']
                if pre_dif - pre_dea > 0:
                    df.at[idx, 'MACD_DIRECTION'] = -1
                else:
                    df.at[idx, 'MACD_DIRECTION'] = 1
            macd_msg = f"日期:{cur_date}, 近5日单价波动(但MACD趨近0)"
        else:
            macd_5_days = macd_5_days.values
            print(macd_5_days)
            df.at[idx, 'MACD_DIRECTION'], macd_msg = is_trend(macd_5_days)  # 計算前5日

        # end (MACD計算)
        # if abs(amp_pvr) > 3:
        # if abs(diff_pvr) > abs(avg_pvr):
        if abs(diff_pvr) > abs(df.at[idx, '15_E_MA'] * 3.1):
            # T:有能無量， T+1:有量則漲
            if df.at[idx, 'MACD_DIRECTION'] > 0:
                if round(df.at[idx, 'MACD']) > 2 and df.at[idx, 'DIF'] < df.at[idx, 'DEA']:
                    df.at[idx, 'result'] = "* 0買入時機"
                    df.at[idx, 'ind'] = 1
                    volume_after_extra_prv = []
                    volume_after_extra_prv.append(cur_volume)
                elif round(df.at[idx, 'MACD']) <= 2 and df.at[idx, '5_V_MA'] < (
                        df.at[idx, '10_V_MA'] * 1.5):  # 前期量減(偏差值0.2)
                    df.at[idx, 'result'] = "* 1買入時機"
                    df.at[idx, 'ind'] = 1
                    volume_after_extra_prv = []
                    volume_after_extra_prv.append(cur_volume)
                elif df.at[idx, '5_V_MA'] < (df.at[idx, '15_V_MA']):  # 前期量減(偏差值0.2)
                    df.at[idx, 'result'] = "* 2買入時機"
                    df.at[idx, 'ind'] = 1
                    volume_after_extra_prv = []
                    volume_after_extra_prv.append(cur_volume)
                elif round(df.at[idx, 'MACD']) > 2 and df.at[idx, 'DIF'] > df.at[idx, 'DEA']:
                    df.at[idx, 'result'] = "* 11賣出時機(1.拉高出貨)"
                    df.at[idx, 'ind'] = -1
                elif cur_volume > (df.at[idx, '15_V_MA'] * 3):
                    df.at[idx, 'result'] = "* 3賣出時機(1.拉高出貨)"
                    df.at[idx, 'ind'] = -1
            else:
                if df.at[idx, '5_V_MA'] < (df.at[idx, '15_V_MA']):  # 前期量減(偏差值0.2)
                    df.at[idx, 'result'] = "* 2買入時機"
                    df.at[idx, 'ind'] = 1
                    volume_after_extra_prv = []
                    volume_after_extra_prv.append(cur_volume)
                elif round(df.at[idx, 'MACD']) > 2 and df.at[idx, 'DIF'] > df.at[idx, 'DEA']:
                    df.at[idx, 'result'] = "* 3賣出時機(1.拉高出貨)"
                    df.at[idx, 'ind'] = -1
                elif cur_volume > (df.at[idx, '15_V_MA'] * 3):
                    df.at[idx, 'result'] = "* 4賣出時機(1.拉高出貨)"
                    df.at[idx, 'ind'] = -1
            current_date = datetime.now()
            # 计算日期差
            date_difference = current_date - datetime.strptime(str(cur_date), '%Y-%m-%d')
            if date_difference.days <= rec_days:
                if avg_volume >= rec_volume:
                    if df.at[idx, 'ind'] == 1:
                        df.at[idx, 'result'] = f"股票:{stock_code},{rec_days}日内存在買入時機"
                        stock_exists = any(stock['stock_code'] == stock_code for stock in rec_stocks)
                        if not stock_exists:
                            rec_stocks.append(
                                {'stock_code': stock_code, 'stock_name': stock_name, 'volume': cur_volume})
        else:
            # 出現大能後, 記錄量
            volume_after_extra_prv.append(cur_volume)
            if cur_volume <= (sum(volume_after_extra_prv) / len(volume_after_extra_prv)) * 0.5:
                # 量小於15均量, 則能量耗盡:看空
                df.at[idx, 'result'] = "* 賣出時機(2.拉高出貨)" + df.at[idx, 'result']
                df.at[idx, 'ind'] = -1
        print("######################################################################################")
        if df.at[idx, 'result'] != "":
            df.at[idx, 'result'] = f"{df.at[idx, 'result']}, "
        df.at[idx, 'result'] = f"{df.at[idx, 'result']}日期:{cur_date}, 股票:{stock_code}"
        print(df.at[idx, 'result'])
        df.at[idx, 'result'] = f"日期:{cur_date}, RSI:{df.at[idx, 'RSI']} PVR(波動):{amp_pvr}"
        print(df.at[idx, 'result'])
        df.at[
            idx, 'result'] = f"日期:{cur_date}, 動能:{diff_pvr} (5均能:{df.at[idx, '5_E_MA']}, 10均能:{df.at[idx, '10_E_MA']}, 15均能:{df.at[idx, '15_E_MA']})"
        print(df.at[idx, 'result'])
        df.at[
            idx, 'result'] = f"日期:{cur_date}, 股價:{cur_price}, 差價:{diff_price} (5均價:{df.at[idx, '5_MA']}, 10均價:{df.at[idx, '10_MA']}, 15均價:{df.at[idx, '15_MA']})"
        print(df.at[idx, 'result'])
        df.at[
            idx, 'result'] = f"日期:{cur_date}, 股量:{cur_volume}, 差量:{diff_volume}, (5均量:{df.at[idx, '5_V_MA']}, 10均量:{df.at[idx, '10_V_MA']}, 15均量:{df.at[idx, '15_V_MA']})"
        print(df.at[idx, 'result'])
        # print(f"日期:{cur_date}, 當量:{cur_volume}，均量{sum(volume_after_extra_prv) / len(volume_after_extra_prv)}")
        df.at[idx, 'result'] = f"日期:{cur_date}, 趨勢:{macd_msg}"
        print(df.at[idx, 'result'])
        df.at[
            idx, 'result'] = f"日期:{cur_date}, MACD:{df.at[idx, 'MACD']}, DIF:{df.at[idx, 'DIF']}, DEA:{df.at[idx, 'DEA']}, 方向{df.at[idx, 'MACD_DIRECTION']}"
        print(df.at[idx, 'result'])
    if tol_cnt > 0:
        if tgt_price:
            est_price = round(sum(tgt_price) / len(tgt_price), decimal_place)
        if est_price:
            print(f"股票:{stock_code}, 指標價:{est_price} ,均價:{avg_price}, 當前股價:{cur_price}")
            print("######################################################################################")
    print(df.to_string())
    if display_matplot == 1:
        # 繪製圖案
        plot_stock(stock_code, stock_name, df)
print(rec_stocks)  # 指標股票
