from datetime import datetime
import pandas_ta as ta
import seaborn as sns
from matplotlib import pyplot as plt

from utp import MySQL

"""
expanding: 行累積合計(階段合計)
"""
decimal_place = 2
analyse_days = 90
codes = MySQL.get_stock('90')  # 股票列表
sns.set(style="whitegrid")

display_matplot = 1  # 是否顯示圖表
rec_days = 7  # 最近幾日檢查
rec_volume = 1000  # 最小成交量
rec_stocks = []  # 記錄符合條件股票


# ===================== 向量化計算 RSI [简单移动平均（SMA）]=====================
def calculate_rsi_sma(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


# ===================== 向量化計算 RSI [ Wilder's smoothing（平滑移动平均）]=====================
import pandas as pd
import numpy as np


def calculate_rsi_wilder(df, column='price', period=14):
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rsi = pd.Series(np.nan, index=df.index)

    # 第一個有效值
    if len(df) > period:
        rsi.iloc[period] = 100 - 100 / (1 + (avg_gain.iloc[period] / avg_loss.iloc[period]))

        # Wilder 遞推
        for i in range(period + 1, len(df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
            rs = avg_gain.iloc[i] / avg_loss.iloc[i]
            rsi.iloc[i] = 100 - (100 / (1 + rs))

    return rsi


# ===================== 向量化計算 MACD =====================
def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    DIF = ema12 - ema26
    DEA = DIF.ewm(span=9, adjust=False).mean()
    MACD = 2 * (DIF - DEA)
    return DIF, DEA, MACD


# ===================== 計算各種均線 =====================
def calculate_moving_averages(df):
    for p in [5, 10, 15]:
        df[f'{p}_MA'] = df['price'].rolling(p).mean()
        df[f'{p}_V_MA'] = df['volume'].rolling(p).mean()
        df[f'{p}_E_MA'] = df['diff_pvr'].abs().rolling(p).mean()
    return df


# ===================== 趨勢判斷 =====================
def is_trend(seq):
    n = len(seq)
    total_increase = 0
    total_decrease = 0
    for i in range(1, n):
        weight = i / (n * (n + 1) / 2) * 2
        diff = seq[i] - seq[i - 1]
        if diff > 0:
            total_increase += diff * weight
        else:
            total_decrease += abs(diff) * weight
    if total_increase > total_decrease:
        return 1, "遞增趨勢"
    elif total_increase < total_decrease:
        return -1, "遞減趨勢"
    else:
        return 0, "沒有明顯趨勢"


# 定义鼠标移动事件处理程序
def on_mouse_move(event, df, ax1, ax2, ax3, ax4, stock_code, stock_name, est_price, avg_price):
    """滑鼠移動事件：顯示游標位置資訊"""
    if event.inaxes is None:
        return

    # 初始化列表
    if not hasattr(ax1, '_indicator_lines'):
        ax1._indicator_lines = []
    if not hasattr(ax1, '_indicator_texts'):
        ax1._indicator_texts = []

    # 只在價格圖 ax1 上顯示指示線和文字
    if event.inaxes != ax1:
        # 隱藏舊指示線
        for l in ax1._indicator_lines:
            try:
                l.remove()
            except Exception:
                pass
        ax1._indicator_lines = []

        for t in ax1._indicator_texts:
            try:
                t.remove()
            except Exception:
                pass
        ax1._indicator_texts = []

        event.canvas.draw_idle()
        return

    # 清除舊指示線與文字
    for l in ax1._indicator_lines:
        try:
            l.remove()
        except Exception:
            pass
    ax1._indicator_lines = []

    for t in ax1._indicator_texts:
        try:
            t.remove()
        except Exception:
            pass
    ax1._indicator_texts = []

    # 畫新指示線
    idx = int(event.xdata)
    if idx < 0 or idx >= len(df):
        return

    cur = df.iloc[idx]
    cur_date = cur['price_date']
    cur_price = cur['price']
    cur_volume = cur['volume']
    diff_pvr = cur['diff_pvr']

    # 壓力/支撐
    press_top_data = df[(df['ind'] != 0) & (df['price'] > cur_price) & (df['price_date'] < cur_date)]
    press_low_data = df[(df['ind'] != 0) & (df['price'] < cur_price) & (df['price_date'] < cur_date)]
    press_top_price = press_top_data.iloc[-1]['price'] if len(press_top_data) > 0 else 0
    press_low_price = press_low_data.iloc[-1]['price'] if len(press_low_data) > 0 else 0

    # 畫指示線
    hline = ax1.axhline(y=event.ydata, color='gray', linestyle='--', alpha=0.6)
    vline = ax1.axvline(x=event.xdata, color='gray', linestyle='--', alpha=0.6)
    ax1._indicator_lines = [hline, vline]

    # 顯示文字
    msg = (f"日期: {cur_date}\n"
           f"PVR: {diff_pvr:.2f}\n"
           f"價格: {cur_price:.2f}, 量: {cur_volume}\n"
           f"(壓力: {press_top_price:.2f} 支撐: {press_low_price:.2f})")
    text = ax1.text(0.98, 0.98, msg, ha='right', va='top', transform=ax1.transAxes,
                    color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    ax1._indicator_texts = [text]

    # 更新標題
    ax1.set_title(
        f"{stock_name}({stock_code}) | 指標價:{est_price} 均價:{avg_price} 價:{cur_price} 量:{cur_volume} "
        f"(壓:{press_top_price} 支:{press_low_price})"
    )

    event.canvas.draw_idle()

# ===================== 畫圖 =====================
from matplotlib.gridspec import GridSpec


def plot_stock(stock_code, stock_name, df, est_price, avg_price):
    fig = plt.figure(figsize=(12, 10))
    plt.get_current_fig_manager().set_window_title('買賣建議')
    plt.rcParams['font.sans-serif'] = ['Heiti TC']

    # GridSpec: 四層圖
    gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])  # 價格圖
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # 成交量
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # RSI
    ax4 = fig.add_subplot(gs[3], sharex=ax1)  # MACD

    x = df['price_date'].astype(str).values

    # 上方價格圖
    sns.lineplot(x=x, y=df['price'], color='red', label='價格', linewidth=1, ax=ax1)
    sns.lineplot(x=x, y=df['amp_pvr'], color='blue', label='波動PVR', linewidth=1, ax=ax1)
    for p in [5, 10, 15]:
        sns.lineplot(x=x, y=(df[f'{p}_MA']), label=f'{p}日均價', ax=ax1, linestyle='dashed')

    # 標記買賣訊號
    for idx, detail in df.iterrows():
        if df.at[idx, 'ind'] > 0:
            ax1.scatter(idx, df['price'].min() * 0.99, marker='^', color='blue', s=80)
        elif df.at[idx, 'ind'] < 0:
            ax1.scatter(idx, df['price'].min() * 0.99, marker='v', color='red', s=80)

    # 成交量圖
    ax2.bar(df.index, df['volume'], color='#ff00ff', alpha=0.6, width=0.8, label='成交量')
    ax2.set_ylabel('成交量')

    # RSI圖
    ax3.plot(df.index, df['RSI'], color='purple', label='RSI')
    ax3.axhline(70, color='red', linestyle='dashed', linewidth=0.7)
    ax3.axhline(30, color='green', linestyle='dashed', linewidth=0.7)
    ax3.set_ylabel('RSI')

    # MACD圖
    ax4.bar(df.index, df['MACD'].where(df['MACD'] > 0, 0), color='red', alpha=0.6, width=0.8)
    ax4.bar(df.index, df['MACD'].where(df['MACD'] < 0, 0), color='blue', alpha=0.6, width=0.8)
    ax4.set_ylabel('MACD')
    ax4.set_xlabel('日期')

    # 隱藏上方子圖 x 標籤
    ax1.xaxis.set_tick_params(labelbottom=False)
    ax2.xaxis.set_tick_params(labelbottom=False)
    ax3.xaxis.set_tick_params(labelbottom=False)
    ax4.set_xticks(df.index)
    ax4.set_xticklabels(df['price_date'].astype(str), rotation=90, fontsize=8)

    # 合併圖例
    lines = []
    labels = []
    for ax in [ax1, ax2, ax3, ax4]:
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, fontsize=8, loc='upper left')

    # 滑鼠互動事件
    fig.canvas.mpl_connect(
        'motion_notify_event',
        lambda event: on_mouse_move(event, df, ax1, ax2, ax3, ax4, stock_code, stock_name, df['est_price'].iloc[-1],
                                    df['avg_price'].iloc[-1])
    )

    # 手動調整間距
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.07, right=0.95, hspace=0.05)
    plt.show()


def detect_row(row):
    volume_after_extra_prv = []
    if abs(row['diff_pvr']) > abs(df.at[idx, '15_E_MA'] * 3.1):
        # T:有能無量， T+1:有量則漲
        if df.at[idx, 'MACD'] > 0:
            if round(df.at[idx, 'MACD']) > 2 and df.at[idx, 'DIF'] < df.at[idx, 'DEA']:
                df.at[idx, 'result'] = "* 0買入時機"
                df.at[idx, 'ind'] = 1
                volume_after_extra_prv = []
                volume_after_extra_prv.append(row['volume'])
            elif round(df.at[idx, 'MACD']) <= 2 and df.at[idx, '5_V_MA'] < (
                    df.at[idx, '10_V_MA'] * 1.5):  # 前期量減(偏差值0.2)
                df.at[idx, 'result'] = "* 1買入時機"
                df.at[idx, 'ind'] = 1
                volume_after_extra_prv = []
                volume_after_extra_prv.append(row['volume'])
            elif df.at[idx, '5_V_MA'] < (df.at[idx, '15_V_MA']):  # 前期量減(偏差值0.2)
                df.at[idx, 'result'] = "* 2買入時機"
                df.at[idx, 'ind'] = 1
                volume_after_extra_prv = []
                volume_after_extra_prv.append(row['volume'])
            elif round(df.at[idx, 'MACD']) > 2 and df.at[idx, 'DIF'] > df.at[idx, 'DEA']:
                df.at[idx, 'result'] = "* 11賣出時機(1.拉高出貨)"
                df.at[idx, 'ind'] = -1
            elif row['volume'] > (df.at[idx, '15_V_MA'] * 3):
                df.at[idx, 'result'] = "* 3賣出時機(1.拉高出貨)"
                df.at[idx, 'ind'] = -1
        else:
            if df.at[idx, '5_V_MA'] < (df.at[idx, '15_V_MA']):  # 前期量減(偏差值0.2)
                df.at[idx, 'result'] = "* 2買入時機"
                df.at[idx, 'ind'] = 1
                volume_after_extra_prv = []
                volume_after_extra_prv.append(row['volume'])
            elif round(df.at[idx, 'MACD']) > 2 and df.at[idx, 'DIF'] > df.at[idx, 'DEA']:
                df.at[idx, 'result'] = "* 3賣出時機(1.拉高出貨)"
                df.at[idx, 'ind'] = -1
            elif row['volume'] > (df.at[idx, '15_V_MA'] * 3):
                df.at[idx, 'result'] = "* 4賣出時機(1.拉高出貨)"
                df.at[idx, 'ind'] = -1
        current_date = datetime.now()
        # 计算日期差
        date_difference = current_date - datetime.strptime(str(row['price_date']), '%Y-%m-%d')
        if date_difference.days <= rec_days:
            if row['avg_volume'] >= rec_volume:
                if df.at[idx, 'ind'] == 1:
                    df.at[idx, 'result'] = f"股票:{stock_code},{rec_days}日内存在買入時機"
                    stock_exists = any(stock['stock_code'] == stock_code for stock in rec_stocks)
                    if not stock_exists:
                        rec_stocks.append(
                            {'stock_code': stock_code, 'stock_name': stock_name, 'volume': row['volume']})
    else:
        # 出現大能後, 記錄量
        volume_after_extra_prv.append(row['volume'])
        if row['volume'] <= (sum(volume_after_extra_prv) / len(volume_after_extra_prv)) * 0.5:
            # 量小於15均量, 則能量耗盡:看空
            df.at[idx, 'result'] = "* 賣出時機(2.拉高出貨)" + df.at[idx, 'result']
            df.at[idx, 'ind'] = -1


# ===================== 主流程 =====================
for master in codes:
    stock_code = master['stock_code']
    stock_name = master['stock_name']
    details = MySQL.get_price(stock_code, analyse_days, 'asc')
    if not details:
        continue

    df = pd.DataFrame(details)
    df['volume'] = df['volume'] / 1000
    df['diff_price'] = df['price'].diff().fillna(0)
    df['diff_volume'] = df['volume'].diff().fillna(0)
    df['diff_pvr'] = np.where(df['diff_volume'] != 0, df['diff_price'] / (df['diff_volume'] / 10000), 0)
    df['avg_pvr'] = df['diff_pvr'].abs().expanding().mean().round(decimal_place)
    df['tgt_price'] = np.where(abs(df['diff_pvr']) > abs(df['avg_pvr']), df['price'].fillna(0), 0)  # 指標價格
    df['est_price'] = df['tgt_price'].where(df['tgt_price'] > 0).expanding().mean().round(decimal_place)  # 平均指標價格
    df['amp_pvr'] = (df['diff_pvr'] / df['avg_pvr']).fillna(0).round(decimal_place)
    df['avg_price'] = df['price'].expanding().mean().round(decimal_place)
    df['avg_volume'] = df['volume'].expanding().mean().round(decimal_place)
    df['RSI'] = ta.rsi(df['price'], length=14)  # 指定window=14
    # df['DIF'], df['DEA'], df['MACD'] = calculate_macd(df['price'])
    macd = ta.macd(df['price'])
    df['DIF'], df['DEA'], df['MACD'] = macd['MACD_12_26_9'].fillna(0), macd['MACDs_12_26_9'].fillna(0), macd[
                                                                                                            'MACDh_12_26_9'].fillna(
        0) * 2
    df = calculate_moving_averages(df)
    df['ind'] = 0
    df['result'] = ""
    # ===================== 判斷買賣時機 =====================
    for idx, row in df.iterrows():
        detect_row(row)
    print(df.to_string())  # 輸出詳細
    print(
        f"股票:{stock_code}, 指標價:{df['est_price'].iloc[-1]}, 均價:{df['avg_price'].iloc[-1]}, 當前股價:{df['price'].iloc[-1]}, MACD:{df['MACD'].iloc[-1]}")

    if display_matplot:
        plot_stock(stock_code, stock_name, df, df['est_price'].iloc[-1], df['avg_price'].iloc[-1])

print("指標股票:", rec_stocks)
