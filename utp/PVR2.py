from datetime import datetime
import pandas_ta as ta
import seaborn as sns
from matplotlib import pyplot as plt

from utp import MySQL
import pandas as pd
import numpy as np

"""
expanding: 行累積合計(階段合計)
"""
decimal_place = 2
analyse_days = 90
codes = MySQL.get_stock(stock_status=None, stock_code='3324')  # 股票列表
sns.set(style="whitegrid")

display_matplot = 1  # 是否顯示圖表
rec_days = 3  # 最近幾日檢查
rec_volume = 1000  # 最小成交量
rec_stocks = []  # 記錄符合條件股票
threshold_macd = 0.05
"""
# 動能（Momentum）類指標
| 檔名          | 全名                          | 用途與解釋                       |
| ----------- | --------------------------- | --------------------------- |
| **ao.py**   | Awesome Oscillator          | 比較短期與長期移動平均的差，用來判斷多空動能轉折。   |
| **apo.py**  | Absolute close Oscillator   | 短期與長期移動平均差（絕對值版 MACD）。      |
| **mom.py**  | Momentum                    | 計算今日收盤價與前幾日收盤價的差值，顯示價格變化速度。 |
| **roc.py**  | Rate of Change              | 價格變動百分比版的 Momentum。         |
| **ppo.py**  | Percentage close Oscillator | 類似 MACD，但輸出百分比變化。           |
| **trix.py** | Triple Exponential Average  | 三重指數平均的變化率，用於濾除短期波動。        |
| **tsi.py**  | True Strength Index         | 根據平滑後的動能變化率判斷趨勢強度。          |
| **uo.py**   | Ultimate Oscillator         | 結合多個時間週期的動能，減少假突破。          |
| **tmo.py**  | True Momentum Oscillator    | 改良版的 Momentum，結合 RSI、平滑化處理。 |


# 震盪指標
| 檔名              | 全名                      | 用途與解釋                       |
| --------------- | ----------------------- | --------------------------- |
| **rsi.py**      | Relative Strength Index | 最常用的超買超賣指標 (>70 超買、<30 超賣)。 |
| **rsx.py**      | Smoothed RSI            | RSI 的平滑化版本，反應更穩定。           |
| **stoch.py**    | Stochastic Oscillator   | 隨機指標，衡量收盤價相對於價格區間的位置。       |
| **stochf.py**   | Fast Stochastic         | 快速版 KD 指標。                  |
| **kdj.py**      | KDJ Indicator           | KD 加上 J 線，放大轉折信號。           |
| **stochrsi.py** | Stochastic RSI          | 把 RSI 套入隨機指標，更敏感的震盪工具。      |
| **willr.py**    | Williams %R             | 判斷價格是否接近高點或低點。              |
| **crsi.py**     | Connors RSI             | RSI 的改良版，結合多個時間週期與變化率。      |


# 趨勢（Trend）類指標
| 檔名             | 全名                                    | 用途與解釋                      |
| -------------- | ------------------------------------- | -------------------------- |
| **macd.py**    | Moving Average Convergence Divergence | 經典趨勢判斷，使用 DIF、DEA、MACD 柱線。 |
| **kst.py**     | Know Sure Thing                       | 結合多重 ROC（動能），強化趨勢確認。       |
| **stc.py**     | Schaff Trend Cycle                    | 結合 MACD 與週期偵測，反應更快的趨勢指標。   |
| **cg.py**      | Center of Gravity                     | 用加權平均計算價格重心，找出轉折。          |
| **slope.py**   | Linear Regression Slope               | 回歸線斜率，用於判斷趨勢方向。            |
| **smc.py**     | Smart Money Concepts                  | 將市場結構與高低點結合的趨勢工具。          |
| **inertia.py** | Inertia Indicator                     | 根據動能與方向性強弱判斷持續性。           |


# 波動性（Volatility）與壓縮（Squeeze）類指標
| 檔名                 | 全名                     | 用途與解釋                          |
| ------------------ | ---------------------- | ------------------------------ |
| **squeeze.py**     | Bollinger Band Squeeze | 結合布林帶與 Keltner Channel，偵測波動壓縮。 |
| **squeeze_pro.py** | Enhanced Squeeze       | 改良版壓縮訊號，用於突破預測。                |
| **pgo.py**         | Pretty Good Oscillator | 衡量價格偏離移動平均的程度。                 |
| **eri.py**         | Elder Ray Index        | 根據牛熊力量計算趨勢與波動。                 |


# 量價與市場心理（Volume / Sentiment）類
| 檔名             | 全名                              | 用途與解釋                |
| -------------- | ------------------------------- | -------------------- |
| **bop.py**     | Balance of Power                | 測量買方與賣方力量平衡。         |
| **brar.py**    | BRAR Indicator                  | 代表市場多空力量對比（常用於 A 股）。 |
| **cfo.py**     | Chande Forecast Oscillator      | 預測未來趨勢的動能指標。         |
| **dm.py**      | Directional Movement            | DMI 系統的一部分，用來判斷趨勢強度。 |
| **er.py**      | Efficiency Ratio                | 衡量價格變化效率（越高代表趨勢明確）。  |
| **exhc.py**    | Exponential Hull Moving Average | 改良版移動平均，結合平滑與反應速度。   |
| **bais.py**    | Bias Ratio                      | 判斷價格偏離均線的程度。         |
| **coppock.py** | Coppock Curve                   | 長期動能指標，用於判斷牛市開始。     |
| **cmo.py**     | Chande Momentum Oscillator      | RSI 類似，但上下對稱的動能震盪器。  |
| **psl.py**     | Psychological Line              | 根據上漲天數比例反映投資者心理。     |
| **rvgi.py**    | Relative Vigor Index            | 比較收盤價與開盤價的強弱。        |
| **cci.py**     | Commodity Channel Index         | 判斷價格偏離平均的程度。         |

# 複合或進階分析類
| 檔名             | 全名                          | 用途與解釋              |
| -------------- | --------------------------- | ------------------ |
| **fisher.py**  | Fisher Transform            | 將價格標準化為常態分布，用於抓轉折。 |
| **cti.py**     | Correlation Trend Indicator | 測量趨勢持續性與價格相關性。     |
| **coppock.py** | Coppock Curve               | 用於判斷長期趨勢（多用於週線）。   |
| **inertia.py** | Inertia Indicator           | 量化價格動能的「慣性」。       |


# 建議分類應用
| 類別         | 代表指標                     | 用途           |
| ---------- | ------------------------ | ------------ |
| **趨勢追蹤**   | MACD、KST、STC、CG、SLOPE    | 用於長線進出判斷     |
| **短線震盪**   | RSI、KDJ、STOCH、WILLR、CRSI | 抓超買超賣、反轉     |
| **波動壓縮**   | SQUEEZE、SQUEEZE_PRO      | 偵測盤整即將突破     |
| **量價關係**   | BOP、BRAR、PSL、ER          | 用於判斷市場情緒與力量  |
| **動能強弱**   | MOM、ROC、TSI、UO、AO        | 評估當前價格動能     |
| **心理面與預測** | FISHER、CTI、COPPOCK       | 捕捉趨勢轉折、周期啟動點 |
"""


# ===================== 向量化計算 RSI [简单移动平均（SMA）]=====================
def calculate_rsi_sma(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


# ===================== 向量化計算 RSI [ Wilder's smoothing（平滑移动平均）]=====================
def calculate_rsi_wilder(df, column='close', period=14):
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
def calc_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    DIF = ema12 - ema26
    DEA = DIF.ewm(span=9, adjust=False).mean()
    MACD = 2 * (DIF - DEA)
    return DIF, DEA, MACD


# ===================== 計算各種均線 =====================
def calc_ma(df):
    for p in [5, 10, 15]:
        df[f'{p}_MA'] = df['close'].rolling(p).mean()
        df[f'{p}_V_MA'] = df['volume'].rolling(p).mean()
        df[f'{p}_E_MA'] = df['diff_pvr'].abs().rolling(p).mean()
    return df


# 定义鼠标移动事件处理程序
def on_mouse_move(event, df, ax1, ax2, ax3, ax4, stock_code, stock_name, est_price, avg_price):
    """滑鼠移動事件：在所有子圖同步顯示指示線，橫線對應各軸資料"""
    if event.inaxes is None or event.xdata is None:
        return

    axes = [ax1, ax2, ax3]  # 若包含正負資料，可自行擴充

    # 計算索引
    idx = int(round(event.xdata))
    if idx < 0 or idx >= len(df):
        return

    cur = df.iloc[idx]

    # 各軸對應資料
    y_values = {
        ax1: cur.get('close', 0),
        ax2: cur.get('volume', 0),
        ax3: cur.get('RSI', 0),
    }

    # 壓力/支撐
    press_top = df[(df['ind'] != 0) & (df['close'] > cur['close']) & (df['price_date'] < cur['price_date'])]
    press_low = df[(df['ind'] != 0) & (df['close'] < cur['close']) & (df['price_date'] < cur['price_date'])]
    press_top_price = press_top.iloc[-1]['close'] if not press_top.empty else 0
    press_low_price = press_low.iloc[-1]['close'] if not press_low.empty else 0

    for ax in axes:
        # 初始化屬性
        ax._indicator_lines = getattr(ax, '_indicator_lines', [])
        ax._indicator_texts = getattr(ax, '_indicator_texts', [])

        # 清除舊線
        for l in ax._indicator_lines:
            try:
                l.remove()
            except Exception:
                pass
        ax._indicator_lines.clear()

        # 清除文字只在 ax1
        if ax is ax1:
            for t in ax._indicator_texts:
                try:
                    t.remove()
                except Exception:
                    pass
            ax._indicator_texts.clear()

        # 取 y 範圍 (避免 None)
        ymin, ymax = ax.get_ylim()
        if ymin is None or ymax is None:
            continue

        # 畫線（確保 y 值合法）
        yv = y_values.get(ax, 0)
        if not isinstance(yv, (int, float)):
            continue

        hline = ax.axhline(y=yv, color='gray', linestyle='--', alpha=0.6)
        vline = ax.axvline(x=idx, color='gray', linestyle='--', alpha=0.6)
        ax._indicator_lines = [hline, vline]

        # 文字只在 ax1 顯示
        if ax is ax1:
            msg = (f"日期: {cur['price_date']}\n"
                   f"PVR: {cur['diff_pvr']:.2f}\n"
                   f"價格: {cur['close']:.2f}, 量: {cur['volume']}\n"
                   f"(壓力: {press_top_price:.2f} 支撐: {press_low_price:.2f})")
            text = ax.text(0.98, 0.98, msg, ha='right', va='top', transform=ax.transAxes,
                           color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            ax._indicator_texts = [text]

    # 更新標題
    ax1.set_title(
        f"{stock_name}({stock_code}) | 指標價:{est_price} 均價:{avg_price} "
        f"價:{cur['close']} 量:{cur['volume']} (壓:{press_top_price} 支:{press_low_price})"
    )

    event.canvas.draw_idle()

# ===================== 畫圖 =====================
def plot_stock(stock_code, stock_name, df, est_price, avg_price):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 12))
    plt.get_current_fig_manager().set_window_title('買賣建議')
    plt.rcParams['font.sans-serif'] = ['Heiti TC']

    # GridSpec: 五層圖
    gs = GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.05)
    ax_price = fig.add_subplot(gs[0])  # 價格圖
    ax_amp = fig.add_subplot(gs[1], sharex=ax_price)  # 波動PVR
    ax_vol = fig.add_subplot(gs[2], sharex=ax_price)  # 成交量
    ax_rsi = fig.add_subplot(gs[3], sharex=ax_price)  # RSI
    ax_macd = fig.add_subplot(gs[4], sharex=ax_price)  # MACD

    # 上方價格圖
    ax_price.plot(df.index, df['close'], color='red', label='價格', linewidth=2)
    ax_price.set_ylabel('價格')
    for p in [5, 10, 15]:
        ax_price.plot(df.index, df[f'{p}_MA'], label=f'{p}日均價', linestyle='dashed')

    # 標記買賣訊號
    for idx, detail in df.iterrows():
        if df.at[idx, 'ind'] > 0:
            ax_price.scatter(idx, df['close'].min() * 0.99, marker='^', color='red', s=80)
        elif df.at[idx, 'ind'] < 0:
            ax_price.scatter(idx, df['close'].min() * 0.99, marker='v', color='green', s=80)

    # 波動PVR圖
    ax_amp.plot(df.index, df['amp_pvr'], color='blue', label='波動PVR(異常主力)', linewidth=1)
    ax_amp.set_ylabel('波動PVR')
    ax_amp.legend(fontsize=8, loc='upper left')

    # 成交量圖
    ax_vol.bar(df.index, df['volume'], color='#ff00ff', alpha=0.6, width=0.8, label='成交量')
    ax_vol.set_ylabel('成交量')
    ax_vol.legend(fontsize=8, loc='upper left')

    # RSI圖
    ax_rsi.plot(df.index, df['RSI'], color='purple', label='RSI')
    ax_rsi.axhline(70, color='red', linestyle='dashed', linewidth=0.7)
    ax_rsi.axhline(30, color='green', linestyle='dashed', linewidth=0.7)
    ax_rsi.set_ylabel('RSI')
    ax_rsi.legend(fontsize=8, loc='upper left')

    # MACD圖
    ax_macd.bar(df.index, df['MACD'].where(df['MACD'] > 0, 0), color='red', alpha=0.6, width=0.8)
    ax_macd.bar(df.index, df['MACD'].where(df['MACD'] < 0, 0), color='blue', alpha=0.6, width=0.8)
    ax_macd.set_ylabel('MACD')
    ax_macd.set_xlabel('日期')

    # 隱藏上方子圖 x 標籤
    ax_price.xaxis.set_tick_params(labelbottom=False)
    ax_amp.xaxis.set_tick_params(labelbottom=False)
    ax_vol.xaxis.set_tick_params(labelbottom=False)
    ax_rsi.xaxis.set_tick_params(labelbottom=False)
    ax_macd.set_xticks(df.index)
    ax_macd.set_xticklabels(df['price_date'].astype(str), rotation=90, fontsize=8)

    # 合併圖例到價格圖
    lines = []
    labels = []
    for ax in [ax_price, ax_amp, ax_vol, ax_rsi, ax_macd]:
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    ax_price.legend(lines, labels, fontsize=8, loc='upper left')

    # 滑鼠互動事件
    fig.canvas.mpl_connect(
        'motion_notify_event',
        lambda event: on_mouse_move(event, df, ax_price, ax_vol, ax_rsi, ax_macd,
                                    stock_code, stock_name,
                                    df['est_price'].iloc[-1], df['avg_price'].iloc[-1])
    )

    # 手動調整間距
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.07, right=0.95, hspace=0.05)
    plt.show()

def detect_rule2(idx, row):
    volume_after_extra_prv = []
    if abs(row['diff_pvr']) > abs(row['avg_pvr'] * 2):
        if idx <= 30: # DEA需要26日計算, 如果數據量過少不做訊號判斷
            return
        # T:有能無量， T+1:有量則漲
        if row['RSI'] > 30:
            print(row['price_date'], row['5_V_MA'], row['15_V_MA'], row['close'], row['10_MA'])
            if (row['5_V_MA'] < (row['15_V_MA'])) and (row['close'] < row['10_MA']):  # 量縮且價格單價10日均下
                df.at[idx, 'result'] = "* 進貨訊號(量縮且價格單價10日均下))"
                df.at[idx, 'ind'] = 1
            else:
                df.at[idx, 'result'] = "* 出貨訊號2"
                df.at[idx, 'ind'] = -1
        elif row['RSI'] < 30:
            df.at[idx, 'result'] = "* 進貨訊號3"
            df.at[idx, 'ind'] = 1
        else:
            if row['MACD_TREAD'] < 0:
                df.at[idx, 'result'] = "* 出貨訊號4"
                df.at[idx, 'ind'] = -1
            else:
                df.at[idx, 'result'] = "* 進貨訊號5"
                df.at[idx, 'ind'] = 1
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
    df['diff_price'] = df['close'].diff().fillna(0)
    df['diff_volume'] = df['volume'].diff().fillna(0)
    # 異常主力
    df['diff_pvr'] = np.where(df['diff_volume'] != 0, df['diff_price'] / (df['diff_volume'] / 10000), 0)  # 單位量差
    df['avg_pvr'] = df['diff_pvr'].abs().rolling(window=10, min_periods=1).mean()
    # df['amp_pvr'] = (df['diff_pvr'] / df['avg_pvr']).fillna(0).round(decimal_place)
    df['amp_pvr'] = (
        (df['diff_pvr'] / df['avg_pvr']).replace([np.inf, -np.inf], 0).fillna(0)
    ).clip(-5, 5).round(decimal_place)
    df['tgt_price'] = np.where(abs(df['diff_pvr']) > abs(df['avg_pvr']), df['close'].fillna(0), 0)  # 指標價格
    df['est_price'] = df['tgt_price'].where(df['tgt_price'] > 0).expanding().mean().round(decimal_place)  # 平均指標價格
    df['avg_price'] = df['close'].expanding().mean().round(decimal_place)
    df['avg_volume'] = df['volume'].expanding().mean().round(decimal_place)
    df['RSI'] = ta.rsx(df['close'], length=14)  # 指定window=14
    # 計算MACD (含DIF/DEA)
    if len(df['close']) < 30:  # MACD慢線需要至少26個數值
        print(f"股票:{stock_code} 價格資料太少，無法計算 MACD")
        continue
    macd = ta.macd(df['close'])
    df['DIF'] = macd['MACD_12_26_9'].fillna(0)
    df['DEA'] = macd['MACDs_12_26_9'].fillna(0)
    df['MACD'] = macd['MACDh_12_26_9'].fillna(0)
    #df['DIF'], df['DEA'], df['MACD'] = calc_macd(df['close'])

    # 加權平均求MACD趨勢(3日趨勢)
    weights = np.arange(1, 4)  # [1,2,3,4,5]，越近越重
    df['MACD_5wma'] = df['MACD'].rolling(3).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    df['MACD_TREAD'] = np.where(df['MACD'] > df['MACD_5wma'], 1, np.where(df['MACD'] < df['MACD_5wma'], -1, 0)) # 判斷趨勢
    #df.drop(columns=['MACD_5wma'], inplace=True) # 刪掉中間欄位（可選）

    # 偵測金叉、死叉
    df['MACD_SIG'] = 0.0
    df.loc[(df['DIF'].shift(1) < df['DEA'].shift(1)) & (df['DIF'] > df['DEA']), 'MACD_SIG'] = 1  # 金叉
    df.loc[(df['DIF'].shift(1) > df['DEA'].shift(1)) & (df['DIF'] < df['DEA']), 'MACD_SIG'] = -1  # 死叉
    # 接近交叉（預警）
    diff = df['DIF'] - df['DEA']
    df.loc[(diff.between(-threshold_macd, 0)) & (df['DIF'] < df['DEA']), 'MACD_SIG'] = 0.5  # 接近金叉
    df.loc[(diff.between(0, threshold_macd)) & (df['DIF'] > df['DEA']), 'MACD_SIG'] = -0.5  # 接近死叉

    # TSI > 0 → 多方強勢，可考慮買入
    tsi_df = ta.tsi(df['close'], r=2, s=2)
    df['TSI'] = tsi_df.iloc[:, 0]  # 取第一欄

    # 計算均線
    df = calc_ma(df)
    df['ind'] = 0
    df['result'] = ""
    # ===================== 判斷買賣時機 =====================
    for idx, row in df.iterrows():
        detect_rule2(idx, row)

    print(df.to_string())  # 輸出詳細
    # print( f"股票:{stock_code}, 指標價:{df['est_price'].iloc[-1]}, 均價:{df['avg_price'].iloc[-1]}, 當前股價:{df['close'].iloc[-1]}, MACD:{df['MACD'].iloc[-1]}")

    if display_matplot:
        plot_stock(stock_code, stock_name, df, df['est_price'].iloc[-1], df['avg_price'].iloc[-1])

print("指標股票")
for stock_code in rec_stocks:
    print(f"{stock_code}")
