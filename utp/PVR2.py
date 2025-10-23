from utp import MySQL
from datetime import datetime
import pandas_ta as ta
import seaborn as sns
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import humps
from scipy.optimize import curve_fit

"""
OBV(On Balance Volume)(能量潮指標)(與價同上則看漲, 與價格同下則看跌, 如果與價背離則反轉)
公式：OBV =  OBV(T-1) + Volume X (math.copysign(1, diff_volume))
"""
"""
expanding: 行累積合計(階段合計)
"""
decimal_place = 2
analyse_days = 90
stock_code = ['3324']
codes = MySQL.get_stock(stock_status=90, stock_code=[])  # 股票列表
sns.set_theme(style="whitegrid")
display_matplot = 1  # 是否顯示圖表
display_df = 1  # 是否顯示詳細數據 (0.不顯示 1.全部顯示 2.只顯示趨勢)
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
        df[f'{p}_E_MA'] = df['diffPvr'].abs().rolling(p).mean()
    return df


# ===================== 拉高出貨 + 低位承接檢測 =====================
import numpy as np
import pandas as pd


def detect_trade_signals(df, pct_thresh_up=2.2, pct_thresh_acc=2.0, vol_window=5,
                         upper_shadow_thresh=0.65, lower_shadow_thresh=0.30,
                         rsi=None, macd=None, trend_window=20, cum_window=5):
    """
    detect_trade_signals_v3_6
    智能平衡 + 信號強度版：
    - 維持 v3.5 準確邏輯
    - 新增 SignalStrength 欄位 (0~10)
    """
    df = df.copy()
    df['prevClose'] = df['close'].shift(1)
    df['diffClose'] = (df['close'] - df['prevClose']) / df['prevClose'] * 100

    # 成交量
    df['avgVolume'] = df['volume'].rolling(vol_window).mean()
    df['stdVolume'] = df['volume'].rolling(vol_window).std()
    df['zScoreVolume'] = (df['volume'] - df['avgVolume']) / df['stdVolume']

    # 影線比例
    df['實體長'] = abs(df['close'] - df['prevClose'])
    df['上影線比'] = np.where(df['實體長'] > 0,
                              (df['high'] - df[['close', 'prevClose']].max(axis=1)) / df['實體長'], 0)
    df['下影線比'] = np.where(df['實體長'] > 0,
                              (df[['close', 'prevClose']].min(axis=1) - df['low']) / df['實體長'], 0)

    # 均線與趨勢
    df['MATrend'] = df['close'].rolling(trend_window).mean()
    df['priceVsTrend'] = (df['close'] - df['MATrend']) / df['MATrend'] * 100
    df['cumPct'] = df['diffClose'].rolling(cum_window).sum()

    # -------- 拉高出貨 --------
    df['scoreUp'] = 0
    df['scoreUp'] += (df['diffClose'] > pct_thresh_up) * 3
    df['scoreUp'] += (df['zScoreVolume'] > 1.3) * 2
    df['scoreUp'] += (df['上影線比'] > upper_shadow_thresh) * 1.5
    df['scoreUp'] += (df['priceVsTrend'] > 1.5) * 1

    if rsi is not None:
        df['scoreUp'] += (df[rsi] > 75) * 1
    if macd is not None:
        df['scoreUp'] += (df[macd] < -0.3) * 0.5

    df['拉高出貨'] = (df['scoreUp'] >= 5.5) & \
                     (df['close'] > df['MATrend'] * 1.015) & \
                     (df['cumPct'] > 3.5)

    # -------- 低位承接 --------
    df['scoreAcc'] = 0
    df['scoreAcc'] += ((-1 <= df['diffClose']) & (df['diffClose'] <= pct_thresh_acc)) * 2.5
    df['scoreAcc'] += (df['zScoreVolume'] > -1.0) * 1.5
    df['scoreAcc'] += (df['下影線比'] > lower_shadow_thresh) * 2
    df['scoreAcc'] += (df['priceVsTrend'] < -1.5) * 1

    if rsi is not None:
        df['scoreAcc'] += (df[rsi] < 25) * 1
    if macd is not None:
        df['scoreAcc'] += (df[macd] > 0.3) * 0.5

    # 無量反轉補強
    df['scoreAcc'] += ((df['zScoreVolume'] < -1.0) & (df['diffClose'] > 0)) * 1.5

    df['低位承接'] = (df['scoreAcc'] >= 5.5) & \
                     (df['close'] < df['MATrend'] * 0.985) & \
                     (df['cumPct'] >= -3.5)

    # -------- 信號強度 (0~10) --------
    df['SignalStrength'] = np.select(
        [
            df['拉高出貨'],
            df['低位承接']
        ],
        [
            np.clip(df['scoreUp'] * 1.5, 0, 10),  # 拉高出貨強度
            np.clip(df['scoreAcc'] * 1.5, 0, 10)  # 低位承接強度
        ],
        default=0
    )

    # -------- 清理 --------
    df.drop(columns=['scoreUp', 'scoreAcc', 'prevClose', 'stdVolume', '實體長',
                     'MATrend', 'cumPct', 'priceVsTrend', 'avgVolume'], inplace=True)

    return df


# ==================== 預估函式區 ====================
def exp_func(x, a, b):
    """指數擬合公式 y = a * b^x"""
    return a * (b ** x)


def predict_next(df, days=7):
    """根據歷史 close，自動預測未來 N 天"""
    x_data = np.arange(len(df))
    y_data = df['close'].values

    # 指數擬合
    popt, _ = curve_fit(exp_func, x_data, y_data, p0=(1000, 1.02))
    a, b = popt

    # 預測未來 days 天
    future_x = np.arange(len(df), len(df) + days)
    future_y = exp_func(future_x, a, b)
    future_dates = pd.date_range(df['priceDate'].iloc[-1] + pd.Timedelta(days=1), periods=days, freq='D').date

    result = pd.DataFrame({
        '日期': future_dates,
        'estClose': future_y.round(0).astype(int)
    })
    """
    print(f"=== 指數趨勢線方程式 ===")
    print(f"y = {a:.0f} × {b:.4f}^x")
    print("\n=== 未來預測 ===")
    print(result, "\n")
    """
    return result


# 定义鼠标移动事件处理程序
def on_mouse_move_auto(event, df, axes, stock_code, stock_name):
    """滑鼠移動事件：在所有子圖同步顯示指示線，橫線對應各軸資料"""
    if event.inaxes is None or event.xdata is None:
        return

    # 計算索引
    idx = int(round(event.xdata))
    if idx < 0 or idx >= len(df):
        return

    cur = df.iloc[idx]

    # 各軸對應資料
    y_values = {
        axes.get('close'): cur.get('close', 0),
        axes.get('volume'): cur.get('volume', 0),
        axes.get('RSI'): cur.get('RSI', 0),
        axes.get('OBV'): cur.get('OBV', 0),
        axes.get('ampPvr'): cur.get('ampPvr', 0),
        axes.get('MACD'): cur.get('MACD', 0),
        axes.get('KDJ'): cur.get('KDJ', 0),
    }

    for ax, yv in y_values.items():
        if ax is None or not isinstance(yv, (int, float)):
            continue

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

        # 清除文字只在價格圖
        if ax == axes.get('close'):
            for t in ax._indicator_texts:
                try:
                    t.remove()
                except Exception:
                    pass
            ax._indicator_texts.clear()

        # 畫水平線與垂直線
        hline = ax.axhline(y=yv, color='gray', linestyle='--', alpha=0.6)
        vline = ax.axvline(x=idx, color='gray', linestyle='--', alpha=0.6)
        ax._indicator_lines = [hline, vline]

        # 價格圖上顯示文字
        if ax == axes.get('close'):
            press_top = df[(df['trand'] != 0) & (df['close'] > cur['close']) & (df['priceDate'] < cur['priceDate'])]
            press_low = df[(df['trand'] != 0) & (df['close'] < cur['close']) & (df['priceDate'] < cur['priceDate'])]
            press_top_price = press_top.iloc[-1]['close'] if not press_top.empty else 0
            press_low_price = press_low.iloc[-1]['close'] if not press_low.empty else 0

            msg = f"日期: {cur['priceDate']}\n"
            msg += f"PVR: {cur.get('diff_pvr', 0):.2f}\n"
            if not pd.isna(cur.get('close')):
                msg += f"價格: {cur.get('close', 0):.2f}, 量: {cur.get('volume', 0)}\n"
            else:
                msg += f"估價: {cur.get('estClose')}, 量: {cur.get('volume', 0)}\n"
            msg += f"(壓力: {press_top_price:.2f} 支撐: {press_low_price:.2f})"

            text = ax.text(0.98, 0.98, msg, ha='right', va='top', transform=ax.transAxes,
                           color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            ax._indicator_texts = [text]

    # 更新標題
    if axes.get('close'):
        ax = axes['close']
        msg = f"{stock_name}({stock_code})"
        msg += f"\n指標價:{cur.get('est_price')} 均價:{cur.get('avg_price')}"
        if not pd.isna(cur.get('close')):
            msg += f"\n價:{cur.get('close')} 量:{cur.get('volume')}"
        else:
            msg += f"\n估價:{cur.get('estClose')} 量:{cur.get('volume')}"
        msg += f"\n(壓:{press_top_price} 支:{press_low_price})"
        if not pd.isna(cur.get('reason')):
            msg += f"\n{cur.get('reason')}"

        ax.set_title(msg)

    event.canvas.draw_idle()


# ===================== 畫圖 =====================
PANEL_CONFIG = {
    'close': {'ylabel': '價格', 'type': 'line', 'color': 'red', 'height': 2},
    'volume': {'ylabel': '成交量', 'type': 'bar', 'color': '#ff00ff', 'height': 1},
    'ampPvr': {'ylabel': '波動PVR', 'type': 'line', 'color': 'blue', 'height': 1},
    'RSI': {'ylabel': 'RSI', 'type': 'line', 'color': 'purple', 'height': 1},
    'OBV': {'ylabel': 'OBV', 'type': 'line', 'color': 'purple', 'height': 1},
    'KDJ': {'ylabel': 'KDJ', 'type': 'line', 'color': 'blue', 'height': 1},
    'MACD': {'ylabel': 'MACD', 'type': 'bar', 'color': 'red', 'height': 0.8},
}


def plot_stock(stock_code, stock_name, df):
    plt.rcParams['font.sans-serif'] = ['Heiti TC']
    fig = plt.figure(figsize=(12, 10))
    plt.get_current_fig_manager().set_window_title(f"{stock_code} - {stock_name}")

    # 只保留 df 裡有的 panels
    panels = [p for p in PANEL_CONFIG]

    # 動態 height_ratios
    ratios = [PANEL_CONFIG[p]['height'] for p in panels]
    gs = GridSpec(len(panels), 1, height_ratios=ratios, hspace=0.05)

    axes = {}
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[i], sharex=axes[panels[0]] if i > 0 else None)
        axes[p] = ax
        cfg = PANEL_CONFIG[p]

        # 繪圖
        if cfg['type'] == 'line' and p in df:
            if p == 'close':
                ax.plot(df.index, df[p], color=cfg['color'], label='價格', linewidth=2)
                ax.plot(df.index, df['estClose'], color=cfg['color'], label='估價', linewidth=1,
                        linestyle='dashed')
                # 繪製均線
                for ma in [5, 10, 15]:
                    ma_col = f'{ma}_MA'
                    if ma_col in df:
                        ax.plot(df.index, df[ma_col], label=f'{ma}日均線', linestyle='dashed')
                # 繪製買賣訊號
                for idx, row in df.iterrows():
                    if 'score' in df.columns:
                        min_close = df['close'].min() * 0.9  # 在最低收盤價下方0.99倍的位置畫紅色三角形標記
                        if row['trand'] == 1:
                            ax.scatter(idx, min_close, marker='^', color='red', s=80)
                        elif row['trand'] == 0.5:
                            ax.scatter(idx, min_close, marker='^', color='pink', s=80)
                        elif row['trand'] == -0.5:
                            ax.scatter(idx, min_close, marker='v', color='lightgreen', s=80)
                        elif row['trand'] == -1:
                            ax.scatter(idx, min_close, marker='v', color='green', s=80)
            elif p == 'RSI':
                ax.axhline(70, color='red', linestyle='dashed', linewidth=0.7)
                ax.axhline(30, color='green', linestyle='dashed', linewidth=0.7)
                ax.plot(df.index, df[p], color=cfg['color'], label=p)
            elif p == 'KDJ':
                # 畫三條線
                K = [x[0] if isinstance(x, (list, tuple)) else np.nan for x in df['KDJ']]
                D = [x[1] if isinstance(x, (list, tuple)) else np.nan for x in df['KDJ']]
                J = [x[2] if isinstance(x, (list, tuple)) else np.nan for x in df['KDJ']]

                ax.plot(df.index, K, label='K', color='blue', linewidth=1)
                ax.plot(df.index, D, label='D', color='orange', linewidth=1)
                ax.plot(df.index, J, label='J', color='purple', linewidth=1)

                # 超買/超賣區間
                ax.axhline(80, color='red', linestyle='--', alpha=0.5)
                ax.axhline(20, color='green', linestyle='--', alpha=0.5)

                gold_cross_idx = [i for i in range(1, len(K)) if K[i] > D[i] and K[i - 1] <= D[i - 1]]
                death_cross_idx = [i for i in range(1, len(K)) if K[i] < D[i] and K[i - 1] >= D[i - 1]]

                ax.scatter(gold_cross_idx, [K[i] for i in gold_cross_idx], marker='^', color='red', s=50, label='金叉')
                ax.scatter(death_cross_idx, [K[i] for i in death_cross_idx], marker='v', color='green', s=50,
                           label='死叉')
            else:
                ax.plot(df.index, df[p], color=cfg['color'], label=p)
        elif cfg['type'] == 'bar' and p in df:
            if p == 'macd':
                ax.bar(df.index, df['MACD'].where(df['MACD'] > 0, 0), color='red', alpha=0.6)
                ax.bar(df.index, df['MACD'].where(df['MACD'] < 0, 0), color='blue', alpha=0.6)
            elif p == 'volume':
                ax.bar(df.index, df[p], color=cfg['color'], alpha=0.6)

                y_marker = -df['volume'].min() * 0.8  # 標記位置

                # 拉高出貨 scatter（只加一次 label）
                lh_indices = df.index[df['拉高出貨'].notna() & df['拉高出貨']]
                ax.scatter(lh_indices, [y_marker] * len(lh_indices), marker='v', color='green', s=80, label='拉高出貨')

                # 低位承接 scatter（只加一次 label）
                dw_indices = df.index[df['低位承接'].notna() & df['低位承接']]
                ax.scatter(dw_indices, [y_marker] * len(dw_indices), marker='^', color='red', s=80, label='低位承接')
                # ax.set_ylim(y_marker - 50, df[p].max() * 1.1)
            else:
                ax.bar(df.index, df[p], color=cfg['color'], alpha=0.6)
        ax.set_ylabel(cfg['ylabel'])
        lines, labels = ax.get_legend_handles_labels()
        if lines:  # 有 label 才畫
            ax.legend(lines, labels, fontsize=8, loc='upper left')

    # X 軸顯示
    for name, ax in axes.items():
        if name != panels[-1]:
            ax.tick_params(labelbottom=False)
    # 1️⃣ 畫完所有面板
    axes[panels[-1]].set_xticks(df.index)
    axes[panels[-1]].set_xticklabels(df['priceDate'].astype(str), rotation=90, fontsize=8)
    # 2️⃣ 調整子圖間距
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.07, right=0.95)
    # 只顯示有效分析的日期 (分析天數 - MACD的30天)
    axes[panels[-1]].set_xlim(df.index[-(analyse_days - 30)], df.index[-1])
    # 3️⃣ 綁定滑鼠事件（這裡必須用 axes 字典）
    fig.canvas.mpl_connect(
        'motion_notify_event',
        lambda event: on_mouse_move_auto(
            event, df, axes,
            stock_code, stock_name
        )
    )
    plt.show()


def detect_rule3(idx, row, df):
    """
    detect_rule3_v15 - 支援金叉/死叉型態 + 築底修正
    - 高位放量/超買 → 偏空
    - 低位縮量/超賣 → 偏多
    - 死叉分為高檔死叉（偏空）與低檔死叉（築底偏多）
    - 無量下跌 → 底部偏多加分
    - RSI / KDJ / MACD 使用趨勢化比例分數
    """

    score = 0.0
    reasons = []

    # 權重設定
    weights = {
        'RSI': 1.4,
        'KDJ': 1.0,
        'MA': 1.0,
        'VOL': 0.6,
        'PVR': 0.8,
        'MACD': 1.0,
    }

    # 底部 / 高位加權
    bottom_boost = 5
    top_penalty = 5

    # === 基本資料 ===
    rsi = row['RSI']
    kdj = row['KDJ']
    if isinstance(kdj, (list, tuple)):
        K, D, J = kdj
    else:
        K, D, J = np.nan, np.nan, np.nan

    pvr = row['ampPvr']
    macd_strength = row['DIF'] - row['DEA']
    diff_price = row['diffPrice']

    prev_rsi = row.get('prev_RSI', rsi)
    prev_J = row.get('prev_J', J)

    # 前一日KDJ（供金叉死叉判斷）
    prev_K, prev_D, _ = (np.nan, np.nan, np.nan)
    if 'prev_KDJ' in row and isinstance(row['prev_KDJ'], (list, tuple)):
        prev_K, prev_D, _ = row['prev_KDJ']

    # === 底部 / 高位 ===
    is_bottom = (rsi < 40 and J < 40 and pvr < 0)
    is_top = (rsi > 60 and J > 70 and pvr > 2)

    # === 量比 ===
    vol_ratio = 1
    if '15_V_MA' in row and row['15_V_MA'] != 0:
        vol_ratio = row['5_V_MA'] / row['15_V_MA']

    # === RSI 趨勢化分數 ===
    rsi_trend = prev_rsi - rsi
    rsi_score = np.clip(rsi_trend / 50, -1, 1)
    if is_bottom:
        rsi_score += bottom_boost
    elif is_top:
        rsi_score -= top_penalty
    rsi_score = np.clip(rsi_score, -1, 1)
    score += rsi_score * weights['RSI']
    reasons.append(f"RSI={rsi:.1f} (前{prev_rsi:.1f}) → {rsi_score:+.2f}")

    # === 🔥 KDJ 金叉／死叉 + 趨勢分數 ===
    kdj_diff = J - prev_J
    kdj_score = np.clip(kdj_diff / 50, -1, 1)

    # --- 新增：金叉 / 死叉 判斷 ---
    is_gold_cross = (K > D) and (prev_K <= prev_D)
    is_dead_cross = (K < D) and (prev_K >= prev_D)

    # 高檔死叉 / 低檔死叉分流
    is_top_dead = is_dead_cross and (K > 70 or D > 70)
    is_bottom_dead = is_dead_cross and (K < 40 and D < 40 and J < 30)

    if is_gold_cross:
        kdj_score += 0.6
        reasons.append("KDJ金叉 → 多方啟動")
    elif is_top_dead:
        kdj_score -= 0.6
        reasons.append("KDJ高檔死叉 → 出貨警訊")
    elif is_bottom_dead:
        kdj_score += 0.3
        reasons.append("KDJ低檔死叉 → 築底吸籌")
    elif is_dead_cross:
        kdj_score -= 0.3
        reasons.append("KDJ死叉 → 趨勢轉弱")

    # 原底部 / 高位 加權
    if is_bottom:
        kdj_score += bottom_boost
    elif is_top:
        kdj_score -= top_penalty

    kdj_score = np.clip(kdj_score, -1, 1)
    score += kdj_score * weights['KDJ']
    reasons.append(f"KDJ趨勢 J={J:.1f} (前{prev_J:.1f}) → {kdj_score:+.2f}")

    # === 均線乖離 ===
    if '10_MA' in row and row['10_MA'] != 0:
        ma_diff = (row['close'] - row['10_MA']) / row['10_MA']
        ma_score = np.clip(ma_diff * 4, -1, 1)
        if is_top and ma_diff > 0:
            ma_score *= -0.5
        if is_bottom and ma_diff < 0:
            ma_score *= 0.5
        score += ma_score * weights['MA']
        reasons.append(f"均線乖離={ma_diff * 100:.1f}% → {ma_score:+.2f}")

    # === 成交量 ===
    vol_score = (vol_ratio - 1) * 0.4
    if is_bottom and vol_ratio < 0.8:
        vol_score += bottom_boost
    elif is_top and vol_ratio > 1.2:
        vol_score -= top_penalty
    vol_score = np.clip(vol_score, -1, 1)
    score += vol_score * weights['VOL']
    reasons.append(f"量比={vol_ratio:.2f} → {vol_score:+.2f}")

    # === PVR ===
    pvr_score = np.clip(pvr / 5, -1, 1)
    if is_bottom and pvr < -2:
        pvr_score += bottom_boost
    elif is_top and pvr > 2:
        pvr_score -= top_penalty
    pvr_score = np.clip(pvr_score, -1, 1)
    score += pvr_score * weights['PVR']
    reasons.append(f"PVR振幅={pvr:.2f} → {pvr_score:+.2f}")

    # === MACD ===
    macd_trend = macd_strength / (abs(row['DIF']) + 1e-6)
    if is_bottom and macd_strength < 0 and abs(macd_strength) < 0.5:
        macd_trend += bottom_boost
    elif is_top and macd_strength > 0 and macd_strength < 0.3:
        macd_trend -= top_penalty
    macd_trend = np.clip(macd_trend, -1, 1)
    score += macd_trend * weights['MACD']
    reasons.append(f"MACD差={macd_strength:.4f} → {macd_trend:+.2f}")

    # === 無量下跌判斷 ===
    low_volume_down = (diff_price < 0) and (vol_ratio < 0.8) and (pvr < 0)
    if low_volume_down:
        score += bottom_boost
        reasons.append("無量下跌 → 偏多加分")

    # === 分數標準化 ===
    max_possible = sum(weights.values())
    final_score = np.clip(score / max_possible, -1, 1) * 100

    # 動態觀望閾值
    upper_thresh, lower_thresh = 30, -30
    if is_top: upper_thresh = 25
    if is_bottom: lower_thresh = -25

    # === 決策 ===
    if final_score >= upper_thresh:
        trand, label = 1, '進貨'
    elif final_score <= lower_thresh:
        trand, label = -1, '出貨'
    elif final_score > 0:
        trand, label = 0.5, '正觀望'
    else:
        trand, label = -0.5, '負觀望'

    reason = f"★ {label} ({final_score:+.1f}%) | " + ", ".join(reasons)

    if abs(row['diffPvr']) > abs(row['avgPvr'] * 2):
        df.at[idx, 'trand'] = trand
        df.at[idx, 'score'] = round(final_score, 2)
        df.at[idx, 'reason'] = reason

    return trand, final_score, reason


# ===================== 主流程 =====================
for master in codes:
    stock_code = master['stock_code']
    stock_name = master['stock_name']
    details = MySQL.get_price(stock_code, analyse_days, 'asc')
    details = humps.camelize(details)
    print(details)
    if not details:
        continue

    df = pd.DataFrame(details)
    df['volume'] = df['volume'] / 1000
    df['diffPrice'] = df['close'].diff().fillna(0)
    df['diffVolume'] = df['volume'].diff().fillna(0)
    # 異常主力
    df['diffPvr'] = np.where(df['diffVolume'] != 0, df['diffPrice'] / (df['diffVolume'] / 10000), 0)  # 差價量比
    df['avgPvr'] = df['diffPvr'].abs().rolling(window=10, min_periods=1).mean()  # 近10日平均差價量比
    # df['ampPvr'] = (df['diffPvr'] / df['avgPvr']).fillna(0).round(decimal_place)
    df['ampPvr'] = (
        (df['diffPvr'] / df['avgPvr']).replace([np.inf, -np.inf], 0).fillna(0)
    ).clip(-5, 5).round(decimal_place)
    df['tgtPrice'] = np.where(abs(df['diffPvr']) > abs(df['avgPvr']), df['close'].fillna(0), 0)  # 指標價格
    df['estPrice'] = df['tgtPrice'].where(df['tgtPrice'] > 0).expanding().mean().round(decimal_place)  # 平均指標價格
    df['avgPrice'] = df['close'].expanding().mean().round(decimal_place)
    df['avgVolume'] = df['volume'].expanding().mean().round(decimal_place)
    df['RSI'] = ta.rsx(df['close'], length=14)  # 指定window=14

    # 計算 KDJ
    kd = ta.stoch(high=df['high'], low=df['low'], close=df['close'], k=9, d=3, smooth_k=3)
    K = kd['STOCHk_9_3_3'].round(decimal_place)
    D = kd['STOCHd_9_3_3'].round(decimal_place)
    J = (kd['STOCHk_9_3_3'] - 2 * kd['STOCHd_9_3_3']).round(decimal_place)
    df['KDJ'] = list(zip(K, D, J))
    df['J'] = J
    df['prevKDJ'] = df['KDJ'].shift(1)

    # 計算MACD (含DIF/DEA)
    if len(df['close']) < 30:  # MACD慢線需要至少26個數值
        print(f"股票:{stock_code} 價格資料太少，無法計算 MACD")
        continue
    macd = ta.macd(df['close'])
    df['DIF'] = macd['MACD_12_26_9'].fillna(0)
    df['DEA'] = macd['MACDs_12_26_9'].fillna(0)
    df['MACD'] = macd['MACDh_12_26_9'].fillna(0)
    # df['DIF'], df['DEA'], df['MACD'] = calc_macd(df['close'])

    # 加權平均求MACD趨勢(3日趨勢)
    weights = np.arange(1, 4)  # [1,2,3,4,5]，越近越重
    df['MACD_5wma'] = df['MACD'].rolling(3).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    df['MACD_TREAD'] = np.where(df['MACD'] > df['MACD_5wma'], 1, np.where(df['MACD'] < df['MACD_5wma'], -1, 0))  # 判斷趨勢
    df.drop(columns=['MACD_5wma'], inplace=True)  # 刪掉中間欄位（可選）

    # 偵測金叉、死叉
    df['MACD_SIG'] = 0.0
    df.loc[(df['DIF'].shift(1) < df['DEA'].shift(1)) & (df['DIF'] > df['DEA']), 'MACD_SIG'] = 1  # 金叉
    df.loc[(df['DIF'].shift(1) > df['DEA'].shift(1)) & (df['DIF'] < df['DEA']), 'MACD_SIG'] = -1  # 死叉
    # 接近交叉（預警）
    diff = df['DIF'] - df['DEA']
    df.loc[(diff.between(-threshold_macd, 0)) & (df['DIF'] < df['DEA']), 'MACD_SIG'] = 0.5  # 接近金叉
    df.loc[(diff.between(0, threshold_macd)) & (df['DIF'] > df['DEA']), 'MACD_SIG'] = -0.5  # 接近死叉

    df['OBV'] = ta.obv(close=df['close'], volume=df['volume'])
    # TSI > 0 → 多方強勢，可考慮買入
    tsi_df = ta.tsi(df['close'], r=2, s=2)
    df['TSI'] = tsi_df.iloc[:, 0]  # 取第一欄

    # 計算均線
    df = calc_ma(df)
    # 拉高出貨 + 低位承接檢測
    df = detect_trade_signals(df, pct_thresh_up=2, pct_thresh_acc=2, vol_window=5, rsi='RSI', macd='MACD')

    # 計算每日增長率
    df['日增長率'] = df['close'].pct_change() * 100
    df['週增長率'] = df['close'].pct_change(periods=7) * 100
    df['累積增長'] = (1 + df['日增長率'] / 100).cumprod() * 100 - 100
    df['日增長率_%'] = df['日增長率'].fillna(0).round(1).astype(str) + '%'
    df['指數增長率'] = df['close'] / df['close'].shift(1)  # T/T-1
    # print("=== 每日增長率結果 ===")
    # print(df[['priceDate', 'close', '日增長率_%', '指數增長率']].round(3), "\n")
    # 預測未來 7 天
    predictions = predict_next(df, days=7)
    df['estClose'] = df['close']
    df = pd.concat([df, predictions.rename(columns={'日期': 'priceDate'})], ignore_index=True)

    # ===================== 判斷買賣時機 =====================
    for idx, row in df.iterrows():
        detect_rule3(idx, row, df)

    if display_df == 1:
        print(tabulate(df, headers='keys', tablefmt='simple', showindex=False, stralign='left', numalign='left'))
    elif display_df == 2:
        df_filtered = df[df['trand'].notna()]  # 選出 trand 不為 NaN 的列
        print(
            tabulate(df_filtered, headers='keys', tablefmt='simple', showindex=False, stralign='left', numalign='left'))
    if display_matplot:
        # plot_stock(stock_code, stock_name, df, df['estPrice'].iloc[-1], df['avgPrice'].iloc[-1])
        plot_stock(stock_code, stock_name, df)
print("指標股票")
for stock_code in rec_stocks:
    print(f"{stock_code}")
