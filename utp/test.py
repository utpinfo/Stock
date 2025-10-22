import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utp import MySQL  # 請確保 utp 模組可用

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

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
    future_dates = pd.date_range(df['price_date'].iloc[-1] + pd.Timedelta(days=1), periods=days, freq='D')

    result = pd.DataFrame({
        'price_date': future_dates,
        '預估close': future_y.round(0).astype(int)
    })

    print(f"=== 指數趨勢線方程式 ===")
    print(f"y = {a:.0f} × {b:.4f}^x")
    print("\n=== 未來預測 ===")
    print(result, "\n")

    return result

# ==================== 主程式 ====================

# 取得歷史資料
sales = MySQL.get_price('2464', 30, 'asc')
df = pd.DataFrame(sales)

# 計算每日增長率
df['日增長率'] = df['close'].pct_change() * 100
df['週增長率'] = df['close'].pct_change(periods=7) * 100
df['累積增長'] = (1 + df['日增長率'] / 100).cumprod() * 100 - 100
df['日增長率_%'] = df['日增長率'].fillna(0).round(1).astype(str) + '%'
df['指數增長率'] = df['close'] / df['close'].shift(1)

print("=== 每日增長率結果 ===")
print(df[['price_date', 'close', '日增長率_%', '指數增長率']].round(3), "\n")

# 預測未來 7 天
predictions = predict_next(df, days=7)

# ==================== 直接追加預測到 df ====================
# 新增預估欄位，先填入歷史 close
df['預估close'] = df['close']

# 將預測值追加到 df
df = pd.concat([df, predictions.rename(columns={'日期':'price_date'})], ignore_index=True)

# ==================== 視覺化 ====================
plt.figure(figsize=(14, 6))
plt.plot(df['price_date'], df['close'], 'o-', label='實際', linewidth=2)
plt.plot(df['price_date'], df['預估close'], 'r--', label='預測', linewidth=2)
plt.axvline(x=df['price_date'].iloc[len(df)-len(predictions)-1], color='gray', linestyle='--', alpha=0.6, label='預測起點')
plt.title('實際 vs 預測 close 趨勢', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
