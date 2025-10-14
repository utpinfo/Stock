def is_trend(seq):
    n = len(seq)
    total_increase = 0
    total_decrease = 0
    for i in range(1, n):
        weight = i / (n * (n + 1) / 2)   # 越後面的權重越大
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

# 測試範例
sequence1 = [ -0.24831772, -2.24391239, -1.72928533, -1.25977772]

print(is_trend(sequence1))  # 遞增趨勢