# 首次運行
1. 安裝依賴
pip install -r requirements.txt
2. 下載股票列表
TwseUtils.py
3. 修改狀態
```sql
update stock
   set stock_status = '90'
 where stock_code in ('2610', '4974', '6176');
```