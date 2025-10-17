from utp import MySQL

data = MySQL.get_price(1516, None, 'asc','2025-10-16','2025-10-17')
print(data)