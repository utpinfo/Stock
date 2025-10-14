from MySQLHelper import MySQLHelper


def get_last_trade_date(stock_code, year, month):
    helper = MySQLHelper(host='127.0.0.1', user='root', password='', database='stock')
    helper.connect()
    sql = "select max(price_date) price_date from price where stock_code=%s and year(price_date) = %s and month(price_date) = %s"
    params = (stock_code, year, month)
    data = helper.execute_query(sql, params)
    helper.close()
    return data


def get_stock(stock_status='10', stock_code=None):
    helper = MySQLHelper(host='127.0.0.1', user='root', password='', database='stock')
    helper.connect()
    sql = "select * from stock"
    if stock_code:
        sql += " where stock_code = %s"
        params = (stock_code)
    else:
        sql += " where stock_status = %s"
        params = (stock_status)
    data = helper.execute_query(sql, params)
    helper.close()
    return data


def add_stock(stock_code, stock_name, stock_kind, isin_code):
    helper = MySQLHelper(host='127.0.0.1', user='root', password='', database='stock')
    helper.connect()
    sql = """
        insert into stock
            (stock_code, stock_name, stock_kind, isin_code, stock_status)
        values (%s, %s, %s, %s, '10')
            on duplicate key update stock_name = values(stock_name), stock_kind = values(stock_kind), isin_code = values(isin_code), stock_status = values(stock_status)
    	"""
    params = (stock_code, stock_name, stock_kind, isin_code)
    helper.execute_insert_update(sql, params)
    helper.close()


def get_price(stock_code, limit, sort='asc'):
    helper = MySQLHelper(host='127.0.0.1', user='root', password='', database='stock')
    helper.connect()
    sql = "select stock_code, price_date, price, volume from (select * from price where stock_code = %s order by price_date desc limit %s) as t order by t.price_date " + sort
    params = (stock_code, limit)
    data = helper.execute_query(sql, params)
    helper.close()
    return data


def add_price(stock_code, price_date, price, volume=None):
    helper = MySQLHelper(host='127.0.0.1', user='root', password='', database='stock')
    helper.connect()
    sql = """
    insert into price
        (stock_code, price_date, price,volume)
    values (%s, %s, %s, %s)
        on duplicate key update price = values (price), volume = values(volume)
	"""
    params = (stock_code, price_date, price, volume)
    helper.execute_insert_update(sql, params)
    # if helper.execute_insert_update(sql, params):
    #    print("Data inserted successfully")
    helper.close()


def get_revenue():
    helper = MySQLHelper(host='127.0.0.1', user='root', password='', database='stock')
    helper.connect()
    sql = "select * from revenue"
    data = helper.execute_query(sql)
    helper.close()
    return data


def add_revenue(stock_code, revenue_date, revenue):
    helper = MySQLHelper(host='127.0.0.1', user='root', password='', database='stock')
    helper.connect()
    sql = """
    insert into revenue
        (stock_code, revenue_date, revenue)
    values (%s, %s, %s)
        on duplicate key update revenue = values (revenue)
	"""
    params = (stock_code, revenue_date, revenue)
    helper.execute_insert_update(sql, params)
    # if helper.execute_insert_update(sql, params):
    #    print("Data inserted successfully")
    helper.close()


def get_eps():
    helper = MySQLHelper(host='127.0.0.1', user='root', password='', database='stock')
    helper.connect()
    sql = "select * from eps"
    data = helper.execute_query(sql)
    helper.close()
    return data


def add_eps(stock_code, eps_date, eps):
    helper = MySQLHelper(host='127.0.0.1', user='root', password='', database='stock')
    helper.connect()
    sql = """
    insert into eps
        (stock_code, eps_date, eps)
    values (%s, %s, %s)
        on duplicate key update eps = values (eps)
	"""
    params = (stock_code, eps_date, eps)
    helper.execute_insert_update(sql, params)
    # if helper.execute_insert_update(sql, params):
    #    print("Data inserted successfully")
    helper.close()
