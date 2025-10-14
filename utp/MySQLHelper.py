import pymysql


class MySQLHelper:
    def __init__(self, host, user, password, database, port=3306):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.conn = None

    def connect(self):
        try:
            self.conn = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port,
                database=self.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            # print("Connected to MySQL database")
        except Exception as e:
            print("Connection error:", e)

    def execute_query(self, sql, params=None):
        if not self.conn:
            print("Not connected to database")
            return None

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, params)
                result = cursor.fetchall()
                return result
        except Exception as e:
            print("Query execution error:", e)
            return None

    def execute_insert_update(self, sql, params=None):
        if not self.conn:
            print("Not connected to database")
            return False

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, params)
                self.conn.commit()
                return True
        except Exception as e:
            self.conn.rollback()
            print("Insert/update execution error:", e)
            return False

    def close(self):
        if self.conn:
            self.conn.close()
            # print("Connection closed")


# 示例用法
'''
if __name__ == "__main__":
    # 创建 MySQLHelper 实例
    helper = MySQLHelper(host='localhost', user='root', password='', database='test')

    # 连接数据库
    helper.connect()

    # 执行查询语句
    result = helper.execute_query("SELECT * FROM users")
    print("Query result:", result)

    # 执行插入数据
    insert_sql = "INSERT INTO users (name, age) VALUES (%s, %s)"
    insert_params = ("John", 30)
    if helper.execute_insert_update(insert_sql, insert_params):
        print("Data inserted successfully")

    # 关闭连接
    helper.close()
'''
