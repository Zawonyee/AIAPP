import pymysql

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",  # 替换为你的数据库密码
    "db": "object_tracking"
}

def execute_query(query, params=None):
    """执行SQL查询"""
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            connection.commit()
            return cursor.fetchall()
    finally:
        connection.close()

def store_object_location(object_name, x1, y1, x2, y2):
    """存储物品位置信息"""
    query = """
    INSERT INTO object_locations (object_name, x1, y1, x2, y2)
    VALUES (%s, %s, %s, %s, %s)
    """
    execute_query(query, (object_name, x1, y1, x2, y2))

def get_latest_object_location(object_name):
    """获取物品的最新位置信息"""
    query = """
    SELECT * FROM object_locations
    WHERE object_name=%s
    ORDER BY timestamp DESC
    LIMIT 1
    """
    return execute_query(query, (object_name,))