from pymongo import MongoClient


# 创建mongodb的客户端
mongo_client = MongoClient(
    "mongodb://10.0.0.201:27017/",
)

# mongo数据库和表不需要提前创建，会根据字段自动创建

# 创建数据库
db = mongo_client["know_db"]

# 创建集合（表）
collection = db["users" ]

# 添加数据
def insert_data():
    # result返回生成的id，支持添加一条和多条数据
    # insert_one , insert_many
    result = collection.insert_one(
        {"name": "张三", "age": 18},
    )
    print(result)

if __name__ == "__main__":
    print(db)
    print(collection)

    insert_data()