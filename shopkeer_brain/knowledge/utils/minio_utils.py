# 创建获取minio客户端的方法
import os
from minio import Minio


def get_minio_client():
    try:
        client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT"),
            # 用户名 和 密码
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=False, # http访问
        )
        # 自动创建bucket
        # client.make_bucket()
        return client

    except Exception as e:
        print(e)
        raise e