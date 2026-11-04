import os

from dotenv import load_dotenv

load_dotenv()


def get_milvus_client():
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri = os.getenv("MILVUS_URL"),

        )
        return client

    except Exception as e:
        raise e