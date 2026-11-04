# 切片向量化的数据存入milvus数据库
# 1.构建集合的约束
# 2.负责构建集合索引
# 3.添加数据到向量数据库
# 4. 节点类
import json
from typing import Any, Dict, List

from pymilvus import MilvusClient, DataType
from pymilvus import CollectionSchema

from shopkeer_brain.knowledge.processor.import_process import config
from shopkeer_brain.knowledge.processor.import_process.base import BaseNode
from shopkeer_brain.knowledge.processor.import_process.config import get_config
from shopkeer_brain.knowledge.processor.import_process.exceptions import ValidationError
from shopkeer_brain.knowledge.processor.import_process.state import ImportGraphState
from shopkeer_brain.knowledge.utils.milvus_client_utils import get_milvus_client


class _MilvusSchemaBuilder:
    @staticmethod
    def build( client:MilvusClient) -> CollectionSchema:
        # 当前字段示例
        # {
        #     "title": "3.1 混合召回",
        #     "content": "3.1 混合召回\n混合召回将稠密语义分与稀疏词项分加权求和：score = α·S_dense + (1-α)·S_sparse。",
        #     "file_title": "检索系统用户手册",
        #     "parent_title": "3. 检索接口",
        #     "part": "p2",
        #
        #     "dense_vector": [0.0412, -0.0521, 0.1238, "...共1024项..."],
        #
        #     "sparse_vector": {
        #         "33211": 0.2515,
        #         "18402": 0.1702,
        #         "44511": 0.3110,
        #         "11882": 0.2840,
        #         "99211": 0.3335
        #     }
        # }
        # 创建schema对象
        schema = client.create_schema(enable_dynamic_field=True)
        # 设置约束
        # 设置主键 auto_id = true 则随机生成不重复的id
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=100, auto_id=True)
        # 设置标量字段
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length =65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        #密集向量和稀疏向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR,dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        return schema

class _MilvusIndexBuilder:
    @staticmethod
    def build(client: MilvusClient,collection_name: str):
        index = client.prepare_index_params()

        # 稠密，稀疏向量建立索引
        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            # HNSW FLAT
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            # 倒排索引
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        return index

# 3.添加数据到向量数据库
# 更新chunks，增加主键值到数据中
class _MilvusInsertBuilder:
    def __init__(self,client:MilvusClient,collection_name:str):
        self.collection_name = collection_name
        self.client = client

    def insert(self,chunks:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        # 添加列表数据到向量数据库,返回的结果中带id信息

        # SDK中封装了循环，不用自己写
        insert_result = self.client.insert(
            # 指定表名
            collection_name=self.collection_name,
            # 指定数据
            data=chunks,

        )
        # ids是字典，包含插入数据的id信息
        ids = insert_result.get('ids')
        # 更新chunks，将id信息添加到chunks中
        self._add_ids(chunks, ids)
        return chunks

    # [{},{}]
    # [{}:k1,{}:k2,{}:k3]
    def _add_ids(self,chunks:List[Dict[str,Any]], ids:List[int]):
        for chunk,id in zip(chunks,ids):
            chunk["chunk_id"] = id

# 调用节点
# 主题调用过程
class ImportMilvusNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.参数检验
        chunks, file_title = self.validate_param(state)

        # 2.获取milvus连接对象和集合名称
        client = get_milvus_client()
        config = get_config()
        collection_name = config.chunks_collection

        # 3.判断集合是否存在，不存在则创建
        collection_exists = self.has_collection(client,collection_name)

        # 同一文档重复导入时，先清理旧 chunk，避免重复累积
        if collection_exists:
            client.delete(
                collection_name=collection_name,
                filter=f"file_title == {json.dumps(file_title, ensure_ascii=False)}",
            )

        # 4.添加
        milvus_insert_obj = _MilvusInsertBuilder(client = client,collection_name=collection_name)
        final_chunks = milvus_insert_obj.insert(chunks=chunks)

        # 更新state
        state["chunks"] = final_chunks
        return state

    def has_collection(self,client:MilvusClient,collection_name:str):
        if not client.has_collection(collection_name):
            # 创建集合约束
            schema = _MilvusSchemaBuilder.build(client)
            # 创建集合索引
            index = _MilvusIndexBuilder.build(client,collection_name)
            client.create_collection(collection_name=collection_name, schema=schema, index_params=index)
            return False
        return True


    def validate_param(self, state: ImportGraphState):
        chunks = state.get("chunks")
        if not chunks:
            raise ValidationError("Chunks is required")
        return chunks, state.get("file_title", "")
