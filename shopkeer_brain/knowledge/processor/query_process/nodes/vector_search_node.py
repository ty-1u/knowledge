from typing import List

from shopkeer_brain.knowledge.processor.query_process.base import BaseNode
from shopkeer_brain.knowledge.processor.query_process.state import QueryGraphState
from shopkeer_brain.knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from shopkeer_brain.knowledge.utils.milvus_client_util import get_milvus_client, create_hybrid_search_requests, \
    execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    # 向量检索节点
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1.参数校验
        item_names, rewritten_query = self.validate_param(state)

        # 2.对重写问题进行向量化
        # 获取嵌入模型对象
        bgem3_client = get_bgem3_client()
        milvus_client = get_milvus_client()
        # rewritten_query向量化
        # {"dense":[], "sparse":[]}
        embedding_result = generate_hybrid_embeddings(bgem3_client, [rewritten_query])
        if not embedding_result:
            return {}
        # 3.构建问题向量化的向量条件
        item_name_filter_expr = self.create_item_name_filter(item_names)
        # 4.构建标量字段的条件
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result["dense"][0],
            sparse_vector=embedding_result["sparse"][0],
            # expr是标量字段的表达式
            # 相当于sql语句的where条件
            expr=item_name_filter_expr or None,
            limit=5
        )



        # 5.执行混合检索pymilvus的方法
        res = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_requests,
            norm_score=True,
            output_fields=["chunk_id","content", "item_name"],

        )
        if not res or not res[0]:
            return {}

        # 6.更新state数据
        return {"embedding_chunks": [hit["entity"] for hit in res[0]]}


    # 创建问题向量化的向量条件
    def create_item_name_filter (self, item_names:List[str]) -> str:
        # 宽松包含匹配，避免商品名存在细微差异时被全部过滤
        conditions = [f'item_name like "%{name}%"' for name in item_names]
        return " or ".join(conditions)

    # 参数校验
    def validate_param(self, state: QueryGraphState):
        item_names = state.get("item_names")
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise ValueError("Rewritten query is missing")
        if not item_names:
            raise ValueError("Item names are missing")
        return item_names, rewritten_query


