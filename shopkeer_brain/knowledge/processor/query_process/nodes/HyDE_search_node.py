from typing import List

from shopkeer_brain.knowledge.processor.query_process.base import BaseNode
from shopkeer_brain.knowledge.processor.query_process.state import QueryGraphState
from shopkeer_brain.knowledge.prompt.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from shopkeer_brain.knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from shopkeer_brain.knowledge.utils.llm_client_util import get_llm_client
from shopkeer_brain.knowledge.utils.milvus_client_util import create_hybrid_search_requests, \
    execute_hybrid_search_query, get_milvus_client


class HyDESearchNode(BaseNode):
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1.参数校验
        item_names , rewritten_query = self.validate_param(state)

        # 2.调用大模型，生成假设性文本
        hyde_doc = self.call_llm(item_names , rewritten_query)

        # 3.用户问题 + LLM返回答案进行拼接
        embedding_doc = f"{rewritten_query}\n{hyde_doc}"

        # 4.将拼接的文本进行向量化
        bgem3_client = get_bgem3_client()
        milvus_client = get_milvus_client()
        # rewritten_query向量化
        # {"dense":[], "sparse":[]}
        embedding_result = generate_hybrid_embeddings(bgem3_client, embedding_documents=[embedding_doc])
        if not embedding_result:
            return {}

        # 5.构建查询条件：仅使用向量
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result["dense"][0],
            sparse_vector=embedding_result["sparse"][0],
            limit=5
        )

        # 6.执行混合查询
        res = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_requests,
            norm_score=True,
            output_fields=["chunk_id","content", "item_name"],

        )
        if not res or not res[0]:
            return {}

        # 7.更新state
        return {"hyde_embedding_chunks": [hit["entity"] for hit in res[0]]}

    def call_llm(self, item_names:List[str], rewritten_query:str) -> str:
        # 获取llm连接对象
        client = get_llm_client()
        # 多个商品名用顿号拼接，避免把列表直接拼进提示词
        item_hint = "、".join(item_names)
        # 构建提示词
        system_prompt = f"你是一个{item_hint}的技术文档领域专家，主要擅长编写技术文档、操作手册、文档规格说明"
        user_prompt = USER_HYDE_PROMPT_TEMPLATE.format(
            item_hint=item_hint,
            rewritten_query=rewritten_query
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = client.invoke(messages)
        content = getattr(response, "content", "").strip()
        return content

    def validate_param(self, state: QueryGraphState):
        item_names = state.get("item_names")
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise ValueError("Rewritten query is missing")
        if not item_names:
            raise ValueError("Item names are missing")
        return item_names, rewritten_query
