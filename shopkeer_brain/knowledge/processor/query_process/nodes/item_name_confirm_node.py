import json
import os
from typing import Dict, Any, List, Tuple

import re

from shopkeer_brain.knowledge.processor.query_process.base import BaseNode
from shopkeer_brain.knowledge.processor.query_process.state import QueryGraphState
from shopkeer_brain.knowledge.prompt.query.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from shopkeer_brain.knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from shopkeer_brain.knowledge.utils.llm_client_util import get_llm_client
from shopkeer_brain.knowledge.utils.milvus_client_util import get_milvus_client, create_hybrid_search_requests, \
    execute_hybrid_search_query
from shopkeer_brain.knowledge.processor.query_process.config import get_config

from shopkeer_brain.knowledge.utils.mongo_history_util import get_recent_messages


# 操作大语言模型类
class ItemNameLLM():
    # 根据用户原始问题+上下文记录，提取商品名称
    def extract_item_name(self, original_query:str, history:str) -> Dict[str,Any]:
        # 获取llm连接对象
        llm_client = get_llm_client(response_format=True)
        # 构建提示词
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history, query=original_query)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},

        ]
        response = llm_client.invoke(messages)
        # reponse是llm返回的原始内容
        content = response.content.strip()
        # ```json ``` 对llm_content进行清洗与格式化
        parse_result = self.clean_parse(content)
        if not parse_result.get("item_names"):
            # 兜底提取产品型号，避免 LLM 把 HAK180 这类型号忽略为空
            model_codes = re.findall(
                r"(?i)\b[a-z]{2,}[-\s]?\d{2,}[a-z0-9-]*\b",
                original_query,
            )
            parse_result["item_names"] = list(dict.fromkeys(model_codes))
        return parse_result

    def clean_parse(self,llm_content) -> Dict[str,Any]:
        # ```json   ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 反序列
        parsed_llm_result:Dict[str,Any] = json.loads(content)

        #去掉parsed_llm_result 的item_names空格
        ori_item_names = parsed_llm_result.get('item_names')
        cleaned_item_names = [ori_item for ori_item in ori_item_names if ori_item.strip()]

        return {
            'item_names': cleaned_item_names,
            'rewritten_query': parsed_llm_result.get('rewritten_query')
        }


# 类：操作向量数据库
class VectorDatabase():
    # 1.根据item_name查询向量数据库，得到密集和稀疏向量数据
    # 2.评分对齐
    # 3.分数差异过滤
    # 最终返回两个列表，一个是主要参考来源，分数大于0.7，另一个是候选来源，分数在0.6-0.7之间
    def query(self, item_names:List[str]) -> Tuple[List[str], List[str]]:
        # 1.根据item_name查询向量数据库，得到密集和稀疏向量数据
        search_result:List[Dict[str, Any]] = self.match_vector(item_names)
        # 2.评分对齐，将密集和稀疏向量评分对齐，然后重新排序
        confirmed, options = self.item_name_score_alignment(search_result)

        # 3.分数差异过滤
        confirmed, options = self.score_difference_filtering(confirmed, options, search_result)
        # 最终返回两个列表，一个是主要参考来源，分数大于0.7，另一个是候选来源，分数在0.6-0.7之间
        return confirmed, options

    def match_vector(self, item_names:List[str]) -> List[Dict[str, Any]]:
        # 根据item_name查询向量数据库，得到密集和稀疏向量数据

        # 定义列表
        search_result = []
        # 得到milvus的连接对象
        milvus_client = get_milvus_client()
        # 读取混合检索的稠密/稀疏权重配置
        config = get_config()
        # 将item_name 的向量取出
        embedding_model = get_bgem3_client()
        # 通过embedding_model转换稠密和稀疏向量
        # 使用工具类实现
        hybrid_result = generate_hybrid_embeddings(embedding_model=embedding_model, embedding_documents=item_names)

        # ['商品1','商品2']
        # {"dense": [商品1向量，商品2向量], "sparse": [商品1向量，商品2向量]}

        item_name_collection = os.getenv("ITEM_NAME_COLLECTION")
        stored_item_names = milvus_client.query(
            collection_name=item_name_collection,
            filter="",
            output_fields=["item_name"],
            limit=1000,
        )
        stored_item_names = list(dict.fromkeys(
            row.get("item_name")
            for row in stored_item_names
            if row.get("item_name")
        ))
        # 获取每个item_name
        for index, item_name in enumerate(item_names):
            dense_vector = hybrid_result["dense"][index]
            sparse_vector = hybrid_result["sparse"][index]
            # 构建查询条件
            # 将两个向量转换为指定类型AnnSearchRequest
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector
            )

            # 使用方法执行查询
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=item_name_collection,
                search_requests=hybrid_search_requests,
                ranker_weights=(config.item_name_dense_weight, config.item_name_sparse_weight),
                # 这一步只用到item_name字段，其他字段没用到不需要输出，避免过大的返回
                output_fields=["item_name"],
                norm_score=True
            )
            # 构建返回数据
            # [
            # [{
            #       'pk':'xxxx',
            #        'distance' :xxxx,
            #       'entity': {
            #           'item_name': 'xxxx'
            #       }
            # }]
            # ]
            match_by_name = {}
            for hit in (hybrid_search_result[0] if hybrid_search_result[0] else []):
                matched_name = hit["entity"]["item_name"]
                matched_score = float(hit["distance"])
                if matched_name not in match_by_name or matched_score > match_by_name[matched_name]:
                    match_by_name[matched_name] = matched_score

            # 对型号编码做规范化词面匹配，例如 hak180 -> HAK 180 烫金机
            normalized_query = self.normalize_item_name(item_name)
            if len(normalized_query) >= 3:
                for stored_name in stored_item_names:
                    normalized_stored = self.normalize_item_name(stored_name)
                    if normalized_query in normalized_stored or normalized_stored in normalized_query:
                        match_by_name[stored_name] = max(
                            match_by_name.get(stored_name, 0.0),
                            0.95,
                        )

            matches = [
                {"item_name": matched_name, "score": score}
                for matched_name, score in sorted(
                    match_by_name.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]

            item_name_search_result = {
                "item_name": item_name,
                "matches": matches,
            }
            search_result.append(item_name_search_result)
        return search_result

    @staticmethod
    def normalize_item_name(item_name: str) -> str:
        return re.sub(r"[\s\-_·.,，。()（）]+", "", str(item_name or "").lower())

    def item_name_score_alignment(self, search_result:List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        # 定义两个列表
        confirmed = [] # 分数大于0.7
        options = [] # 0.6-0.7
        for item_name_result in search_result:
            # 获取提取出的商品名称
            extract_name = item_name_result.get('item_name')
            # 获取matches中的列表，把这个列表按照分数进行降序排序
            sorted_matches = sorted(item_name_result.get('matches'), key=lambda x: x['score'], reverse=True)
            high = [m for m in sorted_matches if m['score'] > 0.7]
            if high: # 评分大于0.7的数据
                # 只有一个数据
                if len(high) == 1:
                    # 直接放入confirmed
                    high_item_name_one = high[0]['item_name']
                    if high_item_name_one not in confirmed:
                        confirmed.append(high_item_name_one)
                # 有多条数据
                else :
                    for h in high[:3]:
                        high_item_name_many = h['item_name']
                        if high_item_name_many not in options and high_item_name_many not in confirmed:
                            options.append(high_item_name_many)

            else :
                middle = [m for m in sorted_matches if 0.6 <= m['score'] < 0.7]
                if middle: # > 0.6 <0.7 放入options
                    for m in middle[:3]:
                        middle_item_name = m['item_name']
                        if middle_item_name not in options and middle_item_name not in confirmed:
                            options.append(middle_item_name)
        # confirmed：大于0.7的一个数据
        # options：大于0.7的多个数据 0.6-0.7的数据，让用户选择
        return confirmed, options
    def score_difference_filtering(self, confirmed:List[str], options:List[str], search_result:List[Dict[str, Any]], score_gap_threshold: float = 0.15) -> Tuple[List[str], List[str]]:
        # 从 search_result 中找回每个确认商品名对应的最高分数
        score_map: Dict[str, float] = {}
        for item_name_result in search_result:
            for match in item_name_result.get("matches", []):
                name = match.get("item_name")
                score = float(match.get("score", 0.0))
                if name in confirmed or name in options:
                    if name not in score_map or score > score_map[name]:
                        score_map[name] = score

        # 基准分数：有 confirmed 时以 confirmed 的最高分为准，否则退回 options 的最高分
        reference_names = confirmed or options
        if not reference_names:
            return [], []
        top_score = max(score_map.get(name, float("-inf")) for name in reference_names)

        def _within_gap(name: str) -> bool:
            return (top_score - score_map.get(name, float("-inf"))) <= score_gap_threshold

        # 按分数降序排列，并保留与基准差距不超过阈值的商品名
        filtered_confirmed = [
            name for name in sorted(confirmed, key=lambda name: score_map.get(name, float("-inf")), reverse=True)
            if _within_gap(name)
        ]
        filtered_options = [
            name for name in sorted(options, key=lambda name: score_map.get(name, float("-inf")), reverse=True)
            if _within_gap(name)
        ]
        return filtered_confirmed, filtered_options

# 商品名确认节点
class ItemNameConfirmNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.item_name_llm = ItemNameLLM()
        self.item_name_vector = VectorDatabase()

    def process(self,state:QueryGraphState) -> QueryGraphState:
        # 1.获取用户输入的原始问题
        original_query = state.get("original_query")

        # 2.获取用户上下文会话
        session_id = state.get("session_id")
        # 调用工具类方法，取出用户历史会话
        chat_history = get_recent_messages(session_id,limit=10)
        # 把列表转换成字符串
        history = ""
        for message in chat_history:
            role = message.get("role")
            text = message.get("text")
            history = history + f"{role}: {text}\n"

        # 3.根据用户输入原始问题+上下文会话调用LLM,提取商品名
        llm_result = self.item_name_llm.extract_item_name(original_query, history)


        # 4. 根据LLM提取商品名
        # 查询得到密集和稀疏向量，评分对齐，分数差异化过滤
        item_names = llm_result.get("item_names")
        rewritten_query = llm_result.get("rewritten_query")


        # 判断
        if item_names:
            # 调用封装好的query 来处理
            confirmed, options = self.item_name_vector.query(item_names)
        else:
            # llm未提取出商品名
            confirmed, options = [], []

        # 5. 更新state
        self.update_state(state,item_names=item_names,confirmed=confirmed,options=options,rewritten_query= rewritten_query)

        state["history"] = history
        return state
    def update_state(self,state,item_names,confirmed,options,rewritten_query):
        if confirmed:
            state['item_names'] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            state['answer'] = f"请选择具体的商品名:{','.join(options)}"
        else:
            # 未提取出商品名
            state['answer'] = "当前问题无法识别，清提供更多详细信息"
