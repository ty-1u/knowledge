from typing import Any, Dict, List

from shopkeer_brain.knowledge.processor.query_process.base import BaseNode
from shopkeer_brain.knowledge.processor.query_process.state import QueryGraphState
from shopkeer_brain.knowledge.utils.bge_rerank_util import get_reranker_model


# rerank融合节点
class RerankSearchNode(BaseNode):
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1.获取融合数据
        # web搜索数据 + rrf融合数据
        merge_multi_doc : List[Dict[str,Any]] = self.merge_multi_data(state)



        # 2.获取输入的问题
        user_query = state.get("rewritten_query","") or state.get("original_query","")
        # 3.将输入的问题+数据 传入reranker
        reranker_doc:List[Dict[str,Any]] = self.rerank_mergerd_data(user_query, merge_multi_doc)

        # 4.断崖式检测，将上一步答案中分数低于阈值的答案过滤掉
        cutoff_doc = self.cliff_detection(reranker_doc)
        # 5.更新state返回
        state["reranked_docs"] = cutoff_doc
        return state

    # 合并web和rrf的数据
    def merge_multi_data(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        final_data = []
        rrf_chunks = state.get("rrf_chunks", [])
        for doc in rrf_chunks:
            content = doc.get("content", "").strip()
            if not content:
                continue
            chunk_id = doc.get("chunk_id","")
            title = doc.get("chunk_id","")
            chunk_dict_data = {
                "content": content,
                "chunk_id": chunk_id,
                "title": title,
            }
            final_data.append(chunk_dict_data)

        web_search_docs = state.get("web_search_docs",[])
        for web_doc in web_search_docs:
            content = web_doc.get("snippet","") or web_doc.get("content","")
            if not content:
                continue
            title = web_doc.get("title","")
            url = web_doc.get("url","")
            # 构建返回数据
            web_dict_data = {
                "content": content,
                "title": title,
                "url": url,
            }
            final_data.append(web_dict_data)
        return final_data


    # 用户问题 答案列表
    def rerank_mergerd_data(self, user_query: str, merge_multi_doc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 获取reranker模型对象
        reranker_model = get_reranker_model()

        # 传入问题和答案列表
        # reranker要求此格式
        # [(Q，A)，...]
        query_and_doc = [(user_query,doc.get("content")) for doc in merge_multi_doc]
        # 得到分数的值
        # [s1,s2,...]
        reranker_score = reranker_model.compute_score(sentence_pairs=query_and_doc)
        # 根据分数找文档，并排序
        # ["a","b"...]  [s1,s2,...] -> {"a":s1,"b":s2...}
        # [{doc,score}...] -> [{"url":"", "title":"a","content":"a","score":s1},{"url":"","title":"b","content":"b","score":s2...}]
        reranked_docs = [{**doc,"score":score} for doc, score in zip( merge_multi_doc,reranker_score)]
        # 排序
        sorted_reranked_docs = sorted(reranked_docs, key=lambda x: x["score"], reverse=True)
        return sorted_reranked_docs


    # 去掉相近分数相差过大的数据
    def cliff_detection(self, reranker_doc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 设置上下限 上限 10 下限 3
        upper_bound = min(10, len(reranker_doc))
        lower_bound = max(3, len(reranker_doc))

        cutoff_pos = upper_bound
        # 以下限为基准开始判断
        for i in range(lower_bound-1, upper_bound-1):
            # 当前位置的分数
            current_score = reranker_doc[i].get("score", 0)
            # 下一个位置的分数
            next_score = reranker_doc[i+1].get("score", 0)
            if current_score is None or next_score is None:
                continue

            # 计算差值
            score_diff = current_score - next_score

            # 相对落差百分比
            score_diff_percentage = score_diff / (abs(current_score) + 1e-6)

            if score_diff >= 0.5 or score_diff_percentage >= 0.25:
                cutoff_pos = i + 1
                break
        return reranker_doc[:cutoff_pos]