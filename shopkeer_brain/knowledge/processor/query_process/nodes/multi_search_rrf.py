from typing import List, Tuple, Dict, Any

from shopkeer_brain.knowledge.processor.query_process.base import BaseNode
from shopkeer_brain.knowledge.processor.query_process.state import QueryGraphState


# 融合HyDE_search_node 和 vector_search_node的结果
class RrfSearchNode(BaseNode) :
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 从state中获取向量检索的结果和HyDE的检索结果
        vector_search_chunks  = state.get('embedding_chunks', [])
        hyde_search_chunks = state.get('hyde_embedding_chunks', [])
        # 2.使用RRF公式计算分数，根据分数排序
        # 2.1 构建相关数据，获取没路检索的chunk文档，设置每路检索的权重
        #{
        # "vector_search_result":([{chunk1},{chunk2},...],weight),
        # "hyde_search_result":([{chunk1},{chunk2},...],weight)
        # }
        search_source = {
            "vector_search_result": (self.get_chunk_list(vector_search_chunks), 1),
            "hyde_search_result": (self.get_chunk_list(hyde_search_chunks), 1)
        }
        # 为了方便计算
        # [([{chunk1},{chunk2},...],score),([{chunk1},{chunk2},...],score)]
        rrf_inputs = list(search_source.values())
        rrf_merge_result:List[Tuple[Dict[str,Any], float]] = self.rrf_merge(rrf_inputs)

        # 3.将最终结果放到列表，返回state
        rrf_chunks = [chunk for chunk, score in rrf_merge_result]
        state['rrf_chunks'] = rrf_chunks
        return state

    def get_chunk_list(self, rrf_inputs : List[Dict[str, Any]]) -> List[Dict[str,Any]]:
        result = []
        for doc in rrf_inputs:
            entity = doc.get('entity', doc)
            if not entity:
                continue
            result.append(entity)
        return result
    def rrf_merge(self, rrf_inputs) -> List[Tuple[Dict[str, Any], float]]:
        chunk_scores = {} # 存放chunk_id对应的分数
        chunk_data = {} # 存放chunk_id对应的文档数据
        for rrf_input,weight in rrf_inputs:
            for i,doc in enumerate(rrf_input):
                # 为多路检索相同文档做叠加使用
                chunk_id = doc.get('chunk_id')
                if not chunk_id:
                    continue
                rank = i+1
                # rrf计算
                # 权重/(奇数+排名) {chunk_id : score} , {}
                # 根据chunk_id查询chunk_scores字典，是否有chunk_id数据，有则叠加，没有则新增
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, float(0)) + weight / (self.config.rrf_k + rank)
                # 查一个 key，有就返回它的值；没有就按你给的默认值塞进去，再返回这个默认值。
                # 同一个文档被多次召回，也只记录一次
                chunk_data.setdefault(chunk_id, doc)
        # 排序，取分数最高的10个chunks
        # [(content, score), (content, score)...] x[1]代表取分数排序，x代表前面每个元组
        sorted_chunks = sorted(
            [(chunk_data[chunk_id], score) for chunk_id, score in chunk_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_chunks[:self.config.rrf_max_results]


