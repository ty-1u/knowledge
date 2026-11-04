import json
import os
from typing import Tuple, List, Any, Dict

from pymilvus import DataType

from shopkeer_brain.knowledge.processor.import_process import config
from shopkeer_brain.knowledge.processor.import_process.base import BaseNode
from shopkeer_brain.knowledge.processor.import_process.config import get_config
from shopkeer_brain.knowledge.processor.import_process.import_prompt import ITEM_NAME_SYSTEM_PROMPT, \
    ITEM_NAME_USER_PROMPT_TEMPLATE
from shopkeer_brain.knowledge.processor.import_process.state import ImportGraphState
from shopkeer_brain.knowledge.utils import get_bgem3_client
from shopkeer_brain.knowledge.utils.llm_client_utils import get_llm_client
from shopkeer_brain.knowledge.utils.milvus_client_utils import get_milvus_client


class ItemNameRecognition(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1.校验并获取参数
        file_title,chunks  = self.validate_param(state)

        # 2.构建LLM需要生成提示词数据
        # 从chunks中获取需要的数据
        prompt_context_data = self.get_prompt_context_data(chunks)

        # 3.构建提示词调用LLM，提取商品名
        item_names = self.call_llm(file_title,prompt_context_data)

        # 4.商品名的嵌入，利用bge-m3生成稠密和稀疏向量
        dense , sparse = self.embbeding_item_name(item_names)

        # 5.存储到milvus向量数据库
        self.save_to_milvus(file_title,item_names,dense,sparse)

        # 6.更新state["item_name"]，返回数据
        self.update_state(state,item_names, chunks)

        return state


        pass

    def validate_param(self,state: ImportGraphState) -> Tuple[str, List[Dict[str,Any]]]:
        # 从state中获取需要校验的值
        file_title =  state["file_title"]
        chunks = state["chunks"]
        if not file_title or not chunks:
            raise ValueError("file_title or chunks is None")
        return file_title, chunks

    # [{content:111,file_title:aaa},...]
    def get_prompt_context_data(self, chunks:List[Dict[str,Any]], method:str="structure", **kwargs) -> str:
        configuration = config.get_config()
        threshold = configuration.max_content_length
        # 从chunks列表中获取一部分数据
        # 计算总字符数
        total_chars = sum(len(c.get("content","")) for c in chunks)

        # 不超过阈值，直接拼接全部内容
        if total_chars <= threshold:
            return "\n\n".join(c.get("content","") for c in chunks)

        # 超过阈值，根据 method 选择采样方式，默认普通方法
        if method == "cluster":
            return self.sample_by_cluster(chunks, **kwargs)
        return self.sample_by_structure(chunks, **kwargs)


    def call_llm(self,file_title:str, prompt_context_data:str) -> str:

        # RAFT ，role,action,format,tone
        prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=prompt_context_data)
        message = [
            {"role": "system", "content": ITEM_NAME_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        client = get_llm_client()
        response = client.invoke(message)
        # 返回最终获得的response中的content内容,取不到则默认为空
        item_name = getattr(response, "content","")
        if not item_name :
            self.logger.warning("获取商品名称出错，返回文件名称")
            return file_title
        return item_name
    def embbeding_item_name(self,item_names:str):
        # 得到bge-m3的客户端对象
        client = get_bgem3_client()
        # 对item_name生成向量,要求传入列表[[],[],[]...],可以直接生成稠密和稀疏向量
        embed = client.encode_documents([item_names])
        # 获取稠密和稀疏向量并返回
        # dense[[]]，我们前面仅传入一个列表
        # tolist将numpy类型转成纯 Python 的嵌套列表
        # sparse对应稀疏矩阵，token编号 data权重，格式为返回两个列表 [1 , 5, 6], [0.1, 0.2, 0.3],期望用字典返回 --> {1:0.1,5:0.2,6:0.3}
        dense = embed["dense"][0].tolist()
        sp = embed["sparse"][0]
        #获取编号和权重
        sp_indices = sp.col.tolist()
        sp_data = sp.data.tolist()
        sparse = dict(zip(sp_indices, sp_data))

        return dense, sparse
    def save_to_milvus(self,file_title, item_names,dense,sparse):
        # 基于商品名称建立客户端
        client = get_milvus_client()

        # 判断向量数据库是否有集合，无则创建
        config = get_config()
        # 幂等性，保证只创建一次
        if not client.has_collection(collection_name=config.item_name_collection):
            self.create_item_collection(client, collection_name=config.item_name_collection)

        # 同一文档重复导入时，先清理旧商品名记录
        client.delete(
            collection_name=config.item_name_collection,
            filter=f"file_title == {json.dumps(file_title, ensure_ascii=False)}",
        )

        # 将数据添加
        result = client.insert(collection_name=config.item_name_collection, data={"file_title": file_title, "item_name": item_names, "dense_vector": dense, "sparse_vector": sparse})
        self.logger.info(f"向量数据库添加结果：{result}")

    def create_item_collection(self, client, collection_name):
        # 构建schema
        schema  = client.create_schema()
        # 设置主键 auto_id = true 则随机生成不重复的id
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=100, auto_id=True)
        # 设置其他字段
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length =1000)
        #密集向量和稀疏向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR,dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 添加索引
        # HNSW 倒排
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_name = "dense_vector_index",index_type="AUTOINDEX", metric_type = "COSINE")
        index_params.add_index(field_name="sparse_vector", index_name = "sparse_vector_index",index_type="SPARSE_INVERTED_INDEX", metric_type = "IP")

        #创建collecion
        client.create_collection( collection_name=collection_name, schema=schema, index_params=index_params)

    def update_state(self,  state: ImportGraphState, item_names:str, chunks:List[Dict[str,Any]]):
        # 为state和chunks都增加一个item_name的key值
        for chunk in chunks:
            # 为chunk增加一个key
            chunk["item_name"] = item_names
        state["item_name"] = item_names
        state["chunks"] = chunks

    # 保证首段的完整性，后续每段都切200字符，尾端切1000字符，尾端不保证都有
    def sample_by_structure(self, chunks, head_tail_full=1, head_tail_chars=1000, mid_chars=200, budget=4000):
        total = len(chunks)
        samples = []
        for idx, chunk in enumerate(chunks):
            title = chunk.get("title", "")
            content = chunk.get("content", "")
            if title:
                samples.append(title)
            # 首尾限长，中间只取开头
            if idx < head_tail_full or idx >= total - head_tail_full:
                piece = content[:head_tail_chars]
            else:
                piece = content[:mid_chars]
            samples.append(piece)
        result = "\n\n".join(samples)
        return result[:budget]

    def sample_by_cluster(self, chunks: List[Dict[str, Any]], n_clusters: int = None, max_chars: int = 500, budget: int = 4000) -> str:
        # 进阶采样：BGE-M3 向量聚类，每个簇取离中心最近的 chunk，返回拼接文本
        import numpy as np
        from sklearn.cluster import KMeans

        texts = [c.get("content", "") for c in chunks]
        if not texts:
            return ""

        ef = get_bgem3_client()
        dense = np.asarray(ef.encode_documents(texts)["dense"], dtype="float32")

        n = n_clusters or max(1, min(len(texts), int(len(texts) * 0.3)))
        n = min(n, len(texts))
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(dense)

        selected = []
        for i in range(n):
            idxs = [j for j, lab in enumerate(labels) if lab == i]
            if not idxs:
                continue
            center = kmeans.cluster_centers_[i]
            best = min(idxs, key=lambda j: float(np.linalg.norm(dense[j] - center)))
            selected.append(best)
        selected.sort()

        # 每个选出的 chunk 限长，再整体按预算截断
        result = "\n\n".join(chunks[j].get("content", "")[:max_chars] for j in selected)
        return result[:budget]
