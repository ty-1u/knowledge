# 切片的嵌入
from typing import List, Dict, Any

from shopkeer_brain.knowledge.processor.import_process.base import BaseNode
from shopkeer_brain.knowledge.processor.import_process.exceptions import ValidationError
from shopkeer_brain.knowledge.processor.import_process.state import ImportGraphState
from shopkeer_brain.knowledge.utils import get_bgem3_client


class ChunksEmbeddingNode(BaseNode):
    def process(self,state: ImportGraphState) -> ImportGraphState:
        # 1.参数校验
        chunks = self.validate_param(state)

        # 2.chunks内容进行嵌入
        # 批量嵌入，每batch_size段嵌入一次,例如3段
        # 定义批量需要的变量
        batch_size = 3
        total_length = len(chunks)
        final_chunks = []
        for i in range(0, total_length, batch_size):
            batch_data = chunks[i:i + batch_size]
            # 把每三段内容向量化
            # 这里得到的proc_result应该是一些chunk,每个chunk增加了稠密和稀疏的向量
            proc_result = self.exec_batch_chunks(batch_data, i ,total_length)
            final_chunks.extend(proc_result)


        # 3.更新state数据
        state["chunks"] = final_chunks


        return state


        # 添加密集向量和稀疏向量的值

    def exec_batch_chunks(self,batch_data:List[Dict[str, Any]], start_index: int, total_length: int) -> List[Dict[str, Any]]:
        # 实现逻辑：batch_data列表遍历，得到每段{}内容获取content数据，将content数据嵌入模型向量化，返回密集和稀疏向量
        final_contents = []
        for index, chunk in enumerate(batch_data)   :
            content = chunk.get("content")
            if not content:
                raise ValidationError("Chunk内容为空")
            # 嵌入模型向量化
            # 这里得到的proc_result应该是一些chunk,每个chunk增加了稠密和稀疏的向量

            # 规则：用item_name + content 构成嵌入内容，item_name 为空则不添加
            item_name = chunk.get("item_name")
            embed_content = f"{item_name}\n{content}" if item_name else content
            final_contents.append(embed_content)

        # 得到嵌入模型对象，调用方法
        client = get_bgem3_client()
        embed_result = client.encode_documents(final_contents)

        # 取出稠密和稀疏向量
        for index,chunk in enumerate(batch_data):
            dense_vector = embed_result.get("dense")[index].tolist()
            # 稀疏向量为 coo_array 格式，索引在 col，权重在 data
            csr_results = embed_result.get("sparse")[index]
            # 获取稀疏向量的 token_id 和权重值
            token_id = csr_results.col.tolist()
            data = csr_results.data.tolist()
            #转换成字典
            spare_vector = dict(zip(token_id,data))

            # 将密集和稀疏向量更新到chunk
            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = spare_vector
        return batch_data


        # 在batch_data里面增加稠密和稀疏向量的字段


    def validate_param(self,state :ImportGraphState ):
        chunks = state.get("chunks")
        if not chunks:
            raise ValidationError("Chunks为空")
        return chunks

