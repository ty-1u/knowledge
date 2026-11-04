# 构建知识图谱
import concurrent
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict, Any

from pymilvus import MilvusClient, DataType

from shopkeer_brain.knowledge.processor.import_process.base import BaseNode
from shopkeer_brain.knowledge.processor.import_process.config import get_config
from shopkeer_brain.knowledge.processor.import_process.import_prompt import KNOWLEDGE_GRAPH_SYSTEM_PROMPT
from shopkeer_brain.knowledge.processor.import_process.state import ImportGraphState
from shopkeer_brain.knowledge.utils import get_bgem3_client
from shopkeer_brain.knowledge.utils.llm_client_utils import get_llm_client
from shopkeer_brain.knowledge.utils.milvus_client_utils import get_milvus_client
from shopkeer_brain.knowledge.utils.neo4j_util import _Neo4jGraphWriter, get_neo4j_driver

# 实体标签白名单
ALLOWED_ENTITY_LABLES: set[str] = {
    "Device","Part","Operation","Step","Warning","Condition","Tool",
}
# 实体名称最大长度
MAX_ENTITY_LENGTH = 20
# 关系类型白名单
ALLOWED_RELATION_TYPES: set[str] = ({
    "HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
    "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES",
    "MENTIONED_IN", "RELATED_TO",
})
DEFAULT_RELATION_TYPE ="RELATED_TO"

# 操作milvus的内部类
class _MilvusOperator:
    # 类似于c++里面的属性，后续方法可以选择调用属性，也可以不用
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    # 根据item_name进行删除
    def clear(self, item_name: str ,client: MilvusClient):
        # 集合是否存在
        collection_name = self.collection_name
        if client.has_collection(collection_name):
            # 调用client的删除方法
            client.delete(collection_name=collection_name, filter=f"item_name == '{item_name}'")


    def insert(self, entities: List[Dict[str, Any]], chunk_id: str, item_name: str, content: str   , client: MilvusClient):
        # 去重，得到entity_names
        entity_names = list(dict.fromkeys(entity['name'] for entity in entities if entity['name']))
        if not entity_names:
            raise Exception("No valid entity names found in entities.")
        # 判断集合是否存在，不存在则创建
        self._ensure_collection(client, self.collection_name)

        # 获取向量client
        bgem3_client = get_bgem3_client()
        embedding_result = bgem3_client.encode_documents(entity_names)

        #调用方法构建结果
        records = self._build_records(
            entities_names=entity_names,
            embedded_result=embedding_result,
            chunk_id=chunk_id,
            content=content,
            item_name=item_name,
        )
        client.insert(collection_name=self.collection_name, data=records)


    @staticmethod
    def _build_records(
                entities_names: List[str],
                embedded_result: Dict[str, Any],
                chunk_id: str,
                content: str,
                item_name: str,
        ) -> List[Dict[str, Any]]:
            """组装插入记录。"""

            # 1. 校验嵌入结果
            if not embedded_result:
                raise ValueError("嵌入结果为空")

            # 2. 获取稠密向量和稀疏向量
            dense_vector_list = embedded_result.get("dense")
            sparse_list = embedded_result.get("sparse")

            # 3. 校验向量是否存在
            if not dense_vector_list or sparse_list is None:
                raise ValueError("参数校验失败，向量不存在")

            # 4. 获取对应块的部分内容作为上下文
            context = content[:200]
            records: List[Dict] = []

            # 5. 遍历每一个实体名，构建记录
            for idx, entity_name in enumerate(entities_names):
                # 5.1 边界检查
                if idx >= len(dense_vector_list):
                    break

                # 5.2 获取稠密向量
                dense = dense_vector_list[idx].tolist()

                # 5.3 解构稀疏向量（coo_array，索引在 col，权重在 data）
                sp = sparse_list[idx]
                indices = sp.col.tolist()
                data = sp.data.tolist()
                sparse_dict = dict(zip(indices, data))

                # 5.4 构建单条记录
                record = {
                    "entity_name": entity_name,
                    "context": context,
                    "item_name": item_name,
                    "source_chunk_id": chunk_id,
                    "dense_vector": dense,
                    "sparse_vector": sparse_dict,
                }

                records.append(record)

            return records
    def _ensure_collection(self, client, collection_name: str) -> None:
            """集合不存在则创建（schema + 索引）。"""

        # 1. 判断集合是否已存在
            if client.has_collection(collection_name):
                return

            # 2. 构建 schema
            schema = client.create_schema(enable_dynamic_field=True)
            schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field("entity_name", DataType.VARCHAR, max_length=65535)
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("source_chunk_id", DataType.VARCHAR, max_length=65535)
            schema.add_field("context", DataType.VARCHAR, max_length=65535)
            schema.add_field("item_name", DataType.VARCHAR, max_length=65535)

            # 3. 构建索引
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128},
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )

            # 4. 创建集合
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

class KnowledgeGraphNode(BaseNode):
    def __init__(self):
        # 让父类的init也执行
        super().__init__()
        config = get_config()
        self.config = config
        self._milvus_operator = _MilvusOperator(collection_name=config.entity_name_collection)
        self.client = get_milvus_client()
        self._Neo4j_obj = _Neo4jGraphWriter(
            config.neo4j_database

        )
        self.llm_client = get_llm_client()
        self.neo4j_driver = get_neo4j_driver()
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.参数校验
        # 获取切片数据
        chunks,item_name = self.validate_param(state)

        # 2. 删除已经存在的数据，包括milvus和neo4j中的数据
        self._milvus_operator.clear(item_name=item_name, client=self.client)
        self._Neo4j_obj.clear(self.neo4j_driver, item_name)
        # 3. 处理所有切片数据
        # 通过item_name查询是否已经存在数据
        # 构建LLM需要的提示词
        # 将构建的提示词提交LLM，提取知识图谱（entity
        # label
        # relation）
        # 对LLM提取的知识图谱的数据进行清洗
        # 将相关数据写入Milvus向量数据库，并写入neo4j图数据库（双写）

        self.execute_all_chunks(chunks)
        # 可选 ：实现多线程版本处理知识图谱数据
        # self.concurrent_execute_chunks(chunks)

        return state

    def concurrent_execute_chunks(self, chunks: List[Dict[str, Any]]):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            # 遍历chunks得到每个chunk，获取数据
            for i,chunk in enumerate(chunks):
                # 对每个切片进行处理
                if not isinstance(chunk, dict):
                    self.logger.warning(f"第 {i} 个 chunk 不是字典类型，已抛弃。")
                    continue
                chunk_id = chunk.get("chunk_id")
                item_name = chunk.get("item_name")
                content = chunk.get("content")
                # 向线程池值提交任务
                #submit(func,args)
                future = executor.submit(self.process_single_chunk, chunk_id, item_name, content)

            #     # 可选：获取每个任务的返回结果
            #     futures[future] = (chunk_id, item_name)
            # # 等待所有任务完成
            # #每有一个执行成功迭代器就会立即返回结果
            # for future in as_completed(futures):
            #     chunk_id, item_name = futures[future]
            #     try:
            #         result = future.result()
            #         self.logger.info(f"Chunk {chunk_id} processed successfully. Result: {result}")
            #     except Exception as e:
            #         self.logger.error(f"Chunk {chunk_id} failed with error: {e}")

    def execute_all_chunks(self, chunks: List[Dict[str, Any]]):
        for i, chunk in enumerate(chunks):
            # 对每个切片进行处理
            if not isinstance(chunk, dict):
                self.logger.warning(f"第 {i} 个 chunk 不是字典类型，已抛弃。")
                continue
            # 从chunk中获取数据
            chunk_id = chunk.get("chunk_id")
            item_name = chunk.get("item_name")
            content = chunk.get("content")

            # 构建提示词，调用LLM，对结果清洗，双写
            self.process_single_chunk(chunk_id, item_name, content)






    def process_single_chunk(self, chunk_id: str, item_name: str, content: str):
        message = [{"role": "system", "content": KNOWLEDGE_GRAPH_SYSTEM_PROMPT}, {"role": "user", "content": f"切片内容\n\n{content}"}]
        #调用llm
        client = self.llm_client
        response = client.invoke(message)
        result = getattr(response, "content", "").strip()
        graph_result  = self.parse_and_clean(result)

        # 得到最终数据
        entities = graph_result.get("entities")
        relations = graph_result.get("relations")

        # 无有效实体则跳过，不中断整个流程
        if not entities:
            self.logger.warning(f"Chunk {chunk_id} 未提取到有效实体，跳过图谱写入")
            return

        #双写操作 写入milvus和neo4j
        self._milvus_operator.insert(entities, chunk_id, item_name, content, client=self.client)
        self._Neo4j_obj.insert(self.neo4j_driver, entities, relations, chunk_id, item_name)

    def parse_and_clean(self, result: str) -> Dict[str, Any]:
        # 解析llm的返回数据
        # 去除特殊符号 ``` 代码块 ```
        # 反序列化 json --> Dict
        # 分别获取实体和关系数据

        # sub+正则表达式
        result = re.sub(r"^```(?:json)?\s```", "", result.strip(), flags=re.DOTALL)
        result = re.sub(r"\s*```$","", result, flags=re.DOTALL)

        # 转换json为dict
        parse_result:Dict[str, Any] = json.loads(result)

        # 得到实体和关系数据
        entities = parse_result.get("entities",[])
        relations = parse_result.get("relations",[])

        # 清洗实体
        cleaned_entities = self.clean_entities(entities)

        # 集合推导式，最终得到的是集合
        # 集合的特点是 去重无序
        cleaned_unique_entity_name = {entity.get('name') for entity in cleaned_entities}

        # 根据清洗后的实体名称 来 清洗关系
        cleaned_relations = self.clean_relations(relations, cleaned_unique_entity_name)

        # 构建最终结果
        final_data = {
            "entities": cleaned_entities,
            "relations": cleaned_relations
        }
        return final_data

    def clean_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 清洗无用的实体 例如空实体
        final_result = []
        # 创建set集合
        unique_set = set()
        for entity in entities:
            entity_name = str(entity.get("name", "")).strip()
            # 空则不处理
            if not entity_name:
                continue

            # 对过长实体名称进行截取
            if len(entity_name) > MAX_ENTITY_LENGTH:
                entity_name = entity_name[:MAX_ENTITY_LENGTH]

            # 去除不在白名单范围内的标签
            entity_label = entity.get("label")
            if entity_label not in ALLOWED_ENTITY_LABLES:
                continue
            # 去重
            # 规则：实体的名字和标签完全一致，则去重
            unique_key = (entity_name, entity_label)
            if unique_key in unique_set:
                continue
            unique_set.add(unique_key)
            # 最终返回entity数据
            clean_entity= {
                "name": entity_name,
                "label": entity_label,
            }
            entity_description = entity.get("description", "")
            if entity_description:
                clean_entity["description"] = entity_description
            final_result.append(clean_entity)
        # 返回最终数据
        return final_result





    def clean_relations(self, relations: List[Dict[str, Any]], cleaned_unique_entity_name: set[str]) -> List[Dict[str, Any]]:
        # 判断head和tail是否都存在
        # {"head":aaa , "tail":bbb, "type":xxx}
        final_relations = []
        for relation in relations:
            head = str(relation.get("head")).strip()
            tail = str(relation.get("tail")).strip()
            if not head or not tail:
                continue

            # 对超过固定长度的tail和head进行截取
            if len(head) > MAX_ENTITY_LENGTH:
                head = head[:MAX_ENTITY_LENGTH]
            if len(tail) > MAX_ENTITY_LENGTH:
                tail = tail[:MAX_ENTITY_LENGTH]

            # head和tail是否在清晰后的实体名称集合中
            if head not in cleaned_unique_entity_name or tail not in cleaned_unique_entity_name:
                continue

            # head和tail间的关系必须在关系的白名单中
            relation_type = str(relation.get("type")).strip()
            if relation_type not in ALLOWED_RELATION_TYPES:
                # 添加默认关系
                relation_type = DEFAULT_RELATION_TYPE

            # 构建最终返回数据
            clean_relation = {
                "head": head,
                "tail": tail,
                "type": relation_type,
            }
            final_relations.append(clean_relation)
        return final_relations

    def validate_param(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], str]:
        self.log_step("step1", "知识图谱构建参数校验")

        # 1. 获取基础字段
        chunks = state.get("chunks") or []
        global_item_name = str(state.get("item_name", "")).strip()

        # 2. 校验整体 chunks 是否存在
        if not chunks:
            raise ValueError("待提取图谱的切块(chunks)不存在，跳过图谱构建。")

        # 3. 逐个校验 Chunk 的有效性
        validated_chunks = []
        for i, chunk in enumerate(chunks):

            # 3.1 chunk 是否是字典
            if not isinstance(chunk, dict):
                self.logger.warning(f"第 {i} 个 chunk 不是字典类型，已抛弃。")
                continue

            # 3.2 处理 chunk_id
            raw_id = chunk.get("chunk_id")
            chunk_id = str(raw_id).strip() if raw_id is not None else f"kg_chunk_temp_{i}"

            # 3.3 获取 content 内容
            content = str(chunk.get("content", "")).strip()
            if not content:
                self.logger.warning(f"Chunk {chunk_id} 缺少 content，已抛弃。")
                continue

            # 3.4 获取 item_name（chunk 级别优先，全局兜底）
            chunk_item = str(chunk.get("item_name", "")).strip() or global_item_name
            if not chunk_item:
                self.logger.warning(f"Chunk {chunk_id} 缺少 item_name 归属，已抛弃。")
                continue

            # 3.5 更新 chunk 字段
            chunk["chunk_id"] = chunk_id
            chunk["item_name"] = chunk_item
            chunk["content"] = content

            # 3.6 加入有效列表
            validated_chunks.append(chunk)

        # 4. 校验清洗后是否还有有效数据
        if not validated_chunks:
            raise ValueError(f"经过清洗后，没有任何有效的 chunk（{len(validated_chunks)}）可用于构建图谱。")

        return validated_chunks, global_item_name
