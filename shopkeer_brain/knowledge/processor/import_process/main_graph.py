# 通过langGraph机制来执行多个节点
import json

from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from shopkeer_brain.knowledge.processor.import_process.nodes.chunks_embedding_node import ChunksEmbeddingNode
from shopkeer_brain.knowledge.processor.import_process.nodes.document_split import DocumentSplitNode
from shopkeer_brain.knowledge.processor.import_process.nodes.entry_node import EntryNode
from shopkeer_brain.knowledge.processor.import_process.nodes.import_milvus_node import ImportMilvusNode
from shopkeer_brain.knowledge.processor.import_process.nodes.item_name_recognition import ItemNameRecognition
from shopkeer_brain.knowledge.processor.import_process.nodes.knowledge_graph_node import KnowledgeGraphNode
from shopkeer_brain.knowledge.processor.import_process.nodes.md_img import MdImageNode
from shopkeer_brain.knowledge.processor.import_process.nodes.pdf_to_md import  PdfToMd
from shopkeer_brain.knowledge.processor.import_process.state import ImportGraphState, create_default_state


# 两个节点
#entry_node: 文件类型检查
#pdf_to_md：pdf转换为md
def import_router(state: ImportGraphState) -> str:
    if state["is_pdf_read_enabled"]:
        return "pdf_to_md"
    elif state["is_md_read_enabled"]:
        return "md_img"
    else:
        return END

def create_graph_import() -> CompiledStateGraph:
    # 创建builder,builder是stateGraph的类型
    builder = StateGraph(ImportGraphState)

    #添加节点
    #设置入口节点
    builder.set_entry_point("entry_node")

    # 添加节点
    # 节点添加的参数可以是个返回字典状态的方法，也可以是自定的的对象，对象必须调用自己的方法并且返回字典状态

    ## python中约定，把对象当函数使用时会触发__call__方法
    # 这里的括号中的参数自动传入__call__方法，不需要手动传state，state会自动传入
    nodes = {
        "entry_node":EntryNode(),
        "pdf_to_md":PdfToMd(),
        "md_img":MdImageNode(),
        "document_split":DocumentSplitNode(),
        "item_name_recognition": ItemNameRecognition(),
        "chunks_embedding_node":ChunksEmbeddingNode(),
        "import_milvus_node": ImportMilvusNode(),
        "knowledge_graph_node": KnowledgeGraphNode(),
    }

    # 遍历
    for key,value in nodes.items():
        # 添加节点
        builder.add_node(key,value)

    # 添加边
    builder.add_edge("pdf_to_md","md_img")
    builder.add_edge("md_img","document_split")
    builder.add_edge("document_split","item_name_recognition")
    builder.add_edge("item_name_recognition","chunks_embedding_node")
    builder.add_edge("chunks_embedding_node","import_milvus_node")
    builder.add_edge("import_milvus_node","knowledge_graph_node")
    builder.add_edge("knowledge_graph_node",END)
    # 添加条件边
    # 根据entry_node节点判断是否是md类型
    # (开始节点位置，判断方法，进入哪个节点)
    builder.add_conditional_edges(
        "entry_node",
        import_router,
        {
            "pdf_to_md": "pdf_to_md",
            "md_img": "md_img",
            END: END,
        }
    )


    # 图编译
    return builder.compile()

# 测试
# 构建状态数据，流式输出
def run_graph_import(import_file_path:str,file_dir:str ):
    # 获取编译后的graph对象
    graph = create_graph_import()
    #构建状态数据
    state = {
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }
    # 解包
    init_state = create_default_state(**state)

    #图执行
    final_state =None
    for event in graph.stream(init_state):
        for node_name,state in event.items():
            print(f"运行节点的：{node_name},state{state}")
            # 由于循环，final_state最后得到的是最后一个节点的状态信息
            final_state = state
    return final_state
if __name__ == "__main__" :
    import_file_path= r"E:\掌柜智库\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\hak180产品安全手册.pdf"
    # 输出目录（可选），默认值为文件所在目录
    file_dir= r"E:\掌柜智库\shopkeer_brain\knowledge\processor\import_process\import_temp_dir"
    final_state = run_graph_import(import_file_path=import_file_path, file_dir=file_dir)
    print(json.dumps(final_state, indent=2,ensure_ascii=False))

