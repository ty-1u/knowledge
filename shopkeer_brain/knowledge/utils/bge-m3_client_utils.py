import os

from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
load_dotenv()
# 获取嵌入模型客户端对象
def get_bgem3_client():
    try:

        bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=os.getenv("BGE_M3_PATH"),
            device=os.getenv("BGE_DEVICE", "cuda:0"),  # cuda：gpu模式 / cpu: 使用cpu计算
            use_fp16=os.getenv("BGE_FP16", "True") == "True",  # 加速，只有gpu支持加速
            return_colbert_vecs=False  # 彻底关闭Colbert，减少计算量
        )
        return bge_m3_ef

    except  Exception as e:
        raise e
