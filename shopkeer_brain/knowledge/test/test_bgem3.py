from pymilvus.model.hybrid import BGEM3EmbeddingFunction


# 验证嵌入模型是否可用
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=r"E:/bge-m3",
    device="cuda", # cuda：gpu模式 / cpu: 使用cpu计算
    use_fp16=True,  # 加速，只有gpu支持加速
    return_colbert_vecs=False  # 彻底关闭Colbert，减少计算量
)

test_texts = ["你好", "Milvus + BGE-M3 测试"]
result = bge_m3_ef.encode_documents(test_texts)

# 打印稠密向量信息
print("稠密向量维度:", len(result["dense"][0]))
print("第一条稠密向量:\n", result["dense"][0])

# 打印稀疏向量信息
print("\n第一条稀疏向量:\n", result["sparse"][0])