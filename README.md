# 掌柜智库 Shopkeeper Brain

掌柜智库是一套面向设备说明书和产品资料的知识库问答系统，覆盖文档导入、Markdown 解析、图片处理、文档切分、商品名识别、向量检索、HyDE 检索、网络检索、RRF 融合、重排序和答案生成。

## 功能

- 上传 PDF/Markdown，并保存原始文件与解析产物
- 使用 MinerU 将 PDF 转换为 Markdown
- 使用视觉模型生成图片标题，并将图片保存到 MinIO
- 按标题和长度切分文档
- 使用 LLM 识别商品名
- 使用 BGE-M3 生成稠密和稀疏向量
- 支持 Milvus 混合检索、HyDE 检索和 MCP 网络搜索
- 使用 RRF 融合多路召回结果
- 使用 bge-reranker-large 精排
- 使用 LLM 生成最终答案和参考图片链接
- 使用 MongoDB 保存聊天历史
- 使用 Neo4j 保存实体关系图谱

## 架构

```text
PDF / Markdown
      │
      ▼
文件上传服务 :8000
      │
      ├── MinerU PDF → Markdown
      ├── VLM 图片标题 → MinIO
      ├── 文档切分
      ├── LLM 商品名识别
      ├── BGE-M3 向量化
      ├── Milvus 向量入库
      └── Milvus + Neo4j 知识图谱入库

用户问题
      │
      ▼
查询对话服务 :8001
      │
      ├── 商品名确认
      ├── 普通向量检索
      ├── HyDE 检索
      ├── MCP 网络检索
      ├── RRF 融合
      ├── Reranker 精排
      └── LLM 答案生成 / SSE 流式输出
```

## 目录结构

```text
.
├── shopkeer_brain/
│   ├── knowledge/
│   │   ├── front/                 # 聊天查询 API 与 chat.html
│   │   ├── upload/                # 文件导入 API 与 import.html
│   │   ├── processor/
│   │   │   ├── import_process/    # 导入 LangGraph 流水线
│   │   │   └── query_process/     # 查询 LangGraph 流水线
│   │   ├── prompt/                # LLM 提示词
│   │   ├── schema/                # 请求与响应模型
│   │   └── utils/                 # Milvus、Neo4j、MongoDB、MinIO、模型工具
│   ├── README.md
│   └── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 环境要求

- Python 3.10+
- Conda
- Milvus 2.6+
- Neo4j 5+
- MongoDB 7+
- MinIO
- MinerU 4+
- NVIDIA GPU（推荐，用于 BGE-M3 和 Reranker）
- DashScope/OpenAI 兼容 API

## 安装

```bash
conda create -n knowledge python=3.10 -y
conda activate knowledge
pip install -r requirements.txt
```

如果使用 PyCharm，请把项目解释器设置为 `knowledge` 环境。

## 配置

复制配置模板：

```bash
cp .env.example shopkeer_brain/knowledge/.env
```

Windows PowerShell：

```powershell
Copy-Item .env.example shopkeer_brain\knowledge\.env
```

然后修改以下内容：

- LLM API Key、Base URL 和模型名称
- Milvus 地址与集合名称
- Neo4j 地址、账号和密码
- MongoDB 地址
- MinIO 地址、账号、密码和 Bucket
- BGE-M3 与 Reranker 本地模型路径
- MinerU 可执行文件和模型目录

`.env` 已加入 `.gitignore`，不要提交真实密钥。

## 启动服务

项目由两个独立服务组成，请分别打开两个终端。

### 1. 文件导入服务

```powershell
cd shopkeer_brain\knowledge\upload\api
python import_router.py
```

访问：

```text
http://127.0.0.1:8000/import
```

### 2. 查询对话服务

```powershell
cd shopkeer_brain\knowledge\front\api
python query_router.py
```

访问：

```text
http://127.0.0.1:8001/chat.html
```

使用顺序：先通过 8000 导入文档，完成入库后，再通过 8001 提问。

## 数据存储

| 组件 | 用途 |
| --- | --- |
| Milvus `kb_chunks_v2` | chunk 稠密/稀疏向量与正文 |
| Milvus `kb_item_names_v2` | 商品名稠密/稀疏向量 |
| Milvus `kb_graph_entity_names_v2` | 图谱实体向量 |
| Neo4j | 实体、Chunk 与实体关系 |
| MongoDB `kb001` | 聊天历史 |
| MinIO `knowledge-base-v2` | 原始文件、Markdown 图片 |

重复导入同一文档时，chunk 和商品名按 `file_title` 先删后写；MinIO 原文件按 SHA-256 内容哈希去重；知识图谱按 `item_name` 清理后重建。

## 日志

项目默认使用 `logging.basicConfig()` 输出到终端。直接在 PyCharm 或 PowerShell 启动时，日志显示在运行终端中。

如果使用重定向启动，日志示例：

```powershell
python query_router.py *> query.log
```

实时查看：

```powershell
Get-Content .\query.log -Tail 100 -Wait
```

## 常见问题

### 端口被占用

```powershell
netstat -ano | findstr :8000
netstat -ano | findstr :8001
```

### MinerU advanced 档无法使用

先启动本地解析服务：

```powershell
mineru server start
mineru server status
```

也可以设置：

```text
MINERU_TIER=flash
```

### 图片无法显示

确认浏览器可以访问 `MINIO_ENDPOINT`，并确认 Bucket 中的图片允许公开读取。

### CUDA 内存不足

- 确认 `BGE_DEVICE=cuda:0`
- 确认 `BGE_FP16=True`
- 关闭其他占用 GPU 的 Python 进程
- 必要时改用 CPU 或更小模型


