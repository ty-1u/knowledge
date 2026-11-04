import asyncio
import json
from typing import Any, Dict, List

from agents.mcp import MCPServerStreamableHttp

from shopkeer_brain.knowledge.processor.query_process.base import BaseNode
from shopkeer_brain.knowledge.processor.query_process.state import QueryGraphState

# 基于ali百炼的联网搜索mcp
class WebSearchNode(BaseNode):
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1.参数校验
        item_names,rewritten_query = self.validate_param(state)
        # 2.调用mcp工具得到返回结果
        mcp_result = asyncio.run(self.execute_web_search_mcp(rewritten_query))
        if not mcp_result:
            return {}
        return {"web_search_docs": mcp_result}

    def validate_param(self, state: QueryGraphState):
        item_names = state.get("item_names")
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise ValueError("Rewritten query is missing")
        if not item_names:
            raise ValueError("Item names are missing")
        return item_names, rewritten_query
    async def execute_web_search_mcp(self,rewritten_query:str) -> List[Dict[str,Any]]:
        # 建立和mcp的连接
        mcp_client = None

        api_key = self.config.openai_api_key
        # 建立连接时的header信息
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # 创建连接
        try:
            mcp_client = MCPServerStreamableHttp(
                name="通用搜索",
                params={
                    # mcp服务器地址
                    "url": self.config.mcp_dashscope_base_url,
                    "headers": headers,
                },
                # 添加缓存
                cache_tools_list=True,
                max_retry_attempts=3,
            )
            # 建立连接
            await mcp_client.connect()
            # 调用工具
            # 如果后续内容用到了前一步的值，就需要使用await
            result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": rewritten_query, "count": 3}
            )
            if not result:
                return []
            # result默认返回的是对象，直接可以用.取到属性
            if not getattr(result, "content", None):
                return []
            raw_text = result.content[0].text
            # 反序列化 str --> dict
            try:
                content_text = json.loads(raw_text)
            except (json.JSONDecodeError, TypeError):
                return []
            if not isinstance(content_text, dict):
                return []
            pages = content_text.get('pages') or []
            # 从pages中获取具体数据 snippet 问题答案 title标题问题 url 网页地址

            # 定义列表分装最终数据，列表中为Dict类型
            final_data = []
            for page in pages:

                # 构建数据
                data = {
                    "snippet": page.get("snippet",""),
                    "title": page.get("title",""),
                    "url": page.get("url",""),
                }
                final_data.append(data)

            return final_data

        except Exception as e:
            raise Exception(f"Failed to connect to MCP: {e}")
        finally:
            if mcp_client:
                await mcp_client.cleanup()
