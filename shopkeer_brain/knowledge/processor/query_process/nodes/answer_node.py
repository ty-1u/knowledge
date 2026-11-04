from typing import Dict, List

from shopkeer_brain.knowledge.processor.query_process.base import BaseNode
from shopkeer_brain.knowledge.processor.query_process.state import QueryGraphState
from shopkeer_brain.knowledge.prompt.query.query_prompt import ANSWER_PROMPT
from shopkeer_brain.knowledge.front.utils.task_util import set_task_result
from shopkeer_brain.knowledge.utils.llm_client_util import get_llm_client
from shopkeer_brain.knowledge.utils.mongo_history_util import save_chat_message
from shopkeer_brain.knowledge.utils.sse_util import SSEEvent, push_sse_event


class AnswerOutputNode(BaseNode):
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 判断answer是否存在，如果存在则直接返回
        if state.get("answer"):
            # 给前端使用
            set_task_result(state["task_id"], "answer", state["answer"])
            if state.get("is_stream"):
                push_sse_event(
                    state["task_id"],
                    SSEEvent.FINAL,
                    {"answer": state["answer"]},
                )
            return state
        # 2. 没有answer，获取相关数据（reranker , 历史会话, item_name等）
        # 3. 通过上面数据构建提示词
        prompt = self.build_prompt(state)
        state['prompt'] = prompt
        # 4. 提示词提交LLM，生成最终答案
        # 流式输出
        self.call_llm_streaming(state,prompt)

        # 5. 写入历史记录
        self.write_history(state)
        # 6. SSE 流式处理
        is_stream = state.get("is_stream")
        task_id = state.get("task_id")
        if is_stream:
            # sse发送结束事件
            push_sse_event(task_id,SSEEvent.FINAL,{"answer":state.get("answer")})

        return state
    def build_prompt(self , state):
        # 得到提示词中需要拼接的字符
        question = state.get("rewritten_query","") or state.get("original_query","")
        item_names = state["item_names"]

        # 获取答案列表
        reranked_docs= state.get("reranked_docs")
        reranked_str = self.format_rerank_docs(reranked_docs)
        # 历史会话的记录
        history = state.get("history")
        history_str = self.format_history(history)
        # 图谱关系描述（当前查询流程暂未接入图谱检索，先占位）
        graph_relation_description = state.get("kg_triples") or "None"

        # 构建提示词
        return ANSWER_PROMPT.format(
            context = reranked_str or "None",
            history = history_str if history_str else "None",
            question = question,
            item_names = item_names,
            graph_relation_description = graph_relation_description
        )

    # 整理rerank数据 List[Dict[str, Any]] -> Str
    def format_rerank_docs(self, reranked_docs:List[Dict])->str:
        # 定义列表
        result = []
        used = 0

        # 遍历
        for i,doc in enumerate(reranked_docs or []):
            content = doc.get("content")
            if not content:
                continue
            # 拼接字典
            meta = {
                "score": doc.get("score", 0),
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
            }
            data = f"[{i+1}]:{meta}\n{content}"
            if used + len(data) > 2500:
                break

            result.append(data)

            used += len(data)

        # 列表饭返回字符串
        return "\n".join(result)

    def format_history(self, history:List[Dict])->str:
        # 定义列表
        result = []
        used = 0

        if isinstance(history, str):
            return history[:2500]

        for msg in history or []:
            if not isinstance(msg, dict):
                continue
            text = msg.get("text","")
            role = msg.get("role","")
            # user: abc assit: def
            data = f"{role}:{text}"

            if used + len(data) > 2500:
                break
            result.append(data)
            used += len(data)

        return "\n".join(result)
    def call_llm_streaming(self, state: QueryGraphState, prompt: str):
        # 得到llm客户端
        client = get_llm_client()
        task_id = state.get("task_id")
        # 判断是流式还是非流式
        if state["is_stream"] :
            state["answer"] = self.stream_output(client,prompt,task_id)
        else:
            state["answer"] = self.invoke_output(client,prompt)
            # 存入缓存中再返回
            set_task_result(state["task_id"], "answer", state["answer"])

    def stream_output(self, client, prompt, task_id):
        final_answer = ""
        for chunk in client.stream(prompt):
            delta_text = getattr(chunk, "content", "")
            if delta_text:
                final_answer+= delta_text
                # 每生成一段，直接sse推送，推送的标志是delta
                push_sse_event(task_id, SSEEvent.DELTA, {"delta": delta_text})

        return final_answer


    def invoke_output(self, client, prompt):
        response = client.invoke(prompt)
        return response.content
    def write_history(self, state: QueryGraphState):
        session_id = state["session_id"]
        rewritten_query = (state.get("rewritten_query", "")
                           or state.get("original_query", ""))
        item_names = state.get("item_names") or []

        try:  # 1. 写用户问题
            save_chat_message(
                session_id=session_id,
                role="user",
                text=state["original_query"],
                rewritten_query=rewritten_query,
                item_names=item_names,
            )

            # 2. AI回复
            if state.get("answer"):
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=state["answer"],  # 模型的输出
                    rewritten_query=rewritten_query,
                    item_names=item_names,
                )
        except Exception as e:
            self.logger.warning(f"写入历史记录失败: {e}")
