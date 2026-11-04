import os
import re
from typing import Tuple, List, Dict, Any

from shopkeer_brain.knowledge.processor.import_process.base import BaseNode
from shopkeer_brain.knowledge.processor.import_process.config import get_config
from shopkeer_brain.knowledge.processor.import_process.state import ImportGraphState

from langchain_text_splitters import RecursiveCharacterTextSplitter



class DocumentSplitNode(BaseNode):
    def process(self,state: ImportGraphState) -> ImportGraphState:
        # 加载文档 --> 切分 --> 组合（大则继续切分，小则合并） --> 保存

        # 基于/md文档，根据标题切分
        # 1. 将格式标准化 \n \r\n \r
        (md_content, file_title) = self.get_input_param(state)

        # 2.根据标题作切分
        parts = self.split_doc_title(md_content, file_title)

        # 3.对根据标题切分的内容再进行处理
        # 每段内容过大，切分；过小，合并
        # 若段内容过大，阈值为1000字符，使用递归字符文本切分器
        # ["\n\n","\n","。","!","？"，","," "]
        # 若段内容太小，多个段进行合并，阈值为最小500；合并规则为同源合并，必须是同一个parent_title内容进行合并
        # 如果合并后总大小大于2000，不再处理，避免循环切割合并
        config = get_config()
        final_chunks = self.split_and_merge(parts, config.min_content_length, config.max_content_length)

        # 4. 对上面对分和合并返回列表，根据业务需求重新组装，组装成满足具体业务结构
        chunks = self.collect_data(final_chunks)
        #更新state状态
        state["chunks"] = chunks

        # 5. 其他：日志，备份json数据
        self.output_log(md_content, chunks, config.max_content_length,config.min_content_length)
        # 备份
        self.copy_chunks(chunks,state)


        return state

    # 数据的重构
    def collect_data(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for index,chunk in enumerate(final_chunks):
            title = chunk["title"]
            body = chunk["body"]
            file_title = chunk["file_title"]
            parent_title = chunk["parent_title"]
            part = chunk.get("part")

            # 在最终内容：content = title + body
            content = title + "\n\n" + body

            data = {
                "title": title,
                "content": content  ,
                "file_title": file_title,
                "parent_title": parent_title,
            }
            if part :
                data["part"] = part
            chunks.append(data)
        return chunks

    # json数据里面也有list，chunk类型为List[Dict[str, Any]，其中dict和list都可以很容易的转换为json数据
    def copy_chunks(self, chunks: List[Dict[str, Any]], state: ImportGraphState):
        import json
        import os
        from datetime import datetime

        file_dir = state["file_dir"]
        os.makedirs(file_dir, exist_ok=True)

        file_path = os.path.join(file_dir, f"chunks_{datetime.now():%Y-%m-%d_%H-%M-%S}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4, default=str)
        except Exception as e:
            self.logger.error("备份失败：%s", e)
            raise

        self.logger.info("备份文件：%s", file_path)

    #日志输出
    def output_log(self, md_content: str, chunks: List[Dict[str, Any]], max_length: int, min_length: int):
        line_count = md_content.count("\n")+1
        self.logger.info(f"文档总行数：{line_count}")
        self.logger.info(f"文档总长度：{len(md_content)}")
        self.logger.info(f"文档切分后总段数：{len(chunks)}")
        self.logger.info(f"文档切分后最大段长度：{max_length}")
        self.logger.info(f"文档切分后最小段长度：{min_length}")
    def split_and_merge(self, parts: List[Dict[str, Any]], min_length: int, max_length: int) :

        # 每段内容过大，切分；过小，合并
        # 若段内容过大，阈值为1000字符，使用递归字符文本切分器
        # ["\n\n","\n","。","!","？"，","," "]
        # 若段内容太小，多个段进行合并，阈值为最小500；合并规则为同源合并，必须是同一个parent_title内容进行合并
        # 如果合并后总大小大于2000，不再处理，避免循环切割合并
        current_parts = []
        for part in parts:
            # 过大，则再拆分
            chunks_split = self.split_long_part(part, max_length)
            current_parts.extend(chunks_split)

        # 上一步拆分的每段内容进行合并
        final_parts = self.merge_short_part(current_parts,min_length)
        return final_parts
    # 内容过大则切分
    def split_long_part(self, part: Dict[str, Any], max_length: int) -> List[Dict[str, Any]]:
        # 获取出字典的值
        title = part["title"]
        body = part["body"]
        file_title =  part["file_title"]
        parent_title = part["parent_title"]

        # 判断当前段内容是否超过max_length
        if len(title) > 50 :
            title = title[:50]


        title_prefix = f"{title}\n\n"
        total_length = len(title_prefix) + len(body)

        if total_length > max_length:
            # 获取切分后每段的大小
            body_size = max_length - len(title_prefix)

            # 递归字符文本切分器
            # 默认对最小长度没有要求
            text_spliter = RecursiveCharacterTextSplitter(
                chunk_size= body_size,
                chunk_overlap=0,
                separators=["\n\n", "\n", "。", "!", "?", ", ", " "],
                keep_separator=False,
            )
            # 返回结果为列表
            chunks = text_spliter.split_text(body)
            # 把切分后的列表进行处理
            final_parts = []
            for index, text in enumerate(chunks):
                final_parts.append({
                    "title": title+"-"+f"{index+1}",
                    "body": text,
                    "file_title": file_title,
                    "parent_title": parent_title,
                    "part": f"{index+1}"
                })

            return final_parts

        else :
            return [part]


    def merge_short_part(self, current_parts: List[Dict[str, Any]], min_length: int) -> List[Dict[str, Any]]:
        # 过小，则进行合并
        # 同源合并，parent_title必须一样才合并

        #获取列表的第一段内容
        current_part = current_parts[0]

        final_parts = []
        for part in current_parts[1:]:
            if part["parent_title"] == current_part["parent_title"] and len(current_part["body"]) < min_length:
                current_part["body"] = current_part["body"].rstrip() + "\n\n" + part["title"] + "\n" + part["body"].rstrip()
                current_part["title"] = current_part["parent_title"]
                # 先对所有编号置0，然后重新分配编号
                current_part["part"] = 0
            else:
                # 遇到不满足条件的，把前面内容加入最终的列表
                final_parts.append(current_part)
                current_part = part
        final_parts.append(current_part)

        #专门处理每一段中的part字段
        part_counter = {}
        result = []
        for parts in final_parts:
            if "part" in parts:
                parent_title = parts["parent_title"]
                part_counter[parent_title] = part_counter.get(parent_title,0) + 1

                new_part = part_counter[parent_title]
                parts["part"] = new_part
            result.append(parts)
        return result


    def get_input_param(self, state: ImportGraphState) -> Tuple[str, str]:
        # 得到md内容
        md_content = state["md_content"]
        # 获取文件title
        file_title = state["file_title"]
        # 字符串替换方法
        if md_content:
            md_content = md_content.replace("\r\n","\n").replace("\r","\n")
        return md_content, file_title

    # 返回数据示例
# [{body: aaa , title: xxx}, {...}]
    def split_doc_title(self,md_content:str , file_title: str) -> List[dict] :
        # 正则表达式找标题
        title_pattern = re.compile(r"^\s*(#{1,6}\s+)")
        # 逐行匹配
        # 控制位
        in_code = False
        # 临时存储
        temp = []
        # 最终数据
        res = []
        # 记录标题
        title = ""
        # ["","# ","## ","### "...]
        level_title_list = [""] * 7
        current_level = 0 # 无标题

        lines = md_content.split("\n")
        for line in lines:
            # 判断是否是代码块的 # ，不作为标题处理
            if line.strip() in ("```","~~~"):
                in_code = not in_code
            # 不是代码块，且为标题
            if not in_code and title_pattern.match(line):
                # 将[c1,c2,c3] -> "c1\nc2\nc3"
                content = "\n".join(temp)
                # 不为空，则进行添加
                if title or content:
                    # 找到当前标题的上级标题
                    parent_title = ""
                    for i in range(current_level-1,0,-1):
                        if level_title_list[i] != "":
                            parent_title = level_title_list[i]
                            break
                    # 如果没有父标题，添加默认值
                    if not parent_title:
                        parent_title = title if title else file_title

                    res.append(
                        {
                            "title":title,"body":content,"file_title":file_title,"parent_title":parent_title,
                         }
                    )
                # 把标题放到列表中
                # 获取标题数据
                match_obj =title_pattern.match(line)
                if match_obj:
                    # 获取标题#的数量
                    level = match_obj.group(1).count("#")
                    current_level = level
                    level_title_list[level] = line

                    # 当前列表存储标题索引，清空
                    for i in range(current_level+1,7):
                        level_title_list[i] = ""

                # 记录标题
                title = line
                temp = []
            else :
                temp.append(line)

            # 单独处理最后一行
        last_content = "\n".join(temp)
        if title or last_content:
            parent_title = ""
            for i in range(current_level-1,0,-1):
                if level_title_list[i] != "":
                    parent_title = level_title_list[i]
                    break
            if not parent_title:
                parent_title = title if title else file_title

            res.append(
                {
                    "title": title, "body": last_content, "file_title": file_title,"parent_title":parent_title
                }
            )
        return  res
