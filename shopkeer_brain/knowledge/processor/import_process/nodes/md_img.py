import json
import os
import re
import base64
from pathlib import Path
from typing import Tuple, List
from urllib.parse import quote

from openai import OpenAI

from shopkeer_brain.knowledge.processor.import_process.base import BaseNode, T, setup_logging
from shopkeer_brain.knowledge.processor.import_process.config import get_config
from shopkeer_brain.knowledge.processor.import_process.exceptions import ValidationError
from shopkeer_brain.knowledge.processor.import_process.state import ImportGraphState
from shopkeer_brain.knowledge.utils.minio_utils import get_minio_client


class MdImageNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.获取md
        # 获取文档内容

        # 获取图片 元组方式返回（python返回多返回值本质是用了元组进行返回）
        md_content,md_path,image_path = self.get_md_content(state)
        # 1.1 判断md中是否存在图片
        if not image_path.exists():
            state["md_content"] = md_content
            return state

        # 2. 根据图片路径，md内容，获取md的上下文
        target_images_context = self.get_images_context(image_path,md_content)

        # 3. 调用VLM生成图片摘要
        image_summaries = self.generate_image_summaries(md_path,target_images_context)
        # 4. 将图片上传到文件存储服务器,更新md中的图片路径和摘要信息
        new_md_content = self.upload_img_update_md(md_path,md_content,image_summaries,target_images_context)


        # 5.更新state数据
        state['md_content'] = new_md_content
        return  state


    #获取图片
    def get_md_content(self,state: ImportGraphState) -> Tuple[str,Path,Path]:
        md_path = state['md_path']
        md_path_obj = Path(md_path)

        # 判断md是否存在
        if not md_path_obj.exists():
            raise FileNotFoundError(f"md文件不存在:{md_path}")
        # 获取md文件的内容
        # 自动打开关闭
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # 获取图片路径,这个images_path已经是Path变量
        image_path = md_path_obj.parent / 'images'
        return md_content, md_path_obj, image_path

    #根据md内容和图片路径，获取上下文内容
    # 返回格式 list[("fig_name","fig_path",("图片前一个标题","上文","下文"))]
    def get_images_context(self, image_path: Path, md_content: str) -> List[Tuple[str, str, Tuple[str, str, str]]]:
        target_images_context = []
        # 1. 根据图片路径获取所有图片
        for img_name in os.listdir(image_path):
            # 2. 图片格式是否正确
            file_ext = os.path.splitext(img_name)[1].lower()
            config = get_config()
            if file_ext not in config.image_extensions:
                continue
            # 3.符合格式要求的图片生成上下文内容
            img_context = self.build_image_context(md_content, img_name)

            # 把图片的上下文内容加入列表
            if not img_context:
                continue
            single_img_context =  img_context[0]
            target_images_context.append((img_name,image_path,single_img_context))
        return target_images_context
    # 2.1
    def build_image_context(self, md_content: str, img_name: str) -> List[ Tuple[str, str, str]]:
        # 根据图片文件名称到md内容找图片位置
        re_pattern = re.compile(r"!\[.*?\]\(.*?"+re.escape(img_name)+r".*?\)")
        images_context = []
        ## 把md内容差分，一行一行匹配
        for line_index,line in enumerate(md_content.split('\n')):
            if not re_pattern.search(line):
                continue
            # 标题
            front_title = ""
            front_index = -1

            # 获取图片上文
            for i in range(line_index-1, -1, -1):
                if re.match(r"^#{1,6}\s+", md_content.split('\n')[i]):
                    # 获取标题
                    front_title = md_content.split('\n')[i]
                    #获取索引
                    front_index = i
                    break
            # 获取标题和图片之间的位置
            pre_content = md_content.split('\n')[front_index+1 : line_index]

            # 根据上文内容，最多得到200字符，超过则被截断
            final_img_pre_context = self.image_context_limit(pre_content,"front")
            # 得到下文内容
            back_index = len(md_content.split('\n'))
            for i in range(line_index+1,back_index):
                # 匹配标题
                if re.match(r"^#{1,6}\s+", md_content.split('\n')[i]):
                    back_index = i
                    break
            back_context = md_content.split('\n')[line_index+1 : back_index]
            # 向下截取，最多得到200字符，超过则被截断
            final_img_back_context = self.image_context_limit(back_context,"back")

            images_context.append((front_title,final_img_pre_context,final_img_back_context))
        return images_context

    #2.1.1
    def image_context_limit(self, substr_content: list, direction: str) -> str:
        if direction not in ("front", "back"):
            raise ValueError(f"direction 必须是 front 或 back，收到 {direction!r}")

        current_content: List[str] = []
        final_content: List[str] = []

        for line in substr_content:
            clean_line = line.strip()

            # 空行：当作段落分隔，当前段落收尾
            if not clean_line:
                if current_content:
                    final_content.append('\n'.join(current_content))
                    current_content = []
            # 非空行：判断是不是图片行
            else:
                # 图片行：同样当作分隔，且不收集进 current_content
                if re.search(r"!\[[^\]]*\]\([^)]+\)", clean_line):
                    if current_content:
                        final_content.append('\n'.join(current_content))
                        current_content = []
                # 普通文本行：收集
                else:
                    current_content.append(line)

        # 收尾剩余内容
        if current_content:
            final_content.append('\n'.join(current_content))

        # 按【字符数】截取 200，而不是按段数
        full_text = '\n\n'.join(final_content)
        limit = 200

        if direction == "front":
            return full_text[-limit:]
        else:
            return full_text[:limit]


    # 根据图片和上下文内容，调用视觉模型VLM
    # 返回格式 list[("fig_name", "fig_path", ("图片前一个标题", "上文", "下文"))]
    def generate_image_summaries(self, md_path:Path ,target_images_context:List[Tuple[str, str, Tuple[str, str, str]]]) -> dict[str, str]:
        summaries = {}
        for img_name, img_path, img_context in target_images_context:
            # 图片的路径
            img_file_path = str(img_path / img_name)
            # 调用方法生成每个图片概要
            summary = self.generate_image_summary( md_path,img_file_path, img_context)

            # 放入summaries
            summaries[img_name] = summary

        return summaries
    def generate_image_summary(self, md_path:Path ,img_file_path: str, img_context: Tuple[str, str, str]) -> str:
        # 图片+上下文 --> 摘要
        (front_title,pre_content,back_content) = img_context
        context_data = []
        # 不为空则加入列表
        if front_title:
            context_data.append(front_title)
        if pre_content:
            context_data.append(pre_content)
        if back_content:
            context_data.append(back_content)
        # final_data:得到的最终上下文数据
        final_data = '\n'.join(context_data)

        # 获取图片内容，一般而言，上传图片的二进制数据
        img_data_str = ""
        with open(img_file_path, "rb") as img_file:
            img_data_str = img_file.read()
            base64_img_data = base64.b64encode(img_data_str).decode('utf-8')
        # 根据图片后缀推断 MIME 类型
        img_mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }.get(Path(img_file_path).suffix.lower(), "image/png")
        # 调用视觉模型VLM

        # 创建client
        config = get_config()
        client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_api_base,
        )

        # 构建message
        document_title = md_path.stem
        # 构建message
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img_mime};base64,{base64_img_data}"
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"文档名称：{document_title}\n"
                            f"上下文内容：\n{final_data}\n"
                            "请结合以上文档名称、上下文内容和这张图片，为图片生成一个简洁的中文标题。"
                            "要求：直接输出标题本身，不超过20个字，不要加引号、标点或任何解释。"
                        ),
                    },
                ],
            }
        ]

        # 调用
        response = client.chat.completions.create(
            model= config.vl_model,
            messages=message,
            temperature=0,
        )

        # 获取response
        summary = (response.choices[0].message.content or "").strip()
        return summary

    def upload_img_update_md(self,md_path:Path,md_content:str,image_summaries:dict[str, str],target_images_context:List[Tuple[str, str, Tuple[str, str, str]]]):
        # 1.上传图片到minio，生成可访问地址
        # img_name : remote_url
        remote_urls = {}
        # 创建Minio_client
        minio_client = get_minio_client()
        # 上传图片，获取上传的每个图片的信息
        # list[("fig_name", "fig_path", ("图片前一个标题", "上文", "下文"))]
        for img_name, img_path, _ in target_images_context:
            img_file_path = str(img_path / img_name)
            # 路径+名称 放到根文件夹
            obj_name = f"{md_path.stem}/{img_name}"
            # 调用Minio_client方法
            minio_client.fput_object(
                bucket_name=os.getenv("MINIO_BUCKET_NAME"),
                object_name=obj_name,
                file_path=img_file_path,
            )
            # 拼接返回地址，minio没有提供自动返回的选项
            # http://<minio_address>/<bucket_name>/<dir>/<obj_name>
            remote_url = f"http://{os.getenv('MINIO_ENDPOINT')}/{os.getenv('MINIO_BUCKET_NAME')}/{quote(obj_name)}"
            remote_urls[img_name] = remote_url


        # 2.根据地址和摘要，更新md的内容
        # 遍历摘要
        new_md_content = md_content  # 先初始化为原文，逐张累积替换
        for img_name,img_summary in image_summaries.items():
            # 获取访问路径
            remote_url = remote_urls[img_name]
            if not remote_url:
                continue
            replace_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)", re.IGNORECASE)
            new_md_content = replace_pattern.sub(lambda match: f"![{img_summary}]({remote_url})", new_md_content)

        # 把替换后的内容写回 md 文件（可选，后续的操作都是针对状态存储的content进行的）
        # md_path.write_text(new_md_content, encoding='utf-8')
        return new_md_content


if __name__ == "__main__":
    setup_logging()
    md_image_node = MdImageNode()
    # 构建参数
    state = {
        "md_path": r"E:\掌柜智库\minerU-test\6W100-整本手册_span\6W100-整本手册_span.md"
    }

    process_result = md_image_node.process(state)
    print(json.dumps(process_result, indent=4, ensure_ascii=False))
