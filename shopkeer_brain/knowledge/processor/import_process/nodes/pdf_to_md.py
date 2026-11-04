import json
import os
import re
import subprocess
from pathlib import Path
from typing import Tuple

from shopkeer_brain.knowledge.processor.import_process.base import BaseNode, T, setup_logging
from shopkeer_brain.knowledge.processor.import_process.exceptions import PdfConversionError, FileProcessingError
from shopkeer_brain.knowledge.processor.import_process.state import ImportGraphState

# MinerU 可执行文件路径（可用环境变量 MINERU_EXE 覆盖）
MINERU_EXE = os.environ.get("MINERU_EXE", r"E:\miniconda3\envs\knowledge\Scripts\mineru.exe")
# MinerU 模型文件目录（可用环境变量 MINERU_MODEL_DIR 覆盖）
MINERU_MODEL_DIR = os.environ.get("MINERU_MODEL_DIR", r"E:\models")
# 解析档位：flash / basic / standard / advanced
MINERU_TIER = os.environ.get("MINERU_TIER", "advanced")

# 把模型目录传给 MinerU（MinerU 通过 MINERU_MODEL_BASE_DIR 环境变量读取模型路径）
os.environ["MINERU_MODEL_BASE_DIR"] = MINERU_MODEL_DIR


#模块 pdf --> md


# 1.对参数校验，文件是否存在
# 2.使用minerU将pdf转换为md格式
# 3.获取md的文件路径
# 4.将数据返回，包含文件信息
class PdfToMd(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.对参数校验，文件是否存在
        import_file_path, file_dir = self.validate_path(state)

        # 2.用minerU转换pdf文件
        # 转换成功返回0
        process_code = self.execute_minerU(import_file_path, file_dir)
        if process_code != 0:
            raise PdfConversionError(f"pdf转换失败:{process_code}")

        # 获取转换后的路径
        md_path = self.get_md_path(import_file_path, file_dir)

        # 返回数据
        state["md_path"] = md_path
        return state

    def execute_minerU(self, import_file_path: Path, file_dir: Path) -> int:
        self.log_step("execute_minerU", f"开始转换文件:{import_file_path}")
        # 命令行方式执行本地转换
        md_path = self.get_md_path(import_file_path, file_dir)
        # 命令列表
        cmd = [
            MINERU_EXE, "parse",
            str(import_file_path),
            "-o", md_path,
            "--tier", MINERU_TIER,
            "--wait", "300",
        ]

        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,  # 以文本方式输出
            encoding='utf-8',  # minerU 输出为 UTF-8
            errors='replace',
            bufsize=1,
        )
        # 输出日志
        for line in proc.stdout:
            self.log_step("execute_minerU", line)
        # 等待子进程执行完毕
        # 约定0为成功，非0为失败
        # 判断是否为0
        process_code = proc.wait()
        if process_code != 0:
            self.logger.error(f"pdf转换失败:{process_code}")
        else:
            self.log_step("execute_minerU", "pdf转换成功")
            # 导出图片到 images 文件夹，并把 md 中的 doc: 引用替换为本地路径
            self.export_images(Path(md_path))
        return process_code

    def get_md_path(self, import_file_path: Path, file_dir: Path):
        # 获取文件名
        file_name = import_file_path.stem
        # 以文件名作为输出文件夹，md 与 images 都放在该文件夹下
        md_path = file_dir / file_name / f"{file_name}.md"
        return str(md_path)


    # minerU4不再支持直接返回图片，需要手动导出
    def export_images(self, md_path: Path) -> None:
        """把 markdown 中的 doc: 图片引用批量导出为本地图片，并替换为 images/ 相对路径。"""
        images_dir = md_path.parent / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        text = md_path.read_text(encoding="utf-8")
        pattern = re.compile(r'!\[[^\]]*\]\((doc:[^)]+)\)')

        def replace(match) -> str:
            locator = match.group(1)
            page = re.search(r'/page:(\d+)', locator)
            block = re.search(r'/block:(\d+)', locator)
            page_no = page.group(1) if page else '0'
            block_no = block.group(1) if block else '0'
            fname = f'page{page_no}_block{block_no}.png'
            out = images_dir / fname
            try:
                subprocess.run(
                    [MINERU_EXE, 'read', locator, '-f', 'image', '-o', str(out)],
                    check=True, capture_output=True, text=True, timeout=180,
                    encoding='utf-8', errors='replace',
                )
            except Exception as exc:
                self.logger.warning(f"导出图片失败 {locator}: {exc}")
                return match.group(0)
            return f'![图片](images/{fname})'

        new_text = pattern.sub(replace, text)
        md_path.write_text(new_text, encoding='utf-8')
        self.log_step("export_images", f"图片已导出到:{images_dir}")

    def validate_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        pdf_path = state["import_file_path"]
        # 调用path方法判断文件是否存在
        path = Path(pdf_path)
        if not path.exists():
            raise FileProcessingError(f"文件不存在:{pdf_path}")

        # 输出目录：默认值为文件路径所在目录，可通过 state 的 file_dir 指定
        file_dir = Path(state["file_dir"]) if state.get("file_dir") else path.parent
        return path, file_dir


if __name__ == "__main__":
    setup_logging()
    pdf_to_md_node = pdf_to_md()
    # 构建参数
    pdf_to_md_node_state = {
        "import_file_path": r"E:\掌柜智库\minerU-test\6W100-整本手册_span.pdf",
        # 输出目录（可选），默认值为文件所在目录
        "file_dir": r"E:\掌柜智库\minerU-test"
    }

    process_result = pdf_to_md_node.process(pdf_to_md_node_state)
    print(json.dumps(process_result, indent=4, ensure_ascii=False))
