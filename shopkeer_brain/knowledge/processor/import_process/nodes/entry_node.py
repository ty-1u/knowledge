#定义节点操作类
import json
from pathlib import Path

from shopkeer_brain.knowledge.processor.import_process.base import BaseNode, T, setup_logging
from shopkeer_brain.knowledge.processor.import_process.exceptions import ValidationError
from shopkeer_brain.knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    # 实现basenode的process方法
    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("步骤1","[开始检查文件类型]")
        # 获取文件路径
        file_dir = state["file_dir"]
        import_file_path = state["import_file_path"]
        # 获取文件名，看是否是pdf或者md
        path = Path(import_file_path)
        # 得到文件后缀名
        suffix = path.suffix.lower()
        # 判断文件类型
        if suffix == ".pdf":
            self.log_step("pdf","[文件类型为pdf]")
            state["pdf_path"] = import_file_path
            state["is_pdf_read_enabled"] = True
        elif suffix == ".md":
            self.log_step("md","[文件类型为md]")
            state["md_path"] = import_file_path
            state["is_md_read_enabled"] = True
        else:
            # 不是可处理类型，抛验证异常
            self.log_step("error","[文件类型错误]")
            raise ValidationError("文件类型错误")

        # 获取文件的标题
        file_title = path.stem
        state["file_title"] = file_title
        # 返回已经更新的实体类
        return state



if __name__ == '__main__':
    #日志初始化
    setup_logging()

    # 构建字典数据
    entry_state = {
        "file_dir": r"shopkeer_brain/knowledge/processor/import_process/import_temp_dir",
        "import_file_path": r"shopkeer_brain/knowledge/processor/import_process/import_temp_dir/hak180产品安全手册.pdf",
    }


    # 实例化节点（不带参数，使用默认配置）
    entry_node = EntryNode()

    # 调用节点处理状态：__call__ 会自动执行 process 并返回更新后的 state
    res = entry_node(entry_state)

    print(json.dumps(res, ensure_ascii=False, indent=4))


