import importlib

# 模块名含连字符，无法直接 import，这里动态导入后对外导出
_bge_utils = importlib.import_module("shopkeer_brain.knowledge.utils.bge-m3_client_utils")
get_bgem3_client = _bge_utils.get_bgem3_client
