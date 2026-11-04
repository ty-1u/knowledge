# 下载嵌入模型
import os

from dotenv import load_dotenv
from modelscope import snapshot_download
# 加载环境变量的参数
load_dotenv()
path = snapshot_download(os.getenv('MODEL_NAME'), local_dir=os.getenv('MODEL_DIR'))
print(path)