"""插件配置。

在 NoneBot 的 .env 中可按字段名覆盖，例如：
    ollama_url="http://127.0.0.1:11434"
    light_model="qwen2.5:3b"
    max_rounds=20
"""
from pydantic import BaseModel
from typing import List


class Config(BaseModel):
    """Plugin Config Here"""

    # 可供 /change 命令切换的离线模型列表
    Model_list: List[str] = ["CHAT_OFFLINE", "q4_chaoji"]
    # 走联网检索链路时使用的模型
    online_model: str = "CHAT_ONLINE"
    # 用于意图分类与检索词生成的轻量模型
    light_model: str = "qwen2.5:3b"
    # Ollama 服务地址
    ollama_url: str = "http://localhost:11434"
    # 会话上下文保留的最大轮数
    max_rounds: int = 20
    # 请求 Ollama 的超时秒数
    timeout: int = 20
