"""使用本地 Ollama 模型把用户消息改写成检索用的查询词。"""
import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)


class OllamaQuestionExtractor:
    """使用Ollama模型根据输入文本生成一个用于搜索的问题或文本"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:3b"):
        self.base_url = base_url
        self.model = model

    async def generate_question(self, text: str) -> str:
        """
        使用Ollama模型根据输入文本生成一个用于搜索的问题或文本

        参数:
            text: 输入文本
        返回:
            问题文本
        """
        if not text:
            return ""

        # 去除首尾空白如果处理后为空，返回空列表
        stripped = text.strip()
        if not stripped:
            return ""

        # 构建提示词
        prompt = f"""请根据用户消息来生成一个用于搜索的问题或者文本

用户消息：{text}

要求：
1.尽可能的不要太长
2.不要包含引导词，直接给我返回搜索用的问题或文本
3.要尽可能的包含用户消息中提到的关键内容尤其是名词，专有名词，动词，形容词，时间日期等重要信息
"""

        try:
            async with aiohttp.ClientSession() as session:
                # 调用Ollama API
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # 低温度，更稳定
                            "num_predict": 500,   # 限制输出长度
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data.get("response", "").strip()
                        return response_text
                    else:
                        logger.warning("检索词生成接口返回 %s，将以非联网搜索模式执行", response.status)
                        return ""

        except asyncio.TimeoutError:
            logger.warning("检索词生成请求超时，将以非联网搜索模式执行")
            return ""
        except Exception as e:
            logger.exception("生成检索词失败，将以非联网搜索模式执行: %s", e)
            return ""
