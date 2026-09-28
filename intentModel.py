"""使用本地 Ollama 模型判断用户消息是否需要联网检索。"""
import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)


class OllamaIntentClassifier:
    """使用Ollama模型进行意图分类"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:3b"):
        self.base_url = base_url
        self.model = model

    async def classify_intent(self, text: str) -> str:
        """
        使用Ollama模型判断用户查询是否需要搜索

        参数:
            text: 用户输入的查询文本
        返回:
            "yes" 或 "no" - 表示是否需要搜索
        """
        if not text or len(text.strip()) < 1:
            return "no"  # 空文本或过短的文本不需要搜索

        prompt = f"""现在假设你正在群里聊天，一些群友对你发送的消息是需要搜索后得到确切信息才能够很好的回复，
        一些消息则是不需要，只需要进行正常对话就能很好回复，
        请判断一下当前的用户消息 是否需要搜索后得到确切信息才能够很好地回复。
        用户消息：{text}
        只需要回答"yes"或"no"，不要有其他文字。"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0001,  # 低温度，更稳定
                            "num_predict": 10,   # 限制输出长度
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=30)  # 添加超时设置
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # 修正：使用正确的字段名"response"
                        text_response = data.get("response", "").strip()

                        # 清理响应，只保留"yes"或"no"
                        if "yes" in text_response:
                            return "yes"
                        elif "no" in text_response:
                            return "no"
                        else:
                            # 如果模型返回了其他内容，默认返回"no"
                            logger.warning("意图分类返回非预期响应，按 no 处理: %r", text_response)
                            return "no"
                    else:
                        error_text = await response.text()
                        logger.warning("意图分类接口返回 %s: %s", response.status, error_text)
                        return "no"  # 统一返回"no"
        except aiohttp.ClientError as e:
            logger.warning("意图分类网络连接失败，按 no 处理: %s", e)
            return "no"
        except asyncio.TimeoutError:
            logger.warning("意图分类请求超时，按 no 处理")
            return "no"
        except Exception as e:
            logger.exception("意图分类发生未预期错误，按 no 处理: %s", e)
            return "no"
