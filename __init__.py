import asyncio
import logging

from nonebot.rule import to_me
from nonebot import get_plugin_config
from nonebot.plugin import PluginMetadata
from nonebot import on_message, on_command
from nonebot.params import EventMessage, CommandArg
from nonebot.adapters import Event, Message
import aiohttp
from typing import Dict, List
from nonebot.params import ArgPlainText

from .intentModel import OllamaIntentClassifier
from .question import OllamaQuestionExtractor
from .onlineSearch import search_longtext
from .config import Config

logger = logging.getLogger(__name__)

__plugin_meta__ = PluginMetadata(
    name="ollama_plugin",
    description="通过llm来实现意图判断、联网搜索、回复生成的llm插件 ",
    usage="@机器人或者私信都可以\n/clear - 清除当前会话历史",
    config=Config,
)

config = get_plugin_config(Config)

# 当前使用的离线模型，可通过 /change 命令切换
OFFLINE_MODEL = config.Model_list[0]

choice = ""
index = 1
for model in config.Model_list:
    choice += f"\n{index}.{model}"
    index = index + 1

# ============ 缓存和上下文管理 ============
# 按会话存储消息历史，格式：{session_id: [message1, message2, ...]}
conversation_cache: Dict[str, List] = {}
# 保护会话缓存的并发访问（多个群/用户可能同时触发读写）
_cache_lock = asyncio.Lock()


def get_session_id(event: Event) -> str:
    """生成会话ID，用于区分不同聊天环境的上下文"""
    try:
        user_id = event.get_user_id()
        # 群聊：每个群共享同一份上下文
        if hasattr(event, 'group_id') and event.group_id:
            return f"group_{event.group_id}"
        # 私聊：每个用户独立的上下文
        return f"private_{user_id}"
    except Exception:
        logger.warning("获取会话 ID 失败，回退到默认会话")
        return "default_session"


def get_session_messages(session_id: str) -> List:
    """获取或创建会话消息历史"""
    if session_id not in conversation_cache:
        conversation_cache[session_id] = []
    return conversation_cache[session_id]


def clear_session_messages(session_id: str = None):
    """清除会话消息"""
    if session_id:
        if session_id in conversation_cache:
            conversation_cache[session_id] = []
    else:
        conversation_cache.clear()


def session_messagelength_controller(session_id: str, max_rounds: int) -> List:
    if session_id not in conversation_cache:
        return []
    message_list = conversation_cache[session_id]
    # 一轮对话 = 一问一答两条消息，超限时从最旧的一轮开始丢弃
    while len(message_list) > max_rounds * 2:
        message_list.pop(0)
        message_list.pop(0)
    conversation_cache[session_id] = message_list
    return message_list


# ============ 缓存和上下文管理结束 ============

# ============ 清空命令 ============
clear_cmd = on_command("clear", aliases={"清空", "清除历史"}, rule=to_me(), block=True, priority=20)
change_cmd = on_command("change", aliases={"变更模型", "换模型"}, block=True, priority=20)


@change_cmd.got("model_index", prompt=f"选择你要变更的非联网搜索模型序号：{choice}")
async def handle_change(event: Event, model_index: str = ArgPlainText("model_index")):
    global OFFLINE_MODEL
    try:
        idx = int(model_index.strip()) - 1
        if idx < 0 or idx >= len(config.Model_list):
            await change_cmd.reject("序号超出范围，请重新输入。")
            return
        OFFLINE_MODEL = config.Model_list[idx]
        await change_cmd.send(f"已将 offline 模型更换为 {OFFLINE_MODEL}")
    except ValueError:
        await change_cmd.reject("请输入数字序号。")


@clear_cmd.handle()
async def handle_clear(event: Event, args: Message = CommandArg()):
    """清除对话历史命令"""
    session_id = get_session_id(event)
    arg = args.extract_plain_text().strip()

    if arg == "all":
        async with _cache_lock:
            clear_session_messages()  # 清除所有历史
        await clear_cmd.finish("已清除所有对话历史记录。")
    else:
        async with _cache_lock:
            clear_session_messages(session_id)  # 只清除当前会话
        await clear_cmd.finish("已清除当前对话的历史记录。")


# ============ 清空命令结束 ============

# ============ 主消息处理器 ============
ollama = on_message(rule=to_me(), priority=20)


@ollama.handle()
async def handle_message(event: Event):
    QID = event.get_user_id()  # 获取用户QQ号
    text = event.get_plaintext().strip()

    if not text:
        return

    session_id = get_session_id(event)

    # 原有的意图判断逻辑
    search_intent_model = OllamaIntentClassifier(
        base_url=config.ollama_url, model=config.light_model
    )
    need_search = await search_intent_model.classify_intent(text)  # 判断是否需要联网搜索

    if need_search == "yes":
        model = config.online_model
        question_extractor = OllamaQuestionExtractor(
            base_url=config.ollama_url, model=config.light_model
        )
        question = await question_extractor.generate_question(text)  # 提取问题
        # search_longtext 内部使用同步的 ddgs 与 ollama 客户端，
        # 直接调用会阻塞整个事件循环，因此放到线程池中执行
        search_result = await asyncio.to_thread(search_longtext, question)
        message = f"""
    QID: {QID}
    need_search: {need_search}
    text: {text}
    search_result: {search_result}
""".strip()
    else:
        model = OFFLINE_MODEL
        message = f"""
    QID: {QID}
    text: {text}
""".strip()

    # 历史上下文 + 本条带元信息的消息组成请求体；
    # 这里只取快照，避免请求进行中缓存被并发修改
    async with _cache_lock:
        history_messages = get_session_messages(session_id)
        request_messages = list(history_messages) + [{"role": "user", "content": message}]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config.ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": request_messages,
                    "stream": False,
                },
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    reply = data.get("message", {}).get("content", "").strip()

                    if reply:
                        # 只在缓存中保存原始对话，不保存带元信息的临时消息
                        async with _cache_lock:
                            history = get_session_messages(session_id)
                            history.append({"role": "user", "content": text})
                            history.append({"role": "assistant", "content": reply})
                            session_messagelength_controller(session_id, config.max_rounds)
                        await ollama.send(reply)
                else:
                    logger.warning("Ollama 返回异常状态码: %s", response.status)
                    await ollama.send("抱歉，我没有理解你的意思。")
                return
    except asyncio.TimeoutError:
        logger.warning("请求 Ollama 超时: session=%s model=%s", session_id, model)
        await ollama.finish("抱歉，模型想太久超时了，稍后再问一次吧~")
    except aiohttp.ClientError as e:
        # 内部错误只记日志，不把细节抛给用户
        logger.exception("连接 Ollama 失败: session=%s err=%s", session_id, e)
        await ollama.finish("抱歉，我有点不太舒服喵，可能是模型没启动，检查完后再来和我聊天吧~")
    except Exception:
        logger.exception("处理消息时发生未预期错误: session=%s", session_id)
        await ollama.finish("抱歉，我有点不太舒服喵，检查完后再来和我聊天吧~")
