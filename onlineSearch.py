"""联网检索：DuckDuckGo 搜索 + 模型摘要压缩。

注意：本模块是同步实现（ddgs 与 ollama 客户端均为同步接口），
调用方需放到线程池执行，否则会阻塞 NoneBot 的事件循环。
"""
import logging

from ddgs import DDGS
import ollama

logger = logging.getLogger(__name__)


def search_longtext(query_text: str, max_results=10, region: str = "zh-CN", backend: str = "bing") -> str:
    """
    搜索长文本并总结

    Args:
        query_text: 长文本查询
        region: 搜索区域
        backend: 搜索后端

    Returns:
        总结后的结果字符串
    """
    if not query_text or len(query_text.strip()) < 3:
        return "查询文本太短，请提供更多信息"

    # 尝试用前200个字符作为搜索词（避免查询过长）
    search_key = query_text[:200].strip()

    # 搜索
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(search_key, max_results=max_results, region=region, backend=backend):
                if r and r.get('title') and r.get('body'):
                    results.append(r)

            if not results:
                return "未找到相关信息"

            # 构建上下文
            context_lines = [f"用户查询: {query_text}\n\n搜索到以下信息:"]

            for i, r in enumerate(results, 1):
                title = r.get('title', '无标题')
                snippet = r.get('body', '')[:1000]  # 截取前1000字符
                context_lines.append(f"{i}. {title}")
                context_lines.append(f"   {snippet}...")

            context = "\n".join(context_lines)

            # 用模型总结
            prompt = f"""问题: {query_text}\n搜索结果:\n{context}\n请基于问题和搜索结果提取有用信息并总结一段文本，不要太长，尽可能包含搜索结果中的关键信息。"""

            try:
                response = ollama.chat(
                    model="qwen2.5:3b",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.3}
                )
                return response['message']['content']
            except Exception as e:
                # 模型失败，返回原始结果
                logger.warning("搜索结果摘要失败，回退原始检索结果: %s", e)
                return context

    except Exception as e:
        logger.exception("联网搜索失败: %s", e)
        return "搜索时出错，请稍后再试"
