# 简单的测试脚本
import aiohttp
import asyncio

async def test_ollama():
    try:
        async with aiohttp.ClientSession() as session:
            # 测试API可用性
            async with session.get("http://localhost:11434/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"Ollama服务正常，模型: {data.get('models', [])}")
                else:
                    print(f"API返回错误: {resp.status}")
    except Exception as e:
        print(f"连接失败: {e}")

# 运行测试
asyncio.run(test_ollama())

