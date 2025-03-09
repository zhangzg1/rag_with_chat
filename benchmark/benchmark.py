import aiohttp
import asyncio
import json
import logging
import time
from typing import List, Tuple
import numpy as np

# 日志配置
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 存储请求的延迟信息
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

# API配置
API_URL = 'http://127.0.0.1:8000/qwen'
HEADERS = {'Content-Type': 'application/json'}


# 发送HTTP POST请求到指定的API，并处理响应结果
async def send_request(session, payload, prompt_len):
    try:
        request_start_time = time.time()
        async with session.post(API_URL, data=payload, headers=HEADERS) as response:
            if response.status == 200:
                result = await response.json()
                content = result.get('content', '')
                completion_tokens = len(content)
                request_end_time = time.time()
                request_latency = request_end_time - request_start_time
                REQUEST_LATENCY.append((prompt_len, completion_tokens, request_latency))
                return result
            else:
                error_msg = await response.text()
                logger.error(f"Error {response.status}: {error_msg}")
                return {'error': True, 'message': error_msg}
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return {'error': True, 'message': str(e)}


# 基准测试运行器
class BenchMarkRunner:
    def __init__(self, requests: List[Tuple[str, int, int]], concurrency: int):
        self.concurrency = concurrency
        self.requests = requests
        self.request_left = len(requests)
        self.request_queue = asyncio.Queue()

    # 启动基准测试
    async def run(self):
        tasks = []
        for i in range(self.concurrency):
            task = asyncio.create_task(self.worker())
            tasks.append(task)

        for req in self.requests:
            await self.request_queue.put(req)

        await asyncio.gather(*tasks)

    # 处理队列中的请求
    async def worker(self):
        timeout = aiohttp.ClientTimeout(total=2 * 60)  # 增加超时时间
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while self.request_left > 0:
                try:
                    prompt = await self.request_queue.get()
                    message = [{"role": "user", "content": prompt}]
                    payload = json.dumps({"message": message})

                    response = await send_request(session, payload, len(prompt))
                    if 'error' in response:
                        logger.error(f"Request failed: {response['message']}")
                    else:
                        logger.info(f"Response {len(self.requests) - self.request_left}")

                    self.request_left -= 1
                except Exception as e:
                    logger.error(f"Worker error: {str(e)}")
                    break


# 主函数
def main():
    # 并发任务数量
    concurrency = 2
    logger.info("Preparing for benchmark.")

    # 加载测试数据
    with open("bench_data.json", "r") as f:
        test_set = json.load(f)
    input_requests = list(test_set.values())

    logger.info("Benchmark starts.")
    benchmark_start_time = time.time()
    asyncio.run(BenchMarkRunner(input_requests, concurrency).run())
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    # 计算并打印基准测试时间，和吞吐量（请求/秒）：请求数量除以总时间
    print(f"Total time: {benchmark_time:.4f} s")
    print(f"Throughput: {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")

    # 计算并打印所有延迟的平均值
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.4f} s")

    # 计算并打印每个token的平均延迟：延迟除以提示长度和输出长度之和
    avg_per_token_latency = np.mean(
        [latency / (prompt_len + output_len) for prompt_len, output_len, latency in REQUEST_LATENCY]
    )
    print(f"Average latency per token: {avg_per_token_latency:.4f} s")

    # 计算并打印每输出令牌平均延迟：延迟除以输出长度
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    )
    print(f"Average latency per output token: {avg_per_output_token_latency:.4f} s")

    # 计算并打印token吞吐量（token/s）：总输出长度除以基准测试时间
    throughput = sum([output_len for _, output_len, _ in REQUEST_LATENCY]) / benchmark_time
    print(f"Throughput: {throughput} tokens/s")


if __name__ == '__main__':
    main()
