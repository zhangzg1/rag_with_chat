from typing import List, Dict
from collections import defaultdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义请求体的数据结构
class ChatRequest(BaseModel):
    message: List[Dict[str, str]]


# 全局变量
model_path = "/home/zhangzg/models/Qwen2-7B-Instruct"
llm = None
sampling_params = SamplingParams(temperature=0, max_tokens=64, top_k=1)


# 在服务启动时加载模型
@app.on_event("startup")
async def load_model():
    global llm, model_path
    llm = LLM(model=model_path, tensor_parallel_size=1)


# 处理聊天请求
@app.post("/qwen")
async def qwen(chat_request: ChatRequest):
    message = chat_request.message
    result = defaultdict(str)

    # 使用模型生成回复
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text

    result["role"] = "assistant"
    result["content"] = response
    return result


# 启动服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("model_serve:app", host="127.0.0.1", port=8000, reload=True)
