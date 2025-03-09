import time
from config import *
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


# 释放gpu显存
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ChatLLM(object):
    def __init__(self, model_name):
        self.model_name = model_name
        if self.model_name == "qwen2":
            self.model_path = Qwen2_path
        if self.model_name == "baichuan2":
            self.model_path = Baichuan_path
        if self.model_name == "chatglm3":
            self.model_path = ChatGLM_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left', trust_remote_code=True)
        self.model = LLM(model=self.model_path,
                         tokenizer=self.model_path,
                         tensor_parallel_size=1,       # 如果是多卡，可以自己把这个并行度设置为卡数N
                         trust_remote_code=True,
                         gpu_memory_utilization=0.9,   # 可以根据gpu的利用率自己调整这个比例，默认0.9
                         dtype="bfloat16")
        # LLM的采样参数
        sampling_kwargs = {
            "stop_token_ids": [self.tokenizer.eos_token_id],
            "early_stopping": False,
            "top_p": 1.0,
            "top_k": -1,                 # 当使用束搜索时top_k必须为-1
            "temperature": 0.0,
            "max_tokens": 2000,
            "repetition_penalty": 1.05,
            "n": 1,
            "best_of": 2,                # 生成的候选数量和最佳选择数量
            "use_beam_search": True      # 是否使用束搜索
        }
        self.sampling_params = SamplingParams(**sampling_kwargs)

    # 批量推理，输入一个batch，返回一个batch的答案
    def infer(self, prompts):
        batch_text = []
        for q in prompts:
            if self.model_name == "qwen2":
                text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            if self.model_name == "baichuan2":
                text = f"<reserved_106>{q}<reserved_107>"
            if self.model_name == "chatglm3":
                text = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{q}\n<|assistant|>\n"
            batch_text.append(text)

        outputs = self.model.generate(batch_text, sampling_params=self.sampling_params)
        batch_response = []
        for output in outputs:
            output_str = output.outputs[0].text
            if self.tokenizer.eos_token in output_str:
                output_str = output_str[:-len(self.tokenizer.eos_token)]
            if self.tokenizer.pad_token in output_str:
                output_str = output_str[:-len(self.tokenizer.pad_token)]
            batch_response.append(output_str)

        torch_gc()
        return batch_response


if __name__ == "__main__":
    model_name = "qwen2"
    start = time.time()
    llm = ChatLLM(model_name)
    test = ["你好", "吉利汽车语音组手唤醒", "自动驾驶功能介绍"]
    generated_text = llm.infer(test)
    print(generated_text)
    end = time.time()
    print("cost time: " + str((end - start) / 60) + "minutes")
