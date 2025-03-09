import torch
import os

# 加载文本嵌入模型所使用的设备
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# 加载大模型所使用的设备
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# 设备上GPU的数量
num_gpus = torch.cuda.device_count()

# LLM模型路径
Qwen2_path = './models/Qwen2-7B-Instruct'
Baichuan_path = './models/Baichuan2-7B-Chat'
ChatGLM_path = './models/chatglm3-6b'

# 召回模型路径
M3E_embeddings_model_path = "./pre_train_model/m3e-large"
BGE_embeddings_model_path = "./pre_train_model/bge-m3"

# 重排模型路径
BGE_reranker_model = "./pre_train_model/bge-reranker-large"
BCE_reranker_model = "./pre_train_model/bce-reranker-base_v1"

# 相似度模型
SimModel_path = './pre_train_model/text2vec-base-chinese'
