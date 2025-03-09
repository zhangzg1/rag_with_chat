from transformers import AutoModelForSequenceClassification, AutoTokenizer
from retriever.bm25_retriever import Bm25Retriever
from retriever.tfidf_retriever import TfidfRetriever
from retriever.m3e_retriever import M3eRetriever
from retriever.bge_retriever import BgeRetriever
from config import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


# 释放gpu上没有用到的显存以及显存碎片
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# 加载reranker模型
class reRankLLM(object):
    def __init__(self, reranker_name, max_length=512):
        if reranker_name == "bce":
            self.reranker_path = BCE_reranker_model
        if reranker_name == "bge":
            self.reranker_path = BGE_reranker_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.reranker_path)
        self.model.eval()
        self.model.half()
        self.model.cuda()
        self.max_length = max_length

    # 基于召回模型检索得到的文档对，返回每一对(query, doc)的相关得分，并从大到小排序
    def predict(self, query, docs):
        pairs = [(query, doc.page_content) for doc in docs]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt',
                                max_length=self.max_length).to("cuda")
        with torch.no_grad():
            scores = self.model(**inputs).logits
        scores = scores.detach().cpu().clone().numpy()
        response = [doc for score, doc in sorted(zip(scores, docs), reverse=True, key=lambda x: x[0])]

        torch_gc()
        return response


if __name__ == "__main__":
    reranker_name = "bce"
    m3e_embeddings_model_path = "pre_train_model/m3e-large"
    bge_embeddings_model_path = "pre_train_model/bge-m3"
    m3e_vector_path = "vector_db/faiss_m3e_index"
    bge_vector_path = "vector_db/faiss_bge_index"
    data_path = "./all_text.txt"
    pdf_path = "data/car_user_manual.pdf"

    docs = []
    query = "交通事故如何处理"

    # m3e召回
    m3e_retriever = M3eRetriever(m3e_embeddings_model_path, data_path, m3e_vector_path)
    m3e_ans = m3e_retriever.GetTopK(query, 6)
    for doc, score in m3e_ans:
        docs.append(doc)

    # bge召回
    bge_retriever = BgeRetriever(bge_embeddings_model_path, data_path, bge_vector_path)
    bge_ans = bge_retriever.GetTopK(query, 6)
    for doc, score in bge_ans:
        docs.append(doc)

    # bm25召回
    bm25 = Bm25Retriever(data_path)
    bm25_ans = bm25.GetBM25TopK(query, 6)
    docs.extend(bm25_ans)

    # tf_idf召回
    tfidf = TfidfRetriever(data_path)
    tfidf_ans = tfidf.GetBM25TopK(query, 6)
    docs.extend(tfidf_ans)

    rerank = reRankLLM(reranker_name)
    rerank_text = rerank.predict(query, docs)
    print(rerank_text)
