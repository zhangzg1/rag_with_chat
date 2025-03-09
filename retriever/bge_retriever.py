from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from pdf_parse import DataProcess
import torch
import os

script_dir = os.path.dirname(__file__)


# Sparse语义召回BGE-M3
class BgeRetriever(object):
    def __init__(self, embeddings_model_path=None, data_path=None, vector_path=None, pdf_path=None):
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embeddings_model_path,
            model_kwargs={"device": "cuda"},
            encode_kwargs={'normalize_embeddings': True}
        )
        if vector_path is None:
            docs = []
            if data_path is not None:
                with open(data_path, "r", encoding="utf-8") as file:
                    docs = self.data_process(file)
            if pdf_path is not None:
                dp = DataProcess(pdf_path)
                dp.ParseBlock(max_seq=1024)
                dp.ParseBlock(max_seq=512)
                dp.ParseAllPage(max_seq=256)
                dp.ParseAllPage(max_seq=512)
                dp.ParseOnePageWithRule(max_seq=256)
                dp.ParseOnePageWithRule(max_seq=512)
                print("bge pdf_parse is ok")
                docs = self.data_process(dp.data)
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(os.path.normpath(os.path.join(script_dir, "../vector_db/faiss_bge_index")))
            print("bge faiss vector_db is ok")
        else:
            self.vector_store = FAISS.load_local(vector_path, self.embeddings, allow_dangerous_deserialization=True)

        del self.embeddings
        torch.cuda.empty_cache()

    # 对pdf解析后的文本数据进行处理
    def data_process(self, data):
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            docs.append(Document(page_content=words[0], metadata={"id": idx}))
        return docs

    # 获取top_K分数最高的文档块
    def GetTopK(self, query, k):
        context = self.vector_store.similarity_search_with_score(query, k=k)
        return context

    def GetvectorStore(self):
        return self.vector_store


if __name__ == "__main__":
    embeddings_model_path = "../pre_train_model/bge-m3"
    data_path = "../all_text.txt"
    vector_path = "../vector_db/faiss_bge_index"
    pdf_path = "../data/car_user_manual.pdf"

    # faiss_retriever = BgeRetriever(embeddings_model_path, pdf_path)
    faiss_retriever = BgeRetriever(embeddings_model_path, data_path, vector_path)
    faiss_ans = faiss_retriever.GetTopK("座椅加热", 3)
    print(faiss_ans)
