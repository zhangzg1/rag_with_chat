from langchain.retrievers import TFIDFRetriever
from langchain.schema import Document
from pdf_parse import DataProcess
import jieba


# TF-IDF召回
class TfidfRetriever(object):
    def __init__(self, data_path=None, pdf_path=None):
        docs = []
        full_docs = []
        if data_path is not None:
            with open(data_path, "r", encoding="utf-8") as file:
                docs, full_docs = self.data_process(file)
        if pdf_path is not None:
            dp = DataProcess(pdf_path)
            dp.ParseBlock(max_seq=1024)
            dp.ParseBlock(max_seq=512)
            dp.ParseAllPage(max_seq=256)
            dp.ParseAllPage(max_seq=512)
            dp.ParseOnePageWithRule(max_seq=256)
            dp.ParseOnePageWithRule(max_seq=512)
            print("tfidf pdf_parse is ok")
            docs, full_docs = self.data_process(dp.data)
        self.documents = docs
        self.full_documents = full_docs
        self.retriever = self._init_tf_ide()

    # 对pdf解析后的文本数据进行处理
    def data_process(self, data):
        docs = []
        full_docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            if (len(line) < 5):
                continue
            tokens = " ".join(jieba.cut_for_search(line))
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            words = line.split("\t")
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))
        return docs, full_docs

    # 初始化BM25的知识库
    def _init_tf_ide(self):
        return TFIDFRetriever.from_documents(self.documents)

    # 获得得分在top_k的文档和分数
    def GetBM25TopK(self, query, top_k):
        self.retriever.k = top_k
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.get_relevant_documents(query)
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans


if __name__ == "__main__":
    data_path = "../all_text.txt"
    pdf_path = "../data/car_user_manual.pdf"

    # bm25 = TfidfRetriever(pdf_path)
    bm25 = TfidfRetriever(data_path)
    res = bm25.GetBM25TopK("座椅加热", 6)
    print(res)
