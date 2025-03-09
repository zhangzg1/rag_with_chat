from chatgpt_proxy import ChatGPTProxy
from rerank_model import reRankLLM
from vllm_model import ChatLLM
from retriever.m3e_retriever import M3eRetriever
from retriever.bge_retriever import BgeRetriever
from retriever.bm25_retriever import Bm25Retriever
from retriever.tfidf_retriever import TfidfRetriever
from generate_answer import get_emb_distribute_rerank


def test_example(model_name="qwen2",
                 reranker_name="bce",
                 m3e_embeddings_model_path="./pre_train_model/m3e-large",
                 bge_embeddings_model_path="./pre_train_model/bge-m3",
                 data_path="./all_text.txt",
                 m3e_vector_path="./vector_db/faiss_m3e_index",
                 bge_vector_path="./vector_db/faiss_bge_index",
                 mutil_max_length=4000,
                 mutil_top_k=6):

    # 调用大模型
    if "gpt" in model_name:
        llm = ChatGPTProxy(model=model_name)
    else:
        llm = ChatLLM(model_name)
    print("LLM model load ok")

    # 初始化检索器
    m3e_retriever = M3eRetriever(m3e_embeddings_model_path, data_path, m3e_vector_path)
    bge_retriever = BgeRetriever(bge_embeddings_model_path, data_path, bge_vector_path)
    bm25 = Bm25Retriever(data_path)
    tfidf = TfidfRetriever(data_path)
    print("Retriever load ok")

    # 调用reRank模型
    rerank = reRankLLM(reranker_name)
    print("rerank model load ok")

    while True:
        query = input("请输入问题（输入 'exit' 退出）：")
        if query.lower() == "exit":
            print("退出程序。")
            break

        # 多路召回检索文档
        m3e_context = m3e_retriever.GetTopK(query, 15)
        bge_context = bge_retriever.GetTopK(query, 15)
        bm25_context = bm25.GetBM25TopK(query, 15)
        tfidf_context = tfidf.GetBM25TopK(query, 15)

        # 重排检索文档
        mutil_rerank_inputs = get_emb_distribute_rerank(rerank, m3e_context, bge_context, bm25_context, tfidf_context,
                                                        query, max_length=mutil_max_length, top_k=mutil_top_k)

        # 获取回答
        answer = llm.infer([mutil_rerank_inputs])

        print("query: ", query)
        print("answer: ", answer)
        print("=" * 100)


if __name__ == '__main__':
    test_example()
