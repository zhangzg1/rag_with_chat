# 基于RAG的汽车知识问答系统

## 1、介绍

本项目属于大模型RAG任务，使用现有的车主手册构建知识库，然后选择知识库中的相关知识用于辅助大模型生成。整个方案的构建流程主要分为三大部分：构建知识库、知识检索、答案生成。该项目主要结合了 LLM、L angchain、提示工程、优化知识库结构和检索生成流程、vllm 推理优化框架等技术。
## 2、下载源码与环境安装（Linux）

```
# 下载源码
git clone https://github.com/zhangzg1/rag_with_chat.git
cd rag_with_chat

# 创建虚拟环境
conda create -n rag_with_chat python=3.9
conda activate rag_with_chat

# 安装其他依赖包
pip install -r requirements.txt
```

## 2、代码结构

```text
.
├── benchmark
	└── bench_data.json              # 基准测试数据
    └── benchmark.py                 # 基座测试
    └── model_server.py              # 模型服务
├── data
    └── gold_result.jsonn            # 标准答案数据集     
    └── test_question.json           # 测试数据集 
    └── car_user_manual.pdf          # 汽车用户手册文件
├── images 
├── models                           # 基座大语言模型
	└── Baichuan2-7B-Chat        
    └── chatglm3-6b
	└── Qwen2-7B-Instruct        
├── pre_train_model 
	└── bce-reranker-base_v1         # bce重排序模型
	└── bge-reranker-large           # bge重排序模型 
    └── bge-m3                       # bge文本嵌入模型
    └── m3e-large                    # m3e文本嵌入模型 
	└── text2vec-base-chinese        # 相似度模型     
├── retriever
	└── bge_retriever.py             # bge召回    
    └── bm25_retriever.py            # bm25召回      
    └── m3e_retriever.py             # m3e召回
    └── tfidf_retriever.py           # tf-idf召回
├── .env                             # API 密钥
├── config.py                        # 配置文件
├── pdf_parse.py                     # pdf文档解析器
├── rerank_model.py                  # 重排序逻辑
├── generate_answer.py               # rag流程
├── chatgpt_proxy.py                 # chatgpt代理
├── vllm_model.py                    # vllm大模型加速
├── test_score.py                    # 测试集得分计算
├── example_test.py                  # 测试样例
├── run.py                           # 主函数
├── requirements.txt                 # 第三方依赖库
├── README.md                        # 说明文档             
```

## 3、项目概述

### 3.1 基于大模型的文档检索问答

任务：项目要求以大模型为中心制作一个问答系统，回答用户的汽车相关问题。需要根据问题，在文档中定位相关信息的位置，并根据文档内容通过大模型生成相应的答案。本项目涉及的问题主要围绕汽车使用、维修、保养等方面，具体可参考下面的例子：

```text
问题1：怎么打开危险警告灯？
答案1：危险警告灯开关在方向盘下方，按下开关即可打开危险警告灯。

问题2：车辆如何保养？
答案2：为了保持车辆处于最佳状态，建议您定期关注车辆状态，包括定期保养、洗车、内部清洁、外部清洁、轮胎的保养、低压蓄电池的保养等。

问题3：靠背太热怎么办？
答案3：您好，如果您的座椅靠背太热，可以尝试关闭座椅加热功能。在多媒体显示屏上依次点击空调开启按键→座椅→加热，在该界面下可以关闭座椅加热。
```

### 3.2 数据集

这里的训练数据集主要是一本汽车的用户手册（pdf文件）：

![image](https://github.com/zhangzg1/rag_with_chat/blob/main/images/image_fChhMjnifo.png)

测试集问题示例：

```json
{
    "question": "自动模式下，中央显示屏是如何切换日间和夜间模式的？",
    "answer_1": "",
    "answer_2": "",
    "answer_3": ""
},
{
    "question": "如何通过中央显示屏进行副驾驶员座椅设置？",
    "answer_1": "",
    "answer_2": "",
    "answer_3": ""
}
```

## 4、项目流程

### 4.1 pdf解析

![image](https://github.com/zhangzg1/rag_with_chat/blob/main/images/image_RiYKWHwtQa.png)

对于 pdf 文件中这里类似的文本内容，该项目最终采用了三种解析方案的综合：

- pdf分块解析，尽量保证一个小标题+对应文档在一个文档块，其中文档块的长度分别是512和1024。

- pdf滑窗法解析，把文档句号分割，然后构建滑动窗口，其中文档块的长度分别是256和512。

- pdf非滑窗法解析，把文档句号分割，然后按照文档块预设尺寸均匀切分，其中文档块的长度分别是256和512。

按照这个三种解析方案对数据处理之后，然后对文档块做了一个去重，最后把这些文档块输入给召回模块。使用三种解析方法的综合，可以保证文本内容的完整性和跨页连续性。

### 4.2 召回

召回主要使用 langchain 中的 retrievers 进行文本的召回。我们知道深度语义召回，侧重泛化性，字面召回，侧重关键词/实体的字面相关性，这两个召回方法也比较有代表性，因此选用了这两个召回方法。

1. 深度语义召回：这里我们使用了 m3e 召回和 bge 召回两种方法，m3e和bge都是文本嵌入模型，所以我们使用这两种模型分别将处理后的 pdf 文件转换成向量，最后都使用 faiss 向量数据库进行存储。
2. 字面召回：这里我们使用了 BM25 召回和 TF-IDF 召回两种方法，它们通常用于计算两个文本，或者文本与文档之间的相关性。所以可以用于文本相似度计算和文本检索等应用场景。BM25 召回利用 LangChain 的 BM25Retrievers，TF-IDF 召回利用 LangChain 的TFIDFRetriever。

### 4.3 重排序

Reranker 是信息检索生态系统中的一个重要组成部分，用于评估搜索结果，并进行重新排序，从而提升查询结果相关性。在 RAG 应用中，主要在拿到召回结果后使用 Reranker，能够更有效地确定文档和查询之间的语义相关性，更精细地对结果重排，最终提高搜索质量。将 Reranker 整合到 RAG 应用中可以显著提高生成答案的精确度，因为 Reranker 能够在单路或多路的召回结果中挑选出和问题最接近的文档。此外，扩大检索结果的丰富度（例如多路召回）配合精细化筛选最相关结果（Reranker）还能进一步提升最终结果质量。使用 Reranker 可以排除掉第一层召回中和问题关系不大的内容，将输入给大模型的上下文范围进一步缩小到最相关的一小部分文档中。通过缩短上下文， LLM 能够更“关注”上下文中的所有内容，避免忽略重点内容，还能节省推理成本。

![image](https://github.com/zhangzg1/rag_with_chat/blob/main/images/image_tL0rUhQiZB.png)

上图为增加了 Reranker 的 RAG 应用架构。可以看出，这个检索系统包含两个阶段：

1. 在向量数据库中检索出 Top-K 相关文档，同时也可以配合 Sparse embedding（稀疏向量模型，例如TF-DF）覆盖全文检索能力
2. Reranker 根据这些检索出来的文档与查询的相关性进行打分和重排。重排后挑选最靠前的结果作为 Prompt 中的Context 传入 LLM，最终生成质量更高、相关性更强的答案。

在该项目中。我们分别使用了 bge-reranker 和 bce-reranker-base_v1 模型对检索召回的文档进行重排。

### 4.4 vllm 推理优化

vLLM 是一个基于 Python 的 LLM 推理和服务框架，它的主要优势在于简单易用和性能高效。通过 PagedAttention 技术、连续批处理、CUDA 核心优化以及分布式推理支持，vLLM 能够显著提高 LLM 的推理速度，降低显存占用，更好地满足实际应用需求。vLLM 推理框架使大模型推理速度得到明显提升，推理速度比普通推理有 1 倍的加速。在产品级的部署上，vLLM 既能满足 batch 推理的要求，又能实现高并发下的 continuous batching，在实际产品部署中应用是非常广泛的。

在这个项目中，LLM 分别采用 ChatGLM3-6B，Qwen2-7B-Chat 和 Baichuan2-7B-Chat 作为大模型基座，并且都使用了vllm框架来进行加速推理优化。
