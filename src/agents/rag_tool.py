from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

# 初始化嵌入函数
embeddings = OpenAIEmbeddings(api_key="a89a3ffd346549eeb6a2914d8bcba924.vGehkWyAxdym9sHz", base_url="https://open.bigmodel.cn/api/paas/v4", model="embedding-3")
# 指定存储路径
persist_directory = "./my_vector_db"

# 加载已持久化的向量数据库
vectorstore = Chroma(collection_name="rag-chroma", embedding_function=embeddings, persist_directory=persist_directory)

retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "Exam-Board-FAQ",
    "Search for and obtain detailed information about the registration, exam content, scoring, and admission of the autumn and spring college entrance examinations in local documents.",
)

# # 模拟查询
# query = "秋考考什么？"
# result = retriever_tool.run(query)
# print(f"查询结果：{result}")