from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings 
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
load_dotenv()
# 初始化嵌入函数
embeddings = DashScopeEmbeddings(model="text-embedding-v3")
# 指定存储路径
persist_directory = "./my_vector_db"

# 加载已持久化的向量数据库
vectorstore = Chroma(collection_name="rag-chroma", embedding_function=embeddings, persist_directory=persist_directory)

retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "Exam-Board-FAQ",
    "在文件中查询关于上海市高考学考、春考、秋考的相关信息。",
)

# 模拟查询
query = "秋考考什么？"
result = retriever_tool.run(query)
print(f"查询结果：{result}")