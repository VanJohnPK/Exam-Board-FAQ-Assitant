from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv

load_dotenv()

# 初始化嵌入函数
embeddings = DashScopeEmbeddings(model="text-embedding-v3")

# 存储 5 个向量数据库的工具列表
retriever_tools = []

# 定义工具的名字和描述
tool_names = ["HighSchool-Entrance-FAQ", "MiddleSchool-Recruitment-FAQ", 
              "PostGraduate-Exams-FAQ", "SelfStudy-Exams-FAQ", 
              "Certification-Exams-FAQ"]
tool_descriptions = ["在文件中查询高考学考（秋考、春考、艺考、体考、三校生高考、专科自招、学业水平、专升本、华侨港澳台考试）的相关信息。",
                     "在文件中查询中考中招的相关信息。",
                     "在文件中查询研考成考（研究生招生考试即考研、成人高考、同等学力申硕外语及学科综合全国统考）的相关信息。",
                     "在文件中查询自学考试的相关信息。",
                     "在文件中查询证书考试（英语四六级CET、全国中小学教资笔试、全国英语等级考PETS、全国计算机等级考试NCRE、上海市高校信息技术水平考试）的相关信息。"]

# 循环处理 5 个向量数据库
for index in range(5):
    # 指定存储路径和集合名称
    persist_directory = f"./my_vector_db_{index}"
    collection_name = f"rag-chroma_{index}"

    # 加载已持久化的向量数据库
    vectorstore = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_directory)

    # 将向量数据库转换为检索器
    retriever = vectorstore.as_retriever()

    # 创建检索器工具
    retriever_tool = create_retriever_tool(
        retriever,
        tool_names[index],
        tool_descriptions[index],
    )

    # 将工具添加到列表中
    retriever_tools.append(retriever_tool)

# print(retriever_tool)
# # 模拟查询（这里以第一个工具为例）
# query = "秋考考什么？"
# result = retriever_tools[0].run(query)
# print(f"查询结果：{result}")