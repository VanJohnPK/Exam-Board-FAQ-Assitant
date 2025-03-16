from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import Tool
load_dotenv()

# 初始化嵌入函数
embeddings = DashScopeEmbeddings(model="text-embedding-v3")

class RetrieverInput(BaseModel):
    """Input to the retriever."""
    query: str = Field(description="query to look up in retriever")


# 假设 vector_db 已经存在，直接加载
vectorstore = Chroma(
    collection_name="rag-chroma",
    embedding_function=DashScopeEmbeddings(model="text-embedding-v3"),
    persist_directory="./vector_db"
)

# 定义关键词到筛选值的映射
keyword_mapping = {
    "高考": "高考学考",
    "春考": "高考学考",
    "秋考": "高考学考",
    "强基": "高考学考",
    "初中": "中考中招",
    "中考": "中考中招",
    "考研": "研考成考",
    "四级": "证书考试",
    "六级": "证书考试",
}

def search_with_dynamic_filter(query, search_key="部分", k=3):
    # 提取关键词
    relevant_keywords = [keyword for keyword in keyword_mapping if keyword in query]

    if not relevant_keywords:
        # 若未找到关键词，不应用筛选条件
        filter_condition = None
    else:
        # 根据映射转换关键词为筛选值
        filter_values = [keyword_mapping[keyword] for keyword in relevant_keywords]
        print(filter_values)
        # 构建筛选条件
        filter_condition = {search_key: {"$in": filter_values}}

    # 进行相似性搜索并应用筛选条件
    results = vectorstore.similarity_search(
        query,
        k=k,
        filter=filter_condition
    )
    # 提取 page_content
    page_contents = [doc.page_content for doc in results]
    # 处理字符串，添加换行符并去掉方括号
    content_str = '\n\n'.join(page_contents)
    # 返回一个二元组
    return (content_str, page_contents)

retriever_tool = Tool(
    name="Exam-Board-FAQ",
    func=search_with_dynamic_filter,
    description="在文件中查询高考学考、中考中招、研考成考、自学考试和证书考试相关问题回答。",
    args_schema=RetrieverInput,
    response_format="content_and_artifact"
)