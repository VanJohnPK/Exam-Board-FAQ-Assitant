from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from typing import Literal
load_dotenv()

# 初始化嵌入函数
embeddings = DashScopeEmbeddings(model="text-embedding-v3")

class GaoKaoInput(BaseModel):
    """Input to the retriever."""
    query: str = Field(description="query to look up in retriever")
    question_class: Literal["秋考", "春考", "艺术类统一考试", "体育类统一考试", "三校生高考","专科自主招生","高中学业水平考试","中职校学业水平考试","专升本考试","普通高校联合招收华侨港澳台考试", "中考中招"] = Field(description="问题的类别")

class ZhongKaoInput(BaseModel):
    """Input to the retriever."""
    query: str = Field(description="query to look up in retriever")

# 假设 vector_db 已经存在，直接加载
vectorstore_gaokao = Chroma(
    collection_name="rag-chroma-gaokao",
    embedding_function=DashScopeEmbeddings(model="text-embedding-v3"),
    persist_directory="./vector_db_gaokao"
)

def search_with_dynamic_filter(query, question_class, search_key="考试类型", k=4):
    # 提取关键词
    if not question_class:
        # 若未找到关键词，不应用筛选条件
        print("无question_class")
        filter_condition = None
    else:
        print("有question_class"+question_class)
        # 构建筛选条件
        filter_condition = {search_key: question_class}

    # 进行相似性搜索并应用筛选条件
    results = vectorstore_gaokao.similarity_search(
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

gaokao_tool = StructuredTool(
    name="GaoKao-FAQ",
    func=search_with_dynamic_filter,
    description="查询高考学考相关问答，包含秋考、春考、艺术类统一考试、体育类统一考试、三校生高考、专科自主招生、高中学业水平考试、中职校学业水平考试、专升本考试、普通高校联合招收华侨港澳台考试。",
    args_schema=GaoKaoInput, 
    response_format="content_and_artifact"
)

from langchain.tools.retriever import create_retriever_tool

vectorstore_zhongkao = Chroma(
    collection_name="rag-chroma-zhongkao",
    embedding_function=DashScopeEmbeddings(model="text-embedding-v3"),
    persist_directory="./vector_db_zhongkao"
)

retriever = vectorstore_zhongkao.as_retriever()

zhongkao_tool = create_retriever_tool(
    retriever,
    "ZhongKao-FAQ",
    "查询中考中招相关问答。",
)