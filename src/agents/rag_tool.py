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
vectorstore_gaokao = Chroma(
    collection_name="rag-chroma-gaokao",
    embedding_function=DashScopeEmbeddings(model="text-embedding-v3"),
    persist_directory="./vector_db_gaokao"
)

gaokao_keywords = ["秋考", "春考", "艺术类统一考试", "体育类统一考试", "三校生高考", "专科自主招生",
                   "高中学业水平考试", "中职校学业水平考试", "其他考试—专升本考试",
                   "其他考试—普通高校联合招收华侨港澳台考试"]
# 定义关键词到筛选值的映射
keyword_mapping = {
    "高考": ["秋考"],
    "强基": ["秋考"],
    "春季高考": ["春考"],
    "艺考": ["艺术类统一考试"],
    "艺术高考": ["艺术类统一考试"],
    "艺术类": ["艺术类统一考试"],
    "艺术生": ["艺术类统一考试"],
    "体育类": ["体育类统一考试"],
    "体育生": ["体育类统一考试"],
    "体育高考": ["体育类统一考试"],
    "体考": ["体育类统一考试"],
    "体育单招": ["体育类统一考试"],
    "体育单独招生": ["体育类统一考试"],
    "三校": ["三校生高考"],
    "专科自招": ["专科自主招生"],
    "合格考": ["高中学业水平考试"],
    "中职合格考": ["中职校学业水平考试"],
    "专升本": ["其他考试—专升本考试"],
    "华侨": ["其他考试—普通高校联合招收华侨港澳台考试"],
    "港澳台": ["其他考试—普通高校联合招收华侨港澳台考试"],
}

# 将 gaokao_keywords 中的关键词添加到映射，自身映射到自身
for keyword in gaokao_keywords:
    keyword_mapping[keyword] = keyword

def search_with_dynamic_filter(query, search_key="考试类型", k=5):
    # 提取关键词
    relevant_keywords =[keyword for keyword in keyword_mapping if keyword in query]

    if not relevant_keywords:
        # 若未找到关键词，不应用筛选条件
        filter_condition = None
    else:
       # 根据映射转换关键词为筛选值
        filter_values = []
        for keyword in relevant_keywords:
            for value in keyword_mapping[keyword]:
                if value not in filter_values:
                    filter_values.append(value)
        print(filter_values)
        # 构建筛选条件
        filter_condition = {search_key: {"$in": filter_values}}

    # 进行相似性搜索并应用筛选条件
    results = vectorstore_gaokao.similarity_search(
        query,
        k=k,
        filter=filter_condition
    )
    # 提取 page_content，并添加 metadata 信息
    page_contents = []
    for doc in results:
        part = doc.metadata.get('部分', '')
        exam_type = doc.metadata.get('考试类型', '')
        question_type = doc.metadata.get('问题类型', '')
        new_content = f"【{part}】【{exam_type}】【{question_type}】{doc.page_content}"
        page_contents.append(new_content)
    # 处理字符串，添加换行符并去掉方括号
    content_str = '\n\n'.join(page_contents)
    # 返回一个二元组
    return (content_str, page_contents)

gaokao_tool = Tool(
    name="GaoKao-FAQ",
    func=search_with_dynamic_filter,
    description="查询高考学考相关问答，包含秋考、春考、艺术类统一考试、体育类统一考试、三校生高考、专科自主招生、高中学业水平考试、中职校学业水平考试、专升本考试、普通高校联合招收华侨港澳台考试。",
    args_schema=RetrieverInput, 
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