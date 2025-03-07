from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import DashScopeEmbeddings
import re
from dotenv import load_dotenv
load_dotenv()

# 本地文件路径列表，你需要将这些路径替换为你实际的文件路径
file_paths = [
    "./docs/book0.txt",
    "./docs/book1.txt",
    "./docs/book2.txt",
    "./docs/book3.txt",
    "./docs/book4.txt",
]

# 处理每个文件并创建独立的向量数据库
for index, file_path in enumerate(file_paths):
    # 加载单个文件
    loader = TextLoader(file_path, encoding="utf-8")
    all_text = loader.load()[0].page_content

    # 使用正则表达式按行首数字加. 分割文本
    pattern = re.compile(r'^\d+\.', re.MULTILINE)
    split_indices = [m.start() for m in pattern.finditer(all_text)]
    qa_pairs = []
    for i in range(len(split_indices)):
        start = split_indices[i]
        end = split_indices[i + 1] if i + 1 < len(split_indices) else len(all_text)
        qa_pairs.append(all_text[start:end].strip())

    docs = []
    for pair in qa_pairs:
        # 去掉开头的数字和.
        clean_pair = re.sub(r'^\d+\.\s*', '', pair)
        doc = Document(page_content=clean_pair)
        docs.append(doc)

    doc_splits = docs  # 不进行额外分割，直接使用完整问答对

    # 初始化向量数据库
    embeddings = DashScopeEmbeddings(model="text-embedding-v3")
    # 指定存储路径和集合名称
    persist_directory = f"./my_vector_db_{index}"
    collection_name = f"rag-chroma_{index}"
    vectorstore = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_directory)

    # 分批添加文档到向量数据库
    batch_size = 64
    for i in range(0, len(doc_splits), batch_size):
        print(f"Processing batch {i} for file {file_path}")
        batch_docs = doc_splits[i:i + batch_size]
        vectorstore.add_documents(batch_docs)

    print(vectorstore)
    vectorstore.persist()  # 确保数据持久化
    print(f"向量数据库 {collection_name} 构建完成并已持久化。")

print("所有向量数据库构建完成并已持久化。")