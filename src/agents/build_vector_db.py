from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

# 本地文件路径列表，你需要将这些路径替换为你实际的文件路径
file_paths = [
    "./docs/book.txt",
]

# 加载本地文件
all_text = ""
for file_path in file_paths:
    loader = TextLoader(file_path, encoding="utf-8")
    all_text += loader.load()[0].page_content

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
    doc = Document(page_content=pair)
    docs.append(doc)

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=200, chunk_overlap=100
# )
# doc_splits = text_splitter.split_documents(docs)
doc_splits = docs  # 不进行额外分割，直接使用完整问答对
# 初始化向量数据库 # 白嫖的key，不怕泄露
embeddings = OpenAIEmbeddings(api_key="a89a3ffd346549eeb6a2914d8bcba924.vGehkWyAxdym9sHz", base_url="https://open.bigmodel.cn/api/paas/v4", model="embedding-3")
# 指定存储路径
persist_directory = "./my_vector_db"
vectorstore = Chroma(collection_name="rag-chroma", embedding_function=embeddings, persist_directory=persist_directory)

# 分批添加文档到向量数据库
batch_size = 64
for i in range(0, len(doc_splits), batch_size):
    print(i)
    batch_docs = doc_splits[i:i + batch_size]
    vectorstore.add_documents(batch_docs)

print(vectorstore)
vectorstore.persist()  # 确保数据持久化
print("向量数据库构建完成并已持久化。")