import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1️⃣ 교과서 텍스트 파일 로드
textbook_path = "univ_physics_textbook.txt"  # 교과서 텍스트 파일 경로
loader = TextLoader(textbook_path)
documents = loader.load()

# 2️⃣ 텍스트를 문단 단위로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3️⃣ 문단 벡터화 (임베딩 모델 사용)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# 4️⃣ FAISS 벡터 저장소 생성
vectorstore = FAISS.from_documents(texts, embeddings)

# 5️⃣ 벡터 저장소 저장
faiss_db_path = "physics_faiss_index"
vectorstore.save_local(faiss_db_path)

print(f"✅ 벡터 DB 저장 완료! 경로: {faiss_db_path}")
