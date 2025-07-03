import os
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

load_dotenv()

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# LangChain + Pinecone Setup
embedding_model = UpstageEmbeddings(model="embedding-query")
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "manual-qa-index"
vectorstore = PineconeVectorStore(index=pinecone.Index(index_name), embedding=embedding_model)
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={"k": 2})

# OpenAI API Client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==== Request/Response Models ====

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # entire conversation so far


# ==== Main Chat Endpoint ====

@app.post("/chat")
async def chat_with_rag(req: ChatRequest):
    # 최신 사용자 질문
    user_question = req.messages[-1].content

    # Pinecone에서 context 검색
    docs = retriever.get_relevant_documents(user_question)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # system 프롬프트 포함 전체 메시지 구성
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})
    messages.append({
        "role": "system",
        "content": f"참고 문서 내용:\n{context_text}"
    })

    # OpenAI Chat API 호출
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )

    return {"reply": response.choices[0].message.content}

@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
