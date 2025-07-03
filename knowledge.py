import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 환경변수로부터 ID 가져오기
FILE_ID = os.getenv("FILE_ID")
ASSISTANT_ID = os.getenv("ASSISTANT_API")

# ✅ 1. Vector Store 생성
vector_store = openai.vector_stores.create(
    name="My_QA_VectorStore"
)

# ✅ 2. Vector Store에 파일 추가
openai.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=FILE_ID
)

# ✅ 3. Assistant에 연결 (beta 경로 사용)
assistant = openai.beta.assistants.update(
    assistant_id=ASSISTANT_ID,
    tools=[{"type": "file_search"}],
    tool_resources={
        "file_search": {
            "vector_store_ids": [vector_store.id]
        }
    }
)

print("✅ Assistant 연결 완료:", assistant.id)
