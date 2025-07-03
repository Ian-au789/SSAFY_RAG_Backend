import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageDocumentParseLoader
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# upstage models
embedding_upstage = UpstageEmbeddings(
    model = "embedding-query",
    api_key = os.environ.get("UPSTAGE_API_KEY")
)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "manual-qa-index"
manual_dir = "path to your user manual"

# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

print("start")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100)

for filename in os.listdir(manual_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(manual_dir, filename)
        print(f"🔍 Processing {filename}")

        loader = UpstageDocumentParseLoader(
            pdf_path,
            output_format='html',
            coordinates=False
        )
        docs = loader.load()

        splits = text_splitter.split_documents(docs)

        PineconeVectorStore.from_documents(
            splits,
            embedding_upstage,
            index_name=index_name
        )

        print(f"✅ Finished embedding {filename}")

PineconeVectorStore.from_documents(
    splits, embedding_upstage, index_name=index_name
)
print("end")
