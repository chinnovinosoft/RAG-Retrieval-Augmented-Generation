from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os

load_dotenv()
PDF_PATH = "company_policy.pdf"

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

existing_indexes = [i["name"] for i in pc.list_indexes()]

if "company-policy-index" not in existing_indexes:
    pc.create_index(
        name="company-policy-index",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="company-policy-index"
)

print(f"Ingested {len(chunks)} chunks")