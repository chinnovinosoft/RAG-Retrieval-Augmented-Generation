import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_FILE = "labour.pdf"
INDEX_NAME = "labour-index"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

existing_indexes = [i["name"] for i in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:

    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

docs = PyPDFLoader(PDF_FILE).load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = splitter.split_documents(docs)

PineconeVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("Upload completed")