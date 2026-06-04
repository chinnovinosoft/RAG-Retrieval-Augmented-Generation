import os

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [i["name"] for i in pc.list_indexes()]

if "company-rag" not in existing_indexes:

    pc.create_index(
        name="company-rag",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

files = [
    {
        "path": "data/hr_policy.txt",
        "metadata": {
            "department": "HR",
            "document_type": "policy",
            "year": 2025,
            "region": "India"
        }
    },
    {
        "path": "data/finance_report.txt",
        "metadata": {
            "department": "Finance",
            "document_type": "report",
            "year": 2024,
            "region": "USA"
        }
    },
    {
        "path": "data/medical_guide.txt",
        "metadata": {
            "department": "Medical",
            "document_type": "guide",
            "year": 2026,
            "region": "India"
        }
    }
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
documents = []

for file in files:

    with open(file["path"], "r") as f:
        text = f.read()

    chunks = splitter.split_text(text)

    for idx, chunk in enumerate(chunks):

        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    **file["metadata"],
                    "source": file["path"],
                    "chunk_id": idx
                }
            )
        )

vectors = []

for i, doc in enumerate(documents):

    embedding = embeddings.embed_query(
        doc.page_content
    )

    vectors.append(
        {
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {
                **doc.metadata,
                "chunk_text": doc.page_content
            }
        }
    )
index = pc.Index("company-rag")
index.upsert(
    vectors=vectors,
    namespace="company_docs"
)

print("Data Ingested")

