import os
from uuid import uuid4
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

INDEX_NAME = "rag-embeddings-demo"

existing_indexes = [
    index["name"]
    for index in pc.list_indexes()
]

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

index = pc.Index(INDEX_NAME)

pdf_path = "somatosensory.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print(f"Total Characters: {len(text)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print(f"Total Chunks Created: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0])

vectors = []

for chunk in chunks:

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    embedding = response.data[0].embedding

    vectors.append(
        {
            "id": str(uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk
            }
        }
    )

index.upsert(
    vectors=vectors
)

print("Vectors stored successfully")

query = "What is the somatosensory cortex?"

query_embedding = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)

query_vector = query_embedding.data[0].embedding

results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)

print("\nTop Matches:\n")

for match in results["matches"]:
    print(f"Score : {match['score']:.4f}")
    print(f"Chunk : {match['metadata']['text']}")