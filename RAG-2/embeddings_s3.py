import os
from uuid import uuid4

from dotenv import load_dotenv

from openai import OpenAI

import boto3

from pypdf import PdfReader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# CONFIG
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VECTOR_BUCKET_NAME = "my-rag-vector-bucket"
INDEX_NAME = "somatosensory-index"
AWS_REGION = "us-east-1"

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

s3vectors = boto3.client(
    "s3vectors",
    region_name=AWS_REGION
)

pdf_path = "somatosensory.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print(f"Chunks Created : {len(chunks)}")

vectors = []

for chunk in chunks:

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    embedding = response.data[0].embedding

    vectors.append(
        {
            "key": str(uuid4()),
            "data": {
                "float32": embedding
            },
            "metadata": {
                "text": chunk
            }
        }
    )

# s3vectors.create_index(
#     vectorBucketName="my-rag-vector-bucket",
#     indexName="somatosensory-index",
#     dataType="float32",
#     dimension=1536,
#     distanceMetric="cosine"
# )

s3vectors.put_vectors(
    vectorBucketName=VECTOR_BUCKET_NAME,
    vectors=vectors,
    indexName=INDEX_NAME,
)

print("Vectors Stored Successfully")

query = "What is the somatosensory cortex?"

query_embedding = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)

query_vector = query_embedding.data[0].embedding

results = s3vectors.query_vectors(
    vectorBucketName=VECTOR_BUCKET_NAME,
    indexName=INDEX_NAME,
    queryVector={
        "float32": query_vector
    },
    topK=3,
    returnMetadata=True
)
print("Query Results:\n",results)


for vector in results["vectors"]:
    print("=" * 80)
    print(vector["metadata"]["text"])