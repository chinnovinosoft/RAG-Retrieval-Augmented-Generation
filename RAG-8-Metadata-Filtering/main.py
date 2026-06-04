import os

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

load_dotenv()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index = pc.Index("company-rag")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

query_embedding = embeddings.embed_query(
    "How can diseases be prevented?"
)

results = index.search(
    namespace="company_docs",
    query={
        "vector": query_embedding,
        "top_k": 5
    },
    fields=[
        "department",
        "year",
        "chunk_text"
    ]
)

print(results)
print("\n\n=============================\n")
results = index.search(
    namespace="company_docs",
    query={
        "vector": query_embedding,
        "top_k": 5,
        "filter": {
            "department": {
                "$eq": "Medical"
            }
        }
    },
    fields=[
        "department",
        "region",
        "chunk_text"
    ]
)

print(results)

# results = index.search(
#     namespace="company_docs",
#     query={
#         "vector": query_embedding,
#         "top_k": 5,
#         "filter": {
#             "department": {
#                 "$in": [
#                     "HR",
#                     "Medical"
#                 ]
#             }
#         }
#     },
#     fields=[
#         "department",
#         "year",
#         "chunk_text"
#     ]
# )

# print(results)

# results = index.search(
#     namespace="company_docs",
#     query={
#         "vector": query_embedding,
#         "top_k": 5,
#         "filter": {
#             "$and": [
#                 {
#                     "department": "Medical"
#                 },
#                 {
#                     "region": "India"
#                 }
#             ]
#         }
#     },
#     fields=[
#         "department",
#         "region",
#         "year",
#         "chunk_text"
#     ]
# )

# print(results)
