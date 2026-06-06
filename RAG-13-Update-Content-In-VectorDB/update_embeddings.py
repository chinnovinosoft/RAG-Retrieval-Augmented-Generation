import re
from pathlib import Path

from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DOC_PATH = "Employee Policy Handbook 2025.docx"

MAX_WORDS_PER_PARAGRAPH = 1200

# Used only when a paragraph exceeds MAX_WORDS_PER_PARAGRAPH
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

doc = Document(DOC_PATH)

paragraphs = [
    p.text.strip()
    for p in doc.paragraphs
    if p.text.strip()
]

sections = {}

current_section = None

for para in paragraphs:

    if re.match(r"Section\s+\d+:", para):

        current_section = para

        sections[current_section] = []

    elif current_section:

        sections[current_section].append(para)

document_name = (
    Path(DOC_PATH)
    .stem
    .lower()
    .replace(" ", "_")
)

chunks = []

for section_title, section_paragraphs in sections.items():

    section_name = (
        section_title
        .split(":")[1]
        .strip()
        .lower()
        .replace(" ", "_")
    )

    chunk_counter = 1

    for paragraph in section_paragraphs:

        word_count = len(paragraph.split())

        # SMALL PARAGRAPH
        if word_count <= MAX_WORDS_PER_PARAGRAPH:

            chunk_id = (
                f"{document_name}_"
                f"{section_name}_"
                f"chunk_{chunk_counter}"
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "section": section_name,
                    "text": paragraph
                }
            )

            chunk_counter += 1

        # LARGE PARAGRAPH
        else:

            split_chunks = splitter.split_text(paragraph)

            for split_text in split_chunks:

                chunk_id = (
                    f"{document_name}_"
                    f"{section_name}_"
                    f"chunk_{chunk_counter}"
                )

                chunks.append(
                    {
                        "id": chunk_id,
                        "section": section_name,
                        "text": split_text
                    }
                )

                chunk_counter += 1

#Results 
print(f"\nTotal Chunks Created: {len(chunks)}\n")

for chunk in chunks:

    print("=" * 100)

    print("ID:")
    print(chunk["id"])

    print("\nSECTION:")
    print(chunk["section"])

    print("\nTEXT:")
    print(chunk["text"][:200])

    print("\n")


embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index = os.getenv("PINECONE_INDEX_NAME")

existing_indexes = [i["name"] for i in pc.list_indexes()]

if index not in existing_indexes:
    pc.create_index(
        name=index,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

vectors = []

for chunk in chunks:

    embedding = embeddings_model.embed_query(
        chunk["text"]
    )

    vectors.append(
        {
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
                "document_name": document_name,
                "section": chunk["section"],
                "text": chunk["text"]
            }
        }
    )

index = pc.Index(
    os.getenv("PINECONE_INDEX_NAME")
)

# UPSERT VECTORS TO PINECONE
index.upsert(
    vectors=vectors
)

print(
    f"Successfully inserted {len(vectors)} vectors."
)