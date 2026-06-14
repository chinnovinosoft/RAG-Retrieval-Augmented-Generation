"""
PDF → Pinecone Ingestion
========================
Reads iso27001.pdf, splits into chunks, embeds with OpenAI,
and upserts into the Pinecone index defined in .env.

Run:
    python ingest.py
    python ingest.py --pdf iso27001.pdf --chunk-size 1000 --overlap 200
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config from .env ──────────────────────────────────────────────────────────

PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "default")
OPENAI_API_KEY      = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Default PDF path ──────────────────────────────────────────────────────────

DEFAULT_PDF = Path(__file__).parent / "iso27001.pdf"


def ingest(pdf_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
    import pypdf
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    # ── 1. Load PDF ───────────────────────────────────────────────────────────
    logger.info("Loading PDF: %s", pdf_path)
    reader = pypdf.PdfReader(str(pdf_path))
    pages = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"page": i, "source": pdf_path.name},
        )
        for i, page in enumerate(reader.pages)
        if (page.extract_text() or "").strip()
    ]
    logger.info("Loaded %d pages", len(pages))

    # ── 2. Split into chunks ──────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    logger.info("Split into %d chunks (size=%d overlap=%d)", len(chunks), chunk_size, chunk_overlap)

    # Enrich metadata on every chunk
    pdf_name = pdf_path.stem
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "source": pdf_name,
            "chunk_index": i,
            "total_chunks": len(chunks),
        })
        # Move page_content into metadata["text"] so the MCP server can find it
        chunk.metadata["text"] = chunk.page_content

    # ── 3. Create Pinecone index if it doesn't exist ──────────────────────────
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        logger.info("Index '%s' not found — creating it …", PINECONE_INDEX_NAME)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=int(os.getenv("PINECONE_DIMENSION", "1536")),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until the index is ready
        import time
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            logger.info("Waiting for index to be ready …")
            time.sleep(2)
        logger.info("Index '%s' created and ready", PINECONE_INDEX_NAME)
    else:
        logger.info("Index '%s' already exists — skipping creation", PINECONE_INDEX_NAME)

    # ── 4. Embed + upsert ─────────────────────────────────────────────────────
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    logger.info("Upserting %d chunks into Pinecone (namespace='%s') …", len(chunks), PINECONE_NAMESPACE)
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=PINECONE_NAMESPACE,
        pinecone_api_key=PINECONE_API_KEY,
    )

    logger.info("Done — %d chunks written to Pinecone index '%s'", len(chunks), PINECONE_INDEX_NAME)
    return vector_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into Pinecone")
    parser.add_argument(
        "--pdf", type=Path, default=DEFAULT_PDF,
        help=f"Path to PDF file (default: {DEFAULT_PDF})",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap in characters")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    ingest(pdf_path=args.pdf, chunk_size=args.chunk_size, chunk_overlap=args.overlap)
