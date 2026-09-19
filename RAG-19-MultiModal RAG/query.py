"""
QUERY  —  question in, cited answer out.   (the READ side)

    question → embed → Pinecone top 20 → rerank top 5
             → fetch the ORIGINALS from S3
             → one multimodal message → vision model → answer + [file, page]

This is where the pattern pays off. Pinecone gives us back short summaries and
captions - those were only ever search keys. Before we ask anything, we go to
S3 and swap each one for the real thing: the full Markdown table, the actual
PNG of the figure. The model answers while LOOKING at the original.

Run:
    python query.py "How much warming is projected under SSP5-8.5?"
    python query.py "..." --type image      # only search figures
    python query.py "..." --no-rerank
"""

import os
import sys

from common import (
    NAMESPACE, PINECONE_INDEX,
    ask_vision, clip, embed, get_pinecone, get_s3,
    image_part, s3_download, s3_presign, text_part,
)

TOP_K = 20        # how many we pull out of Pinecone
RERANK_TO = 5     # how many survive the reranker
MAX_IMAGES = 5    # cap on images sent in one request
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"


# =============================================================================
# STEP 1  —  Search Pinecone
# =============================================================================

def search(question, kind=None, doc_id=None, use_reranker=USE_RERANKER):
    pc = get_pinecone()
    index = pc.Index(PINECONE_INDEX)

    # Same embedding model as ingest. This is not optional - a different model
    # here means the vectors live in a different space and results are noise.
    vector = embed([question])[0]

    where = {}
    if kind:
        where["type"] = kind          # e.g. only search figures
    if doc_id:
        where["doc_id"] = doc_id

    result = index.query(
        vector=vector,
        top_k=TOP_K,
        include_metadata=True,
        namespace=NAMESPACE,
        filter=where or None,
    )
    hits = [{"id": m["id"], "score": m["score"], **m["metadata"]}
            for m in result["matches"]]

    if not use_reranker or not hits:
        return hits[:RERANK_TO]

    # -------------------------------------------------------------------------
    # STEP 2  —  Rerank
    # -------------------------------------------------------------------------
    # The embedding search compared the question and each document separately.
    # A reranker reads them TOGETHER, which is slower but much more accurate -
    # so we let vector search cast a wide net (20) and rerank picks the 5.
    ranked = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=question,
        documents=[{"id": h["id"], "text": clip(h["text"], 1000)} for h in hits],
        rank_fields=["text"],
        top_n=RERANK_TO,
        return_documents=False,
    )

    out = []
    for r in ranked.data:
        hit = hits[r["index"]]        # map back to the original hit
        hit["score"] = r["score"]
        out.append(hit)
    return out


# =============================================================================
# STEP 3  —  Swap summaries for ORIGINALS
# =============================================================================

def build_context(hits):
    """Turn hits into content parts for the model.

    text  -> the chunk, straight from metadata
    table -> download the FULL Markdown from S3  (not the summary we embedded)
    image -> download the ACTUAL PNG from S3     (not the caption we embedded)
    """
    s3 = get_s3()
    parts = [text_part("Here is the retrieved context.\n")]
    images_used = 0

    for hit in hits:
        label = f"[{hit['source_file']}, page {hit['page']}, {hit['type']}]"

        if hit["type"] == "text":
            # A text chunk is already the original - it came straight out of
            # Pinecone metadata whole, so there is nothing to fetch from S3.
            parts.append(text_part(f"{label}\n{hit['text']}\n"))

        elif hit["type"] == "table":
            markdown = s3_download(s3, hit["s3_uri"]).decode("utf-8")
            parts.append(text_part(f"{label}\n{markdown}\n"))

        elif hit["type"] == "image" and images_used < MAX_IMAGES:
            png = s3_download(s3, hit["s3_uri"])
            parts.append(image_part(png))                        # the image ...
            parts.append(text_part(f"{label} (figure above)\n"))  # ... then its label
            images_used += 1

    return parts


# =============================================================================
# STEP 4  —  Answer
# =============================================================================

SYSTEM = """You answer questions using only the provided document context, which \
may include text passages, full tables in Markdown, and images of figures.

Rules:
1. Use ONLY the provided context. Never use outside knowledge.
2. Cite every factual claim inline as [file, page], using the labels given with each source.
3. Read values directly off the tables and figures shown - those are the originals.
4. If the context does not contain the answer, say exactly that and stop. Do not guess.
5. Be concise. Prefer specific numbers with their units."""


def ask(question, kind=None, doc_id=None, use_reranker=USE_RERANKER):
    hits = search(question, kind, doc_id, use_reranker)
    if not hits:
        print("Nothing found in the index. Has anything been ingested?")
        return

    parts = build_context(hits)
    parts.append(text_part(f"\nQuestion: {question}"))

    n_img = sum(1 for p in parts if p["type"] == "image_url")
    print(f"Sending {len(hits)} sources ({n_img} as real images) to the model ...\n")

    answer = ask_vision(parts, system=SYSTEM, max_out=8000)

    print("=" * 72)
    print(answer)
    print("=" * 72)

    # STEP 5 - show where it came from. Images get a temporary clickable link.
    s3 = get_s3()
    print("\nSOURCES")
    for hit in hits:
        print(f"  [{hit['type']:5}] {hit['source_file']}  p.{hit['page']}  "
              f"score={hit['score']:.3f}")
        if hit["type"] == "image":
            print(f"          {s3_presign(s3, hit['s3_uri'])[:110]}...")
        elif hit["s3_uri"]:
            print(f"          {hit['s3_uri']}")
        else:
            print(f"          section: {hit.get('section') or '(none)'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python query.py "your question" [--type text|table|image] [--no-rerank]')

    question = sys.argv[1]
    kind = None
    if "--type" in sys.argv:
        kind = sys.argv[sys.argv.index("--type") + 1]

    ask(question, kind=kind, use_reranker=USE_RERANKER and "--no-rerank" not in sys.argv)
