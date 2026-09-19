"""
SHARED SETUP  —  used by both ingest.py and query.py

Everything both sides must agree on lives here. The big one is the embedding
model: if ingest and query ever embed with different settings, search silently
returns garbage instead of erroring. One definition, imported by both.
"""

import base64
import hashlib
import io
import os
import sys

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIG
# =============================================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "multimodal-rag")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# One TEXT embedding model for everything: chunks, table summaries, captions.
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

# The vision model. Does three jobs: captions images, summarizes tables,
# and writes the final answer while looking at the original figures.
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5.6-terra")

S3_BUCKET = os.getenv("S3_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

MAX_IMAGE_EDGE = 1568   # shrink before upload; nobody needs a 4000px chart


# =============================================================================
# CLIENTS
# =============================================================================

def get_openai():
    from openai import OpenAI
    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY is not set. Add it to .env")
    return OpenAI(api_key=OPENAI_API_KEY)


def get_s3():
    import boto3
    if not S3_BUCKET:
        sys.exit("S3_BUCKET is not set. Add it to .env")
    return boto3.client("s3", region_name=AWS_REGION)


def get_pinecone():
    from pinecone import Pinecone
    if not PINECONE_API_KEY:
        sys.exit("PINECONE_API_KEY is not set. Add it to .env")
    return Pinecone(api_key=PINECONE_API_KEY)


# =============================================================================
# EMBEDDINGS  —  the one function both sides must share
# =============================================================================

def embed(texts):
    """Embed a list of strings. Returns a list of vectors."""
    client = get_openai()
    vectors = []
    for i in range(0, len(texts), 128):                 # batch to stay under limits
        batch = [t if t.strip() else " " for t in texts[i:i + 128]]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            dimensions=EMBED_DIM,   # pinned so it can never drift from the index
        )
        vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return vectors


# =============================================================================
# VISION MODEL
# =============================================================================

def shrink_image(png):
    """Downscale big figures before sending them over the wire."""
    from PIL import Image
    img = Image.open(io.BytesIO(png))
    if max(img.size) <= MAX_IMAGE_EDGE:
        return png
    img.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def image_part(png):
    """An OpenAI image content part, built from raw PNG bytes."""
    b64 = base64.standard_b64encode(shrink_image(png)).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def text_part(text):
    return {"type": "text", "text": text}


def ask_vision(content, system=None, max_out=2000):
    """Call the vision model with a list of content parts (text and/or images).

    Note: these models want `max_completion_tokens`, not `max_tokens`, and the
    budget also covers internal reasoning tokens — so keep it generous or the
    visible answer can come back empty.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    resp = get_openai().chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        max_completion_tokens=max_out,
    )
    return (resp.choices[0].message.content or "").strip()


# =============================================================================
# S3  —  where the ORIGINALS live
# =============================================================================

def s3_key(doc_id, page, kind, index, ext):
    """Key pattern: {doc_id}/page_{n}/{type}_{index}.{ext}"""
    return f"{doc_id}/page_{page}/{kind}_{index}.{ext}"


def s3_upload(s3, key, data, content_type):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)
    return f"s3://{S3_BUCKET}/{key}"


def s3_download(s3, s3_uri):
    bucket, key = _split_uri(s3_uri)
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def s3_presign(s3, s3_uri, expires=3600):
    """A temporary public link, so retrieved figures are clickable."""
    bucket, key = _split_uri(s3_uri)
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires
    )


def _split_uri(s3_uri):
    rest = s3_uri.replace("s3://", "", 1)
    bucket, _, key = rest.partition("/")
    return bucket, key


# =============================================================================
# IDS  —  this is what makes re-ingesting safe
# =============================================================================

def make_doc_id(pdf_path):
    """Hash the file's bytes. The same PDF always gets the same doc_id."""
    with open(pdf_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def make_vector_id(doc_id, kind, index):
    """Same PDF + same element = same vector ID.

    So re-running ingest UPSERTS over the old rows instead of creating a second
    copy of the whole document. Run it ten times, the vector count won't move.
    """
    h = hashlib.sha1(f"{doc_id}|{kind}|{index}".encode()).hexdigest()[:24]
    return f"{kind}-{h}"


def clip(text, limit=2000):
    """Pinecone caps metadata at 40KB per record, so we keep the stored text
    short. Cut on a byte boundary so a UTF-8 character never gets split."""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
