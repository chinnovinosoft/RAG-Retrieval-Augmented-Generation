"""
INGEST  —  PDF in, vectors out.   (the WRITE side)

    PDF ─ Docling ─┬─ text chunks ─────────────┐
                   ├─ tables  → summarize ─────┼─ embed → Pinecone
                   └─ images  → caption ───────┘    originals → S3

The key idea: what we EMBED and what we later SHOW the model are different
things. We embed a short summary so search is cheap. We keep the original
table and the original PNG in S3, because that is what actually answers
the question later.

Run:
    python ingest.py IPCC_AR6_SYR_SPM-2.pdf
    python ingest.py IPCC_AR6_SYR_SPM-2.pdf --inspect    # parse only, no API calls
"""

import hashlib
import io
import os
import sys
import time

from common import (
    EMBED_DIM, NAMESPACE, PINECONE_INDEX,
    ask_vision, clip, embed, get_pinecone, get_s3,
    image_part, make_doc_id, make_vector_id, s3_key, s3_upload, text_part,
)

IMAGES_SCALE = 2.0    # 1.0 is ~72 DPI. 2.0 gives figures we can actually read.
MIN_IMAGE_PX = 100    # smaller than this is an icon or a bullet, not a figure
CHUNK_TOKENS = 512    # keeps each chunk well under Pinecone's metadata cap

# Pinecone allows 40KB of metadata per record. A 512-token chunk is under 3KB,
# so 8KB is comfortable headroom AND leaves a chunk whole. Set this too low and
# text gets silently cut off mid-sentence before the model ever sees it.
METADATA_TEXT_LIMIT = 8000


# =============================================================================
# STEP 1  —  Parse the PDF with Docling
# =============================================================================

def parse_pdf(pdf_path):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = True                            # read text baked into images
    options.do_table_structure = True                # real rows/cols, not a text blob
    options.table_structure_options.do_cell_matching = True
    options.generate_picture_images = True           # without this, figures have no pixels
    options.images_scale = IMAGES_SCALE

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    print(f"Parsing {pdf_path} ...  (first run downloads Docling models, be patient)")
    return converter.convert(pdf_path).document


# =============================================================================
# STEP 2  —  TEXT  →  section-aware chunks
# =============================================================================

def extract_text(doc, log_path="chunks_sample.txt"):
    """HybridChunker splits on document structure first, then packs to a token
    budget. We hand it the SAME tokenizer our embedding model uses, so '512
    tokens' means the same thing on both sides.

    Writes every chunk to `log_path` so you can actually SEE what it did -
    where it cut, what it merged, and which chunks got dropped.
    """
    import tiktoken
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

    from common import EMBED_MODEL

    encoding = tiktoken.encoding_for_model(EMBED_MODEL)
    chunker = HybridChunker(
        tokenizer=OpenAITokenizer(tokenizer=encoding, max_tokens=CHUNK_TOKENS),
        merge_peers=True,
    )

    out = []
    log = [
        "HYBRID CHUNKER OUTPUT",
        f"embedding model : {EMBED_MODEL}",
        f"max tokens/chunk: {CHUNK_TOKENS}",
        "",
        "'source items' shows which document elements went INTO each chunk.",
        "More than one means merge_peers glued small neighbours together.",
        "=" * 78,
        "",
    ]

    for chunk in chunker.chunk(dl_doc=doc):
        items = chunk.meta.doc_items
        pages = sorted({p.page_no for it in items for p in it.prov})
        sources = ", ".join(type(it).__name__ for it in items) or "none"
        text = chunker.contextualize(chunk=chunk)
        tokens = len(encoding.encode(text))

        def entry(tag):
            return (
                f"--- {tag} | page {pages or '?'} | {tokens} tokens | "
                f"source items: {sources}\n"
                f"    headings: {' > '.join(chunk.meta.headings or []) or '(none)'}\n"
                f"{text}\n"
            )

        # HybridChunker also emits TABLES, flattened into triplet text like
        # "SSP5-8.5, Warming = 4.4C". We do NOT want those here: STEP 3 already
        # handles tables properly, keeping the real Markdown in S3. If we let
        # this copy through, the same table is indexed twice - and the copy is
        # tagged type="text", so it has no link back to the original. Retrieval
        # would hand the model a flattened blob instead of the real table.
        # (Images never appear here at all - a picture contributes at most its
        # caption text, never pixels. That's what STEP 4 is for.)
        # We test self_ref ("#/tables/0") rather than isinstance(it, TableItem).
        # Pydantic hands most of these back as the BASE DocItem class - the
        # subclass is lost - so an isinstance check works today only by luck and
        # would silently become a no-op if that ever changed. The ref string is
        # part of the document format, so it can't degrade the same way.
        if items and all(it.self_ref.startswith("#/tables/") for it in items):
            log.append(entry("SKIPPED (table -> handled by STEP 3)"))
            continue

        log.append(entry(f"CHUNK {len(out)}"))
        out.append({
            "kind": "text",
            "index": len(out),                # index AFTER filtering, so IDs stay stable
            "page": pages[0] if pages else 0,
            "section": " > ".join(chunk.meta.headings or []),
            "search_text": text,              # for text we embed the text itself
            "payload": text.encode("utf-8"),
            "ext": "md",
        })

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print(f"  wrote {len(out)} chunks to {log_path}  (open it to see the splits)")

    return out


# =============================================================================
# STEP 3 & 4  —  TABLES and IMAGES
# =============================================================================
# One walk in reading order gets both. As we go we remember the current heading
# and the last paragraph, so when we hit a table or figure we already know what
# section it belongs to and what text sits next to it. That context is what
# makes the captions in STEP 5 actually useful.

def extract_tables_and_images(doc):
    from docling_core.types.doc import PictureItem, TableItem, TextItem

    tables, images = [], []
    section = ""
    last_paragraph = ""
    seen = set()
    skipped_tiny = skipped_dupe = 0

    for item, _level in doc.iterate_items():

        # --- track where we are in the document ---
        # Match the label EXACTLY. "page_header" is the running head repeated
        # on every page - not a section heading. Matching loosely tags every
        # figure with the page banner instead of its real section.
        if isinstance(item, TextItem):
            label = str(item.label)
            if label in ("section_header", "title"):
                section = item.text.strip()
            elif label in ("text", "paragraph") and len(item.text.strip()) > 40:
                last_paragraph = item.text.strip()
            continue

        page = item.prov[0].page_no if item.prov else 0

        # --- STEP 3: tables ---
        if isinstance(item, TableItem):
            markdown = item.export_to_markdown(doc=doc)
            if not markdown.strip():
                continue
            tables.append({
                "kind": "table",
                "index": len(tables),
                "page": page,
                "section": section,
                "search_text": "",                     # filled in by STEP 5
                "payload": markdown.encode("utf-8"),   # the ORIGINAL, goes to S3
                "ext": "md",
                "html": item.export_to_html(doc=doc),
                "caption": item.caption_text(doc),
            })

        # --- STEP 4: images ---
        elif isinstance(item, PictureItem):
            pil = item.get_image(doc)
            if pil is None:
                continue

            # Drop icons, bullets, decorative rules.
            if pil.width < MIN_IMAGE_PX or pil.height < MIN_IMAGE_PX:
                skipped_tiny += 1
                continue

            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            png = buf.getvalue()

            # Drop repeats: the same logo on every page hashes identically.
            digest = hashlib.sha256(png).hexdigest()
            if digest in seen:
                skipped_dupe += 1
                continue
            seen.add(digest)

            images.append({
                "kind": "image",
                "index": len(images),
                "page": page,
                "section": section,
                "search_text": "",            # filled in by STEP 5
                "payload": png,               # the ORIGINAL, goes to S3
                "ext": "png",
                "caption": item.caption_text(doc),
                "nearby": last_paragraph,
            })

    print(f"  skipped {skipped_tiny} tiny images, {skipped_dupe} duplicates")
    return tables, images


# =============================================================================
# STEP 5  —  Make tables and images SEARCHABLE
# =============================================================================
# A PNG has no text, so a text embedding model can't index it. We ask the vision
# model to describe it, and embed that description. It is a search key, nothing
# more - the original is what gets used to answer.

TABLE_PROMPT = (
    "In 2-3 sentences, describe what this table measures, its units, and what "
    "its rows and columns range over. Name the specific metrics and scenarios, "
    "because your text is what a semantic search matches against. No preamble."
)

IMAGE_PROMPT = (
    "In 2-4 sentences, describe this figure: chart type, what is on each axis, "
    "the variables or scenarios plotted, and the overall trend. Name specifics, "
    "because your text is what a semantic search matches against. No preamble."
)


def summarize_table(table):
    body = table["payload"].decode("utf-8")[:6000]
    return ask_vision(
        [text_part(
            f"Section: {table['section'] or 'unknown'}\n"
            f"Caption: {table['caption'] or 'none'}\n\n"
            f"Table:\n{body}\n\n{TABLE_PROMPT}"
        )]
    )


def caption_image(image):
    # The image ALONE gives you "a line chart with colored lines". Adding the
    # caption, section and neighbouring paragraph is what gets you the scenario
    # names - which is what people actually search for.
    return ask_vision([
        image_part(image["payload"]),
        text_part(
            f"Section: {image['section'] or 'unknown'}\n"
            f"Figure caption: {image['caption'] or 'none'}\n"
            f"Nearby text: {image['nearby'][:1000] or 'none'}\n\n{IMAGE_PROMPT}"
        ),
    ])


# =============================================================================
# STEP 6  —  Upload the ORIGINALS to S3
# =============================================================================

def upload(s3, doc_id, item):
    key = s3_key(doc_id, item["page"], item["kind"], item["index"], item["ext"])
    ctype = "image/png" if item["ext"] == "png" else "text/markdown"
    uri = s3_upload(s3, key, item["payload"], ctype)

    # Tables get an HTML copy alongside the Markdown.
    if item["kind"] == "table" and item.get("html"):
        s3_upload(s3, key.replace(".md", ".html"), item["html"].encode("utf-8"), "text/html")
    return uri


# =============================================================================
# STEP 7  —  Embed everything and upsert to Pinecone
# =============================================================================

def store_text(text):
    """The text we put in Pinecone metadata.

    Clipping here is a last-resort safety net, not normal operation - the model
    only ever sees what lands in metadata, so a silent clip means it answers
    from a sentence that stops halfway. Say so loudly if it ever happens.
    """
    if len(text.encode("utf-8")) > METADATA_TEXT_LIMIT:
        print(f"  WARNING: clipping a chunk to {METADATA_TEXT_LIMIT} bytes "
              f"- consider lowering CHUNK_TOKENS")
    return clip(text, METADATA_TEXT_LIMIT)


def ensure_index(pc):
    from pinecone import ServerlessSpec
    if not pc.has_index(PINECONE_INDEX):
        print(f"Creating index '{PINECONE_INDEX}' ...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(1)
    return pc.Index(PINECONE_INDEX)


# =============================================================================
# MAIN
# =============================================================================

def ingest(pdf_path):
    start = time.time()
    doc_id = make_doc_id(pdf_path)
    filename = os.path.basename(pdf_path)
    print(f"doc_id = {doc_id}")

    # STEPS 1-4
    doc = parse_pdf(pdf_path)
    chunks = extract_text(doc)
    tables, images = extract_tables_and_images(doc)
    print(f"Found {len(chunks)} chunks, {len(tables)} tables, {len(images)} images")

    # STEP 5 - this is the part that costs money, one call per table/image
    for t in tables:
        print(f"  summarizing table {t['index'] + 1}/{len(tables)} (page {t['page']})")
        t["search_text"] = summarize_table(t)
    for im in images:
        print(f"  captioning image {im['index'] + 1}/{len(images)} (page {im['page']})")
        im["search_text"] = caption_image(im)

    everything = chunks + tables + images

    # STEP 6
    s3 = get_s3()
    print("Uploading originals to S3 ...")
    for item in everything:
        # Only tables and images have an "original" that differs from what we
        # embedded, so only they need S3. A text chunk IS its own original - it
        # already sits in Pinecone metadata, and query.py reads it from there.
        # Uploading it too would be write-only: nothing would ever read it back.
        item["s3_uri"] = (
            upload(s3, doc_id, item) if item["kind"] in ("table", "image") else ""
        )

    # STEP 7
    print("Embedding ...")
    vectors = embed([item["search_text"] for item in everything])

    records = [{
        "id": make_vector_id(doc_id, item["kind"], item["index"]),
        "values": vec,
        "metadata": {
            "doc_id": doc_id,
            "source_file": filename,
            "type": item["kind"],
            "page": item["page"],
            "section": clip(item["section"], 500),
            "s3_uri": item["s3_uri"],      # <- the pointer back to the original
            "text": store_text(item["search_text"]),
        },
    } for item, vec in zip(everything, vectors)]

    index = ensure_index(get_pinecone())
    for i in range(0, len(records), 100):            # batches of 100
        index.upsert(vectors=records[i:i + 100], namespace=NAMESPACE)

    print(f"\nUpserted {len(records)} vectors in {time.time() - start:.0f}s")
    print(index.describe_index_stats())


def inspect(pdf_path):
    """Parse only. No S3, no Pinecone, no LLM - so it runs with zero API keys.
    Always run this first on a new PDF."""
    doc = parse_pdf(pdf_path)
    chunks = extract_text(doc)
    tables, images = extract_tables_and_images(doc)

    print(f"\n{len(chunks)} chunks, {len(tables)} tables, {len(images)} images\n")
    for t in tables:
        print(f"  TABLE {t['index']}  page {t['page']:>3}  {t['section'][:55]}")
    for im in images:
        print(f"  IMAGE {im['index']}  page {im['page']:>3}  "
              f"{len(im['payload']) // 1024:>4}KB  {im['section'][:45]}")
    if chunks:
        print(f"\nFirst chunk (page {chunks[0]['page']}):\n{chunks[0]['search_text'][:300]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python ingest.py <file.pdf> [--inspect]")
    if "--inspect" in sys.argv:
        inspect(sys.argv[1])
    else:
        ingest(sys.argv[1])
