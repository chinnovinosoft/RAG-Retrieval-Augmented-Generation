# Multimodal RAG — text, tables, and images from a PDF

**The pattern: summarize for search, send the ORIGINAL for the answer.**

Most RAG pipelines only index text, so everything in the figures and tables is
lost. The naive fix — summarize an image, then answer from the summary — throws
away the detail that made the figure worth reading in the first place.

So we split the job in two:

| type | what gets **embedded** | what the model actually **sees** |
|---|---|---|
| text | the chunk | the chunk |
| table | a short LLM summary | the **full Markdown table** from S3 |
| image | an LLM caption | the **original PNG** from S3 |

The summaries exist only so search can *find* the thing. One text embedding
model covers all three types, so there's one vector space and one `top_k`.

## Three files

| file | role |
|---|---|
| `common.py` | config + the bits both sides must agree on (embeddings, S3, IDs) |
| `ingest.py` | **write side** — PDF → S3 + Pinecone |
| `query.py` | **read side** — question → answer with citations |

`common.py` exists for one reason: if ingest and query ever embed with different
settings, search doesn't error — it just quietly returns nonsense. One
definition, imported by both.

```
ingest.py   PDF ─ Docling ─┬─ text chunks ─────────────┐
                           ├─ tables  → summarize ─────┼─ embed → Pinecone
                           └─ images  → caption ───────┘    originals → S3

query.py    question → Pinecone (top 20) → rerank (top 5)
                     → fetch ORIGINALS from S3
                     → one multimodal message → answer + [file, page]
```

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env     # then fill in your keys
```

You need **two** keys plus AWS:

- `OPENAI_API_KEY` — does everything: embeddings, captions, summaries, answers
- `PINECONE_API_KEY`
- `S3_BUCKET` + AWS credentials (normal boto3 chain)

## Run

```bash
# 1. Parse only — no API keys needed. Always start here on a new PDF.
python ingest.py IPCC_AR6_SYR_SPM-2.pdf --inspect

# 2. Full ingest: parse → caption/summarize → S3 → embed → Pinecone
python ingest.py IPCC_AR6_SYR_SPM-2.pdf

# 3. Ask
python query.py "How much warming is projected under SSP5-8.5?"
python query.py "What does the risk figure show?" --type image
python query.py "..." --no-rerank
```

First run downloads ~500 MB of Docling layout + OCR models. Slow once, then cached.

## ingest.py, step by step

| Step | What it does |
|---|---|
| 1 | Parse with Docling — OCR, table structure, figure images at 2× scale |
| 2 | Text → `HybridChunker`, section-aware, 512 tokens, **same tokenizer as the embedder** |
| 3 | Tables → Markdown + HTML |
| 4 | Images → PNG, skipping tiny (<100px) and duplicate (hash-matched) ones |
| 5 | Vision model captions each image and summarizes each table |
| 6 | Originals → S3 at `{doc_id}/page_{n}/{type}_{index}.{ext}` |
| 7 | Embed all three types with one model, upsert in batches of 100 |

## query.py, step by step

| Step | What it does |
|---|---|
| 1 | Embed the question, Pinecone `top_k=20` |
| 2 | Rerank to 5 with `bge-reranker-v2-m3` (`USE_RERANKER=false` to skip) |
| 3 | **Swap each summary for the original** — download table Markdown / PNG from S3 |
| 4 | One multimodal message → answer, citing `[file, page]`, refusing if unsupported |
| 5 | Print sources, with a presigned link for each figure |

## Details worth knowing

**Idempotent re-ingest.** `doc_id` is a hash of the file bytes, and each vector
ID is a hash of `doc_id + type + index`. Ingest the same PDF ten times and the
vector count doesn't move — upserts land on the same IDs instead of piling up
duplicate copies.

**Image context matters.** Captioning sends the figure *plus* its caption, its
section heading, and the nearest paragraph. Captioned blind you get "a line
chart with several coloured lines"; with context you get the scenario names —
which is what people actually search for.

**Section headings are matched exactly.** Docling's label set contains both
`section_header` and `page_header`. Matching loosely on `"header"` tags every
figure with the running page banner instead of its real section.

**Metadata limits.** Pinecone caps metadata at 40 KB per record. Image bytes and
full tables never go in — only the short search text (clipped on a UTF-8 byte
boundary) and the `s3_uri` pointing at the real thing.

**Token parameter.** These models reject `max_tokens` and require
`max_completion_tokens`, and that budget also covers internal reasoning tokens —
so keep it generous or the visible answer comes back empty.

**No BM25.** Vector search only, by design.

## Tuning

`.env`: `EMBED_MODEL`, `EMBED_DIM`, `VISION_MODEL`, `USE_RERANKER`,
`PINECONE_INDEX`, `PINECONE_NAMESPACE`, `S3_BUCKET`.

Top of `ingest.py`: `IMAGES_SCALE`, `MIN_IMAGE_PX`, `CHUNK_TOKENS`.
Top of `query.py`: `TOP_K`, `RERANK_TO`, `MAX_IMAGES`.
