---
name: retrieval-planning
description: "Select the optimal MCP search tool and top_k for every sub-question before calling any tool. Activate after question-decomposition (or directly after query-classification for SIMPLE queries). Maps each sub-question to semantic_search, keyword_search, hybrid_search, metadata_search, or multi_search."
---

# Retrieval Planning Skill

## Overview
Before calling any search tool, decide WHICH tool to call and HOW MANY results to fetch for each sub-question. Wrong tool choices degrade result quality.

## Available MCP Search Tools

| Tool | Mechanism | Best For |
|------|-----------|----------|
| `semantic_search` | Dense vector similarity (OpenAI embeddings) | Conceptual queries, paraphrasing, topic understanding |
| `keyword_search` | Sparse BM25 matching | Exact terms, proper nouns, acronyms, code identifiers |
| `hybrid_search` | Dense + sparse combined (recommended default) | Most factual questions — balances precision and recall |
| `metadata_search` | Pinecone filter on document attributes | Queries specifying date, author, category, tag, or version |
| `multi_search` | All three strategies in parallel, merged | Research queries needing maximum recall |

## Tool Selection Rules

Apply these rules per sub-question:

### Use `semantic_search` when:
- The sub-question is conceptual ("What is…", "Explain…", "How does…")
- The answer requires meaning-level matching, not exact keywords
- The query is about ideas rather than named entities

### Use `keyword_search` when:
- The sub-question contains specific technical terms, class names, or acronyms
- You need exact-match results (e.g., "LangGraph", "FAISS", "Pinecone")
- The domain has specialized vocabulary

### Use `hybrid_search` (default) when:
- Unsure between semantic and keyword
- The sub-question mixes conceptual and technical vocabulary
- General factual questions
- Comparison sub-questions

### Use `metadata_search` when:
- The query explicitly mentions a filter attribute: "papers from 2024", "by author X", "category: tutorial"
- You need to filter by structured fields rather than content

### Use `multi_search` when:
- Sub-question is broad / research-type
- Maximum recall matters more than latency
- Previous searches returned insufficient results (escalate strategy)

## top_k Selection

| Sub-question type | Recommended top_k |
|-------------------|--------------------|
| SIMPLE definition | 3–5 |
| COMPLEX single question | 5–7 |
| COMPARISON per entity | 5 |
| RESEARCH sub-question | 7–10 |
| Fallback / escalation | 10 |

## Example Plan

**Original query:** "Explain GraphRAG, compare it with Vector RAG, list advantages and disadvantages, and tell me when to use each."

| ID | Sub-question | Tool | top_k | Rationale |
|----|-------------|------|-------|-----------|
| q1 | What is GraphRAG? | `semantic_search` | 5 | Conceptual definition |
| q2 | What is Vector RAG? | `semantic_search` | 5 | Conceptual definition |
| q3 | Advantages of GraphRAG vs Vector RAG | `hybrid_search` | 8 | Mixed vocab + concepts |
| q4 | When to use GraphRAG vs Vector RAG | `hybrid_search` | 5 | Decision-criteria question |

## Fallback Strategy
If the primary tool returns 0 results:
- `semantic_search` → try `hybrid_search`
- `keyword_search` → try `hybrid_search`
- `hybrid_search` → try `multi_search`
- `metadata_search` → try `hybrid_search` (drop filter)
- `multi_search` → no fallback, report no results found
