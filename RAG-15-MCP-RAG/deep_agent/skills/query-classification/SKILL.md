---
name: query-classification
description: "Classify every incoming user query before retrieval begins. Detects SIMPLE, COMPLEX, MULTI_QUESTION, COMPARISON, and RESEARCH queries and decides whether decomposition is needed. Always activate this skill first on any RAG request."
---

# Query Classification Skill

## Overview
Before calling any search tool, classify the user query. The classification drives every downstream decision — whether to decompose, which tools to use, and how many results to fetch.

## Classification Types

| Type | Definition | Complexity | Example |
|------|-----------|------------|---------|
| `SIMPLE` | One focused question, single retrieval sufficient | 1–3 | "What is RAG?" |
| `COMPLEX` | One question but requires deep reasoning or multi-step evidence | 4–6 | "How does attention mechanism work mathematically?" |
| `MULTI_QUESTION` | Two or more distinct questions bundled in one prompt | 5–8 | "What is GraphRAG and how is it different from VectorRAG?" |
| `COMPARISON` | Explicitly asks to compare ≥ 2 entities or approaches | 5–8 | "Compare GraphRAG vs Vector RAG" |
| `RESEARCH` | Broad topic — needs comprehensive, multi-angle coverage | 7–10 | "Give me a complete overview of all RAG architectures" |

## Decision Rules

Apply these in order — stop at the first match:

1. Does the query contain **"compare"**, **"vs"**, **"versus"**, **"difference between"**, **"pros and cons"**?  
   → `COMPARISON`

2. Does the query contain **"and"** joining two distinct topics, **"also"**, **"additionally"**, **"furthermore"**, or multiple **"?"** marks?  
   → `MULTI_QUESTION`

3. Does the query ask for **"overview"**, **"survey"**, **"all"**, **"comprehensive"**, **"explain everything"**, **"list all"**?  
   → `RESEARCH`

4. Is the query a single question but requires multi-source evidence or multi-step reasoning?  
   → `COMPLEX`

5. Otherwise → `SIMPLE`

## Decomposition Decision

Set `requires_decomposition = true` if ANY of the following:
- Type is `MULTI_QUESTION`, `COMPARISON`, or `RESEARCH`
- Complexity score ≥ 5
- The query contains ≥ 2 distinct retrievable topics

## Output to Carry Forward
Record these values to use in the next skills:
- `query_type`: one or more of the types above
- `complexity_score`: integer 1–10
- `requires_decomposition`: true / false
- `reasoning`: one sentence explaining the classification
