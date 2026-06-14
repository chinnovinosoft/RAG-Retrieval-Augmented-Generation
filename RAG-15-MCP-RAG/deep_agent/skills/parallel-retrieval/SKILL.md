---
name: parallel-retrieval
description: "Execute all planned retrieval tasks concurrently rather than sequentially. Activate after retrieval-planning when there are 2 or more sub-questions. Calling tools in parallel reduces total latency and prevents sequential bottlenecks."
---

# Parallel Retrieval Skill

## Overview
When the retrieval plan contains multiple tasks, fire all MCP tool calls at the same time.
Do NOT wait for one search to finish before starting the next.

## The Core Rule

**WRONG — sequential (slow):**
1. Call `semantic_search(query="What is GraphRAG?", top_k=5)` → wait for result
2. Call `semantic_search(query="What is Vector RAG?", top_k=5)` → wait for result
3. Call `hybrid_search(query="Differences?", top_k=8)` → wait for result

**CORRECT — parallel (fast):**
1. Call ALL three tools simultaneously  
2. Wait for ALL results to return  
3. Proceed to answer-synthesis with all results

## When to Use This Skill

| Situation | Action |
|-----------|--------|
| 1 sub-question | Call its tool directly — no parallelism needed |
| 2–5 sub-questions | Call all tools in parallel |
| > 5 sub-questions | Group into batches of 5 and run each batch in parallel |

## Handling Partial Failures

If one tool call fails or returns 0 results:
- Do NOT cancel the other in-progress calls
- Apply the fallback strategy from the retrieval plan for the failed task
- Continue to answer-synthesis with whatever results you have
- If a fallback also fails, note the gap in the synthesis step

## Result Collection

After all parallel calls complete, collect results per sub-question:

```
q1 results: [ {id, score, text, metadata}, ... ]
q2 results: [ {id, score, text, metadata}, ... ]
q3 results: [ {id, score, text, metadata}, ... ]
```

Each `score` is 0–1 (higher = more relevant).
Each result includes:
- `id` — document ID in Pinecone
- `score` — relevance score
- `text` — document content
- `metadata` — structured attributes (source, date, category, etc.)

## Quality Checks Before Synthesis

Before moving to answer-synthesis, verify:
- At least one sub-question has ≥ 1 result with score > 0.5
- If ALL sub-questions return 0 results → report "no relevant information found" immediately
- If SOME return 0 results → proceed with available results, note gaps in the answer

## Deduplication Note
The same document may appear in multiple sub-question result sets.
The answer-synthesis skill handles deduplication — do not filter here.
