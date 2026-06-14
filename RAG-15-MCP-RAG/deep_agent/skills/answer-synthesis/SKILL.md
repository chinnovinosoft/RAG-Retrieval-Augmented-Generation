---
name: answer-synthesis
description: "Merge retrieved document results from all search tasks into one coherent, structured, evidence-based answer to the original user query. Always activate last, after all retrieval tasks complete. Handles deduplication, evidence mapping, and final answer formatting."
---

# Answer Synthesis Skill

## Overview
Take the collected retrieval results and produce a single, comprehensive, well-structured response to the **original** user query (not just the sub-questions).

## Step 1 — Deduplicate Results

Across all result sets, identify duplicate documents:
- Same `id` → keep only the copy with the highest `score`
- Near-identical `text` (> 85% overlap) → keep highest score, discard the rest
- Different `id` with genuinely different content → keep both

## Step 2 — Map Evidence to Sub-Questions

For each sub-question, identify which deduplicated documents answer it.
Rank documents by score descending. Use the top 3–5 per sub-question for synthesis.

## Step 3 — Write the Answer

### Structure (use markdown)
```
## Summary
<2–3 sentences directly answering the original query>

## <Section for Topic 1>
<Evidence-backed explanation. Quote or paraphrase retrieved text.>

## <Section for Topic 2>
<Evidence-backed explanation.>

... (add sections as needed for the query scope)

## Key Takeaways
- <Point 1>
- <Point 2>
- <Point 3>
```

### Rules
1. **Stay grounded** — every factual claim must come from a retrieved document. Do not hallucinate.
2. **Original query first** — the answer must address the original query, not just list sub-question answers.
3. **No redundancy** — if the same fact appears in multiple sources, state it once with the best evidence.
4. **Conflict handling** — if two sources disagree, present both views and note the discrepancy.
5. **Scope matching** — SIMPLE queries get direct answers (1–3 paragraphs). RESEARCH queries get full structured reports.
6. **Honest gaps** — if a sub-question returned 0 results, say "I didn't find information about X in the knowledge base."

## Step 4 — Cite Sources

After the answer, include a sources section:

```
## Sources
| # | Score | ID | Excerpt |
|---|-------|----|---------|
| 1 | 0.92  | doc_abc | "GraphRAG constructs a knowledge graph…" |
| 2 | 0.87  | doc_xyz | "Vector RAG retrieves chunks by cosine similarity…" |
```

Limit to top 5 sources overall.

## Example

**Original query:** "Compare GraphRAG vs Vector RAG and when should I use each?"

**Good synthesis:**
```
## Summary
GraphRAG and Vector RAG are both retrieval-augmented generation strategies but differ
in how they represent knowledge. GraphRAG uses a knowledge graph; Vector RAG uses dense
vector embeddings. The choice depends on whether your domain has rich entity relationships.

## GraphRAG
GraphRAG constructs an explicit knowledge graph from documents, enabling multi-hop
reasoning across connected entities. [Source 1, score 0.92]

## Vector RAG
Vector RAG encodes documents into dense vectors and retrieves the most similar chunks
by cosine distance. It is simpler to implement and scales well to large corpora.
[Source 2, score 0.87]

## When to Use Each
- Use **GraphRAG** when: queries require reasoning over entity relationships,
  the domain has rich structured knowledge, or multi-hop inference is needed.
- Use **Vector RAG** when: queries are topic-based rather than entity-relational,
  you need fast setup, or your corpus lacks clear entity structure.

## Key Takeaways
- GraphRAG = graph-based; stronger multi-hop reasoning
- Vector RAG = embedding-based; faster, simpler, scales easily
- Choose GraphRAG for entity-rich domains; Vector RAG for general retrieval
```
