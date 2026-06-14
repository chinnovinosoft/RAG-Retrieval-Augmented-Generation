---
name: question-decomposition
description: "Decompose MULTI_QUESTION, COMPARISON, and RESEARCH queries into the minimum set of focused, independently answerable sub-questions. Activate after query-classification when requires_decomposition=true. Each sub-question maps to exactly one retrieval task."
---

# Question Decomposition Skill

## Overview
When a query is too broad or contains multiple topics, break it into focused sub-questions.
Each sub-question must be:
- **Independent** — answerable on its own without context from other sub-questions
- **Specific** — about one concept, entity, or comparison axis
- **Retrievable** — answerable via a single search call

## When to Activate
Only decompose when `requires_decomposition = true` from the query-classification skill.
For `SIMPLE` queries, skip this skill and go directly to retrieval-planning.

## Decomposition Strategies by Type

### MULTI_QUESTION
Extract each distinct question as its own sub-question.

**Input:**
> "What is GraphRAG, how does it differ from Vector RAG, and when should I use each?"

**Sub-questions:**
- q1: "What is GraphRAG?" — focus: definition
- q2: "What is Vector RAG?" — focus: definition  
- q3: "How does GraphRAG differ from Vector RAG?" — focus: comparison
- q4: "When should I use GraphRAG versus Vector RAG?" — focus: use cases

---

### COMPARISON
Create one sub-question per entity to establish what each is, then one for the direct comparison.

**Input:**
> "Compare GraphRAG vs Vector RAG — advantages, disadvantages, and use cases."

**Sub-questions:**
- q1: "What is GraphRAG? Advantages and disadvantages." — focus: entity A
- q2: "What is Vector RAG? Advantages and disadvantages." — focus: entity B
- q3: "When should I use GraphRAG versus Vector RAG?" — focus: decision criteria

---

### RESEARCH
Decompose into the fundamental axes: definition, mechanism, use cases, advantages, limitations.

**Input:**
> "Give me a comprehensive overview of Retrieval-Augmented Generation."

**Sub-questions:**
- q1: "What is Retrieval-Augmented Generation (RAG)?" — focus: definition
- q2: "How does the RAG pipeline work step by step?" — focus: mechanism
- q3: "What are the advantages and limitations of RAG?" — focus: trade-offs
- q4: "What are the main use cases for RAG systems?" — focus: applications

---

## Limits
- Maximum **5 sub-questions** per original query — avoid over-decomposition
- If two sub-questions would retrieve the same documents, merge them
- Order sub-questions from foundational to advanced (definitions before comparisons)

## Output
For each sub-question, note:
- `id`: q1, q2, q3 …
- `question`: the focused sub-question text
- `focus`: one-phrase label (definition / comparison / use-cases / advantages / mechanism)
- `search_priority`: high / medium / low
