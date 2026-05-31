# Understanding Embeddings

## What are Embeddings?

Embeddings are numerical representations of text that capture its meaning.

In simple terms, an embedding model converts text into a list of numbers called a vector.

```text
"office work"
        ↓
Embedding Model
        ↓
[0.23, -0.81, 0.45, ...]
```

The numbers themselves are not important. What matters is that similar meanings produce similar vectors.

---

## Why Do We Need Embeddings?

Computers cannot understand text the way humans do.

For example:

```text
Office Work
Corporate Job
Desk Job
```

As humans, we know these phrases are related.

To a computer, they are just different words.

Embeddings help convert text into a mathematical form so that similar meanings are placed close together in vector space.

---

## How Embeddings Help in RAG

Suppose we have a document containing:

```text
Employees receive 20 paid leave days annually.
```

A user asks:

```text
How many vacation days do employees get?
```

Even though the document uses the phrase "paid leave" and the question uses "vacation days", embeddings understand that the meanings are similar.

This allows the retrieval system to find the correct document chunk.

---

## Embeddings in a RAG Pipeline

```text
Document
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Database
   ↓
Similarity Search
   ↓
Relevant Chunks
```

Every chunk is converted into an embedding and stored in a vector database.

When a user asks a question, the question is also converted into an embedding.

The vector database then finds the chunks with the most similar embeddings.

---

## Key Takeaway

Embeddings are the foundation of modern RAG systems.

They convert text into vectors so that machines can search based on meaning rather than exact keywords.

**Text → Embeddings → Similarity Search → Better Retrieval**
