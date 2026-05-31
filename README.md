# Complete RAG Masterclass Roadmap 

A comprehensive roadmap to learn Retrieval-Augmented Generation (RAG) from scratch to building production-grade Agentic RAG systems.

---

## Module 1: Foundations

### 1. What is RAG?

* LLM limitations
* Hallucinations
* Knowledge cutoff
* Motivation for RAG

### 2. How RAG Works End-to-End

* Retrieval
* Augmentation
* Generation workflow

### 3. RAG vs Fine-Tuning vs Long Context Models

* Strengths and weaknesses
* When to use each approach

### 4. Components of a RAG System

* Documents
* Embeddings
* Vector databases
* Retrievers
* LLMs

---

## Module 2: Embeddings & Vector Search

### 5. What are Embeddings?

* Semantic representations
* Vector space concepts

### 6. How Embedding Models Work

* OpenAI embeddings
* BGE
* E5
* Instructor models

### 7. Similarity Search Explained

* Cosine similarity
* Dot product
* Euclidean distance

### 8. What is a Vector Database?

* Why vector databases exist
* Core architecture

### 9. FAISS Explained

* Indexing
* Similarity search

### 10. ChromaDB Explained

* Local vector storage
* Retrieval

### 11. Pinecone, Milvus and Weaviate Explained

* Managed vector databases
* Comparison

### 12. How Vector Search Works Internally

* ANN
* HNSW
* IVF
* Indexing techniques

---

## Module 3: Document Processing

### 13. Loading Data into RAG Systems

* PDFs
* Websites
* APIs
* Databases

### 14. Chunking Explained

* Fixed-size chunking
* Overlapping chunks

### 15. Recursive Chunking

* Hierarchical splitting

### 16. Semantic Chunking

* Meaning-aware chunking

### 17. Context-Aware Chunking

* Preserving document structure

### 18. Chunk Size vs Retrieval Quality

* Trade-offs
* Experiments

---

## Module 4: Building Basic RAG

### 19. Build Your First RAG Chatbot from Scratch

* End-to-end implementation

### 20. Build RAG with LangChain

* Chains
* Retrievers
* Vector stores

### 21. Build RAG with LlamaIndex

* Indexes
* Retrieval workflows

### 22. Query Flow Inside a RAG Pipeline

* End-to-end execution path

---

## Module 5: Retrieval Deep Dive

### 23. Why Simple Retrieval Fails

* Missing context
* Irrelevant retrieval

### 24. Top-K Retrieval Explained

* Ranking and selection

### 25. Similarity Threshold Retrieval

* Filtering weak matches

### 26. Metadata Filtering

* Attribute-based retrieval

### 27. BM25 Explained

* Keyword search

### 28. Dense Retrieval Explained

* Embedding-based retrieval

### 29. Sparse Retrieval Explained

* Traditional information retrieval

### 30. Hybrid Search Explained

* Combining dense and sparse retrieval

---

## Module 6: Advanced Retrieval

### 31. Multi-Query Retrieval

* Query diversification

### 32. Query Expansion Techniques

* Improving recall

### 33. Self-Query Retrieval

* LLM-generated filters

### 34. Parent Document Retrieval

* Parent-child document relationships

### 35. Contextual Compression Retrieval

* Compressing retrieved context

### 36. Ensemble Retrieval

* Combining multiple retrievers

---

## Module 7: Re-ranking

### 37. Why Re-ranking is Needed

* Retrieval limitations

### 38. Cross Encoder Re-rankers

* BGE Reranker
* Cohere Reranker
* Jina Reranker

### 39. Building Retrieval + Re-ranking Pipelines

* Production retrieval stacks

---

## Module 8: Evaluation

### 40. How to Evaluate a RAG System

* Evaluation fundamentals

### 41. Retrieval Metrics

* Recall@K
* Precision@K
* MRR
* Hit Rate

### 42. Generation Metrics

* Faithfulness
* Relevance
* Groundedness

### 43. RAGAS Explained

* Automated evaluation

### 44. DeepEval Explained

* Evaluation framework

### 45. LLM-as-a-Judge Evaluation

* Judge-based assessments

---

## Module 9: Production RAG

### 46. Production Challenges in RAG

* Latency
* Scaling
* Cost optimization

### 47. Caching Strategies for RAG

* Semantic cache
* Retrieval cache

### 48. Incremental Indexing & Real-Time Updates

* Fresh data pipelines

### 49. Security and Access Control in RAG

* Enterprise security

### 50. Monitoring and Observability for RAG Systems

* Metrics
* Logging
* Tracing

---

## Module 10: Advanced RAG Architectures

### 51. GraphRAG Explained

* Knowledge graphs
* Graph-based retrieval

### 52. Multi-Hop RAG Explained

* Multi-document reasoning

### 53. RAPTOR Explained

* Recursive summarization trees

### 54. Long Context RAG

* Retrieval vs long context windows

### 55. Multimodal RAG

* Text
* Images
* Audio

---

## Module 11: Self-Improving RAG Architectures

### 56. Self-RAG Explained

* Self-reflection
* Retrieval validation

### 57. CRAG (Corrective RAG) Explained

* Retrieval correction mechanisms

### 58. Adaptive RAG Explained

* Dynamic retrieval strategies

---

## Module 12: Structured & Enterprise RAG

### 59. SQL RAG Explained

* Natural language to SQL

### 60. Structured Data RAG

* Tables
* Warehouses
* Enterprise data

### 61. Knowledge Graph Retrieval

* Graph-based search systems

---

## Module 13: Agentic RAG

### 62. What is Agentic RAG?

* Traditional vs Agentic RAG

### 63. Tool Calling in Agentic RAG

* APIs
* Search tools
* Databases

### 64. Query Planning Agents

* Task decomposition
* Planning workflows

### 65. Reflection Agents

* Self-review
* Error correction

### 66. Multi-Agent RAG Systems

* Specialized collaborating agents

### 67. Deep Research Systems Explained

* Iterative retrieval
* Reasoning loops

### 68. MCP and Agentic RAG

* Model Context Protocol
* Tool interoperability

---

## Module 14: Capstone Project

### 69. Designing a Production-Ready RAG Architecture

* System design
* Scalability considerations

### 70. Building a Production-Ready Agentic RAG Chatbot

* Complete implementation
* Evaluation
* Monitoring
* Deployment

### 71. Future of RAG

* Agent memory
* Long-context models
* Emerging RAG architectures

---

This roadmap takes a learner from **zero RAG knowledge** to building **production-grade Agentic RAG systems**.
