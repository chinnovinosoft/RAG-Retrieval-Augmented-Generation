"""
Plain LangGraph RAG Agent
=========================
No deepagents, no skills, no MCP.

Flow:
  user query
      │
      ▼
  [retrieve]  ── Pinecone vector search (OpenAI embeddings)
      │
      ▼
  [generate]  ── GPT-4o answers using retrieved context
      │
      ▼
  final answer

Run:
    python langgraph_client.py
    python langgraph_client.py "What is ISO 27001?"
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from typing_extensions import TypedDict

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "default")
OPENAI_API_KEY      = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL           = os.getenv("LLM_MODEL", "gpt-4o")
TOP_K               = int(os.getenv("TOP_K", "5"))

# ── LangGraph State ───────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    documents: List[dict]   # retrieved chunks [{text, score, metadata}]
    answer: str

# ── Pinecone + OpenAI clients (module-level singletons) ───────────────────────

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
_vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=_embeddings,
    namespace=PINECONE_NAMESPACE,
    pinecone_api_key=PINECONE_API_KEY,
)
_llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)

# ── Node 1: Retrieve ──────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> RAGState:
    query = state["query"]
    logger.info("Retrieving top-%d docs for: %.80s", TOP_K, query)

    results = _vectorstore.similarity_search_with_score(query, k=TOP_K)

    documents = [
        {
            "text": doc.page_content,
            "score": round(float(score), 4),
            "metadata": doc.metadata,
        }
        for doc, score in results
    ]

    logger.info("Retrieved %d documents (top score: %.4f)",
                len(documents), documents[0]["score"] if documents else 0)
    return {**state, "documents": documents}

# ── Node 2: Generate ──────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the user's question using ONLY the context below.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

def generate(state: RAGState) -> RAGState:
    query = state["query"]
    docs  = state["documents"]

    if not docs:
        return {**state, "answer": "No relevant documents found in the knowledge base."}

    context = "\n\n---\n\n".join(
        f"[Score: {d['score']}]\n{d['text']}" for d in docs
    )

    prompt = _PROMPT_TEMPLATE.format(context=context, query=query)
    logger.info("Sending %d chars of context to LLM", len(context))

    response = _llm.invoke(prompt)
    answer = response.content.strip()

    logger.info("Answer generated (%d chars)", len(answer))
    return {**state, "answer": answer}

# ── Build LangGraph ───────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, END

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

_graph = build_graph()

# ── Public API ────────────────────────────────────────────────────────────────

def ask(query: str) -> str:
    """Run the RAG pipeline and return the final answer."""
    result = _graph.invoke({"query": query, "documents": [], "answer": ""})
    return result["answer"]

# ── Interactive / one-shot entry-point ────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # One-shot: python langgraph_client.py "my question"
        q = " ".join(sys.argv[1:])
        print(f"\nQ: {q}\n")
        print(ask(q))
    else:
        # Interactive chat loop
        print("\n" + "=" * 60)
        print("  LangGraph RAG Agent  —  type 'exit' to quit")
        print("=" * 60 + "\n")
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            print("\nAgent: (thinking…)\n")
            try:
                print(f"Agent:\n{ask(q)}\n")
            except Exception as exc:
                logger.error("Error: %s", exc, exc_info=True)
                print(f"[Error] {exc}\n")
