"""
GraphRAG Tutorial - Step 2: Query the Knowledge Graph

Flow for every question:
  1. Entity extraction  — LLM pulls named entities out of the question
  2. Full-text search   — Neo4j full-text index finds actual stored node IDs
  3. Graph traversal    — fetch all relationships for those matched nodes
  4. Vector retrieval   — embedding similarity search over Document nodes
  5. Single LLM call    — combined context (graph + vector) → final answer

Why this beats LLM-generated Cypher with exact IDs:
  - LLMGraphTransformer stores "Openai" not "OpenAI" — exact match fails silently
  - Full-text index is case-insensitive and handles partial matches by default
  - Graph traversal with known IDs is reliable; no fragile generated Cypher

Run 1_load_to_neo4j.py first to populate the graph.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ── Models ─────────────────────────────────────────────────────────────────────

llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ── Neo4j ──────────────────────────────────────────────────────────────────────

print("Connecting to Neo4j...")
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)
graph.refresh_schema()
print("Connected.\n")


def setup_fulltext_index() -> None:
    """
    Create a full-text index on all entity node labels present in the graph.
    Dynamically discovers labels so it works regardless of what LLMGraphTransformer chose.
    """
    try:
        # Discover labels that actually have an id property (skip Document)
        label_rows = graph.query(
            "MATCH (n) WHERE n.id IS NOT NULL AND NOT 'Document' IN labels(n) "
            "UNWIND labels(n) AS l RETURN DISTINCT l ORDER BY l"
        )
        labels = [r["l"] for r in label_rows]
        if not labels:
            print("No entity nodes found. Run 1_load_to_neo4j.py first.\n")
            return

        label_pattern = "|".join(labels)
        graph.query(f"""
            CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
            FOR (n:{label_pattern}) ON EACH [n.id]
        """)
        print(f"Full-text index ready on labels: {labels}\n")
    except Exception as e:
        print(f"Full-text index note: {e}\n")


# ── Prompts ────────────────────────────────────────────────────────────────────

ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract named entities (people, organizations, products, technologies)
from the question. Return ONLY a comma-separated list of entity names, nothing else.
Example output: OpenAI, Sam Altman, GPT-4""",
    ),
    ("human", "{question}"),
])

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant answering questions about an AI ecosystem.
You have two sources of retrieved information:

GRAPH DATA (entities and relationships from Neo4j):
{graph_context}

VECTOR DATA (semantically similar text passages):
{vector_context}

Use both sources to give a comprehensive answer.
Prefer graph data for precise entity/relationship facts.
Use vector data for broader context and explanations.""",
    ),
    ("human", "Question: {question}"),
])

# ── Graph retrieval (3-step robust approach) ───────────────────────────────────

entity_chain = ENTITY_EXTRACTION_PROMPT | llm | StrOutputParser()


def extract_entities(question: str) -> list[str]:
    """Step 1 — LLM extracts named entities from the question."""
    raw = entity_chain.invoke({"question": question})
    return [e.strip() for e in raw.split(",") if e.strip()]


def find_nodes_by_fulltext(entities: list[str]) -> list[str]:
    """
    Step 2 — CONTAINS search finds actual node IDs stored in Neo4j.
    Works on any label, case-insensitive — no index dependency.
    Falls back to full-text index if CONTAINS returns nothing.
    """
    found_ids = []
    for entity in entities:
        # Primary: CONTAINS search — handles 'Openai' when searching 'OpenAI'
        results = graph.query(
            """MATCH (n)
               WHERE n.id IS NOT NULL
                 AND toLower(n.id) CONTAINS toLower($search)
               RETURN n.id AS id, labels(n) AS labels
               LIMIT 5""",
            {"search": entity},
        )
        for r in results:
            found_ids.append(r["id"])
            print(f"    contains  '{entity}' → '{r['id']}' {r['labels']}")

        # Fallback: full-text index (if CONTAINS found nothing)
        if not results:
            try:
                ft_results = graph.query(
                    """CALL db.index.fulltext.queryNodes("entity_fulltext", $entity)
                       YIELD node, score
                       WHERE score > 0.3
                       RETURN node.id AS id, score
                       ORDER BY score DESC LIMIT 3""",
                    {"entity": entity},
                )
                for r in ft_results:
                    found_ids.append(r["id"])
                    print(f"    fulltext  '{entity}' → '{r['id']}' (score {r['score']:.2f})")
            except Exception:
                pass

    return list(dict.fromkeys(found_ids))  # deduplicate, preserve order


def traverse_graph(node_ids: list[str]) -> str:
    """
    Step 3 — Fetch all direct relationships for the matched nodes.
    No fragile generated Cypher — just traverse from known IDs.
    """
    if not node_ids:
        return "No matching entities found in the graph."

    results = graph.query(
        """MATCH (n)-[r]-(m)
           WHERE n.id IN $ids
           RETURN n.id AS source, type(r) AS relationship, m.id AS target
           ORDER BY n.id, type(r)
           LIMIT 40""",
        {"ids": node_ids},
    )

    if not results:
        return f"Entities found {node_ids} but no relationships stored."

    lines = [
        f"{row['source']} --[{row['relationship']}]--> {row['target']}"
        for row in results
    ]
    return "\n".join(lines)


def retrieve_from_graph(question: str) -> tuple[list[str], list[str], str]:
    """Run all 3 graph-retrieval steps, return (entities, node_ids, graph_context)."""
    entities = extract_entities(question)
    node_ids = find_nodes_by_fulltext(entities)
    graph_context = traverse_graph(node_ids)
    return entities, node_ids, graph_context


# ── Vector retrieval ───────────────────────────────────────────────────────────

def get_vector_retriever():
    try:
        store = Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            index_name="graphrag_vector_index",
            node_label="Document",
            text_node_property="text",
            embedding_node_property="embedding",
        )
        print("Loaded existing vector index.\n")
    except Exception:
        print("Building vector index from Document nodes...\n")
        store = Neo4jVector.from_existing_graph(
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            index_name="graphrag_vector_index",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
    return store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )


def retrieve_from_vector(question: str, retriever) -> list[str]:
    try:
        docs = retriever.invoke(question)
        return [doc.page_content for doc in docs]
    except Exception as e:
        return [f"Vector retrieval error: {e}"]


# ── Main GraphRAG query function ───────────────────────────────────────────────

def graph_rag_query(question: str, retriever) -> str:
    print("\n" + "=" * 60)
    print(f"INPUT QUERY: {question}")
    print("=" * 60)

    # Step 1+2+3 — Graph retrieval
    print("\n[Step 1] Extracting entities from question...")
    entities, node_ids, graph_context = retrieve_from_graph(question)
    print(f"  Extracted entities : {entities}")
    print(f"  Matched node IDs   : {node_ids}")
    print(f"\n  Graph relationships:\n  {graph_context[:300]}...")

    # Step 4 — Vector retrieval
    print("\n[Step 2] Running vector similarity search...")
    vector_chunks = retrieve_from_vector(question, retriever)
    print(f"  Retrieved {len(vector_chunks)} chunk(s):")
    for i, chunk in enumerate(vector_chunks, 1):
        print(f"  [{i}] {chunk[:120]}...")

    # Step 5 — Single LLM call with combined context
    graph_ctx  = graph_context if graph_context else "No graph data found."
    vector_ctx = "\n\n---\n\n".join(vector_chunks) if vector_chunks else "No vector data found."

    print("\n[Step 3] Sending combined context to LLM...")
    print(f"  - Graph context  ({len(graph_ctx)} chars)")
    print(f"  - Vector context ({len(vector_ctx)} chars)")

    final_chain = FINAL_ANSWER_PROMPT | llm | StrOutputParser()
    return final_chain.invoke({
        "question": question,
        "graph_context": graph_ctx,
        "vector_context": vector_ctx,
    })


# ── Interactive loop ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_fulltext_index()
    retriever = get_vector_retriever()

    print("=" * 60)
    print("GraphRAG Q&A  |  type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        answer = graph_rag_query(user_input, retriever)

        print("\n" + "─" * 60)
        print("FINAL ANSWER:")
        print("─" * 60)
        print(answer)
        print("─" * 60 + "\n")
