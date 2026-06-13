
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

# ── Prompts ────────────────────────────────────────────────────────────────────

CYPHER_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Neo4j Cypher expert. Given the graph schema, write a Cypher
query that retrieves data needed to answer the question.
Only use node labels and relationship types present in the schema.
Always match entity names case-insensitively using toLower(), for example:
  WHERE toLower(n.id) = toLower('OpenAI')
Return ONLY the Cypher query, no explanation.

Schema:
{schema}""",
    ),
    ("human", "Question: {question}"),
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

# ── Graph retrieval ────────────────────────────────────────────────────────────

cypher_chain = CYPHER_GENERATION_PROMPT | llm | StrOutputParser()


def retrieve_from_graph(question: str) -> tuple[str, str]:
    """Ask LLM to generate a Cypher query, run it, return (cypher, results)."""
    cypher_query = cypher_chain.invoke({
        "schema": graph.schema,
        "question": question,
    })

    # Strip markdown code fences if the LLM wrapped the query
    cypher_query = (
        cypher_query.strip()
        .removeprefix("```cypher")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    try:
        results = graph.query(cypher_query)
        results_str = "\n".join(str(row) for row in results) if results else "No results found."
    except Exception as e:
        results_str = f"Cypher execution error: {e}"

    return cypher_query, results_str


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

    # Step 1 — LLM generates and runs Cypher
    print("\n[Step 1] Generating Cypher query from question...")
    cypher_query, graph_results = retrieve_from_graph(question)
    print(f"\n  Generated Cypher:\n  {cypher_query}")
    print(f"\n  Graph results:\n  {graph_results}")

    # Step 2 — Vector retrieval
    print("\n[Step 2] Running vector similarity search...")
    vector_chunks = retrieve_from_vector(question, retriever)
    print(f"  Retrieved {len(vector_chunks)} chunk(s):")
    for i, chunk in enumerate(vector_chunks, 1):
        print(f"  [{i}] {chunk[:120]}...")

    # Step 3 — Single LLM call with combined context
    graph_ctx  = graph_results if graph_results else "No graph data found."
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
    retriever = get_vector_retriever()

    print("=" * 60)
    print("GraphRAG Q&A (Cypher approach)  |  type 'quit' to exit")
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
