import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "397823e8")
KNOWLEDGE_BASE_PATH = "knowledge_base.txt"

# ── LLM & Graph Transformer Setup ─────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# LLMGraphTransformer extracts nodes and relationships from text
graph_transformer = LLMGraphTransformer(
    llm=llm,
    # Optionally restrict which node/relationship types to extract
    # allowed_nodes=["Person", "Organization", "Product", "Technology"],
    # allowed_relationships=["FOUNDED_BY", "DEVELOPED", "ACQUIRED", "INVESTED_IN"],
)

# Neo4j Connection 
print("Connecting to Neo4j...")
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)
print("Connected to Neo4j successfully.\n")


def load_text_file(file_path: str) -> str:
    """Read the knowledge base text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Simple paragraph-aware chunking.
    Splits on double newlines then merges until chunk_size is reached.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + paragraph).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def clear_existing_graph(graph: Neo4jGraph) -> None:
    """Remove all nodes and relationships from the graph before reloading."""
    print("Clearing existing graph data...")
    graph.query("MATCH (n) DETACH DELETE n")
    print("Graph cleared.\n")


def extract_and_store_graph(chunks: list[str], graph: Neo4jGraph) -> None:
    """
    For each text chunk:
      1. Wrap it in a LangChain Document
      2. Use LLMGraphTransformer to extract graph documents (nodes + relationships)
      3. Add the graph documents to Neo4j
    """
    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        print(f"Processing chunk {i}/{total}...")
        doc = Document(page_content=chunk)
        print(f"  Original Text: {chunk[:500]}")

        # Extract entities and relationships with the LLM
        graph_docs = graph_transformer.convert_to_graph_documents([doc])

        # Log what was extracted
        for gd in graph_docs:
            node_labels = [f"{n.id} ({n.type})" for n in gd.nodes]
            rel_labels = [f"{r.source.id} --[{r.type}]--> {r.target.id}" for r in gd.relationships]
            print(f"  Nodes   : {', '.join(node_labels) if node_labels else 'none'}")
            print(f"  Relations: {', '.join(rel_labels) if rel_labels else 'none'}")

        # Persist to Neo4j
        graph.add_graph_documents(
            graph_docs,
            # baseEntityLabel=True,   # adds a generic __Entity__ label to all nodes
            include_source=True,    # links each entity back to its source document
        )

    print()


def print_graph_stats(graph: Neo4jGraph) -> None:
    """Print a summary of what is stored in Neo4j."""
    node_count = graph.query("MATCH (n) RETURN count(n) AS count")[0]["count"]
    rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
    node_types = graph.query(
        "MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label ORDER BY label"
    )
    rel_types = graph.query(
        "MATCH ()-[r]->() RETURN DISTINCT type(r) AS type ORDER BY type"
    )

    print("=" * 60)
    print("Graph Summary")
    print("=" * 60)
    print(f"Total Nodes        : {node_count}")
    print(f"Total Relationships: {rel_count}")
    print(f"Node Types         : {[r['label'] for r in node_types]}")
    print(f"Relationship Types : {[r['type'] for r in rel_types]}")
    print("=" * 60)

if __name__ == "__main__":
    print(f"Loading knowledge base from: {KNOWLEDGE_BASE_PATH}\n")
    raw_text = load_text_file(KNOWLEDGE_BASE_PATH)

    chunks = split_into_chunks(raw_text, chunk_size=1000)
    print(f"Split into {len(chunks)} chunks.\n")

    clear_existing_graph(graph)

    print("Extracting entities and relationships with LLM...\n")
    extract_and_store_graph(chunks, graph)

    # Refresh the schema so the query script sees up-to-date metadata
    graph.refresh_schema()

    print_graph_stats(graph)
    print("\nKnowledge graph successfully loaded into Neo4j!")
    print("Run 2_graph_rag_query.py to start asking questions.\n")
