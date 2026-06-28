"""
ingest.py  –  Push table metadata from NeonDB into Pinecone

Run this once before starting the app, and re-run it whenever the schema changes.

Usage:
    python ingest.py
"""

import os
import time
import psycopg2
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "text2sql-rag")
PINECONE_NAMESPACE  = "text2sql"
DATABASE_URL        = os.getenv("NEONDB_CONNECTION_STRING")


def get_table_metadata(conn) -> dict[str, list[dict]]:
    query = """
        SELECT
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()

    tables: dict[str, list[dict]] = {}
    for table_name, col_name, data_type, nullable, default in rows:
        if table_name not in tables:
            tables[table_name] = []
        tables[table_name].append({
            "column":   col_name,
            "type":     data_type,
            "nullable": nullable,
            "default":  default,
        })
    return tables


def get_foreign_keys(conn) -> list[dict]:
    query = """
        SELECT
            tc.table_name         AS from_table,
            kcu.column_name       AS from_column,
            ccu.table_name        AS to_table,
            ccu.column_name       AS to_column
        FROM information_schema.table_constraints   AS tc
        JOIN information_schema.key_column_usage    AS kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public';
    """
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    return [
        {"from_table": r[0], "from_column": r[1], "to_table": r[2], "to_column": r[3]}
        for r in rows
    ]


def build_fk_lookup(foreign_keys: list[dict]) -> dict:
    fk_map: dict = {}
    for fk in foreign_keys:
        table  = fk["from_table"]
        column = fk["from_column"]
        if table not in fk_map:
            fk_map[table] = {}
        fk_map[table][column] = {"to_table": fk["to_table"], "to_column": fk["to_column"]}
    return fk_map


def build_table_description(table_name: str, columns: list[dict], fk_map: dict) -> str:
    lines = [f"Table: {table_name}", "Columns:"]
    for col in columns:
        nullable_str = "nullable" if col["nullable"] == "YES" else "NOT NULL"
        line = f"  - {col['column']} ({col['type']}, {nullable_str})"
        fk_info = fk_map.get(table_name, {}).get(col["column"])
        if fk_info:
            line += f"  →  FK to {fk_info['to_table']}.{fk_info['to_column']}"
        lines.append(line)

    if table_name in fk_map:
        lines.append("Relationships:")
        for col_name, fk in fk_map[table_name].items():
            lines.append(f"  - {table_name}.{col_name} → {fk['to_table']}.{fk['to_column']}")

    return "\n".join(lines)


def ensure_pinecone_index(pc: Pinecone, index_name: str):
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        print(f"[Pinecone] Creating index '{index_name}' ...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("[Pinecone] Waiting for index to be ready ...")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        print("[Pinecone] Index is ready.")
    else:
        print(f"[Pinecone] Index '{index_name}' already exists.")


def ingest_to_pinecone(documents: list[Document]):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=PINECONE_NAMESPACE,
    )
    print(f"[Pinecone] Upserted {len(documents)} documents into namespace '{PINECONE_NAMESPACE}'.")


def main():
    print("=== Text-to-SQL Ingest ===\n")

    print("[NeonDB] Connecting ...")
    conn = psycopg2.connect(DATABASE_URL)
    print("[NeonDB] Connected.\n")

    print("[NeonDB] Reading table metadata ...")
    tables       = get_table_metadata(conn)
    foreign_keys = get_foreign_keys(conn)
    conn.close()

    fk_map = build_fk_lookup(foreign_keys)
    print(f"[NeonDB] {len(tables)} tables, {len(foreign_keys)} foreign keys.\n")

    documents: list[Document] = []
    for table_name, columns in tables.items():
        description = build_table_description(table_name, columns, fk_map)
        documents.append(Document(
            page_content=description,
            metadata={
                "table_name":   table_name,
                "column_count": len(columns),
                "source":       "neondb_schema",
            },
        ))
        print(f"  Built: {table_name}")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    ensure_pinecone_index(pc, PINECONE_INDEX_NAME)

    print("\n[Pinecone] Ingesting ...")
    ingest_to_pinecone(documents)

    print("\n=== Done. Run app.py to start chatting. ===")


if __name__ == "__main__":
    main()
