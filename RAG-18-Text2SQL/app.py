"""
Text-to-SQL RAG  –  LangGraph

                         ┌──────────────────────┐
              START ────►│     router_node      │
                         └──────────┬───────────┘
                       (greeting)   │   (text_to_sql)
               ┌─────────────────────┘       └────────────────────────────────┐
     ┌──────────▼──────────┐            ┌────────────────────────────────────▼──┐
     │   greeting_node     │            │         enhance_query_node            │
     └──────────┬──────────┘            │   LLM rewrites query with table names │
                │                       └────────────────────────────────────┬──┘
              [END]                                                           │
                                        ┌────────────────────────────────────▼──┐
                                        │        retrieve_context_node          │
                                        │   semantic search in Pinecone         │
                                        └────────────────────────────────────┬──┘
                                                                             │
                                        ┌────────────────────────────────────▼──┐
                                        │          generate_sql_node            │
                                        └────────────────────────────────────┬──┘
                                                                             │
                                        ┌────────────────────────────────────▼──┐
                                        │          validate_sql_node            │  EXPLAIN syntax check
                                        └────┬──────────────────────┬───────────┘
                                   (invalid) │                      │ (valid)
                  ┌──────────────────────────┘                      └────────────────────────┐
        ┌─────────▼──────────┐                                ┌──────────────────────────────▼──┐
        │  rewrite_sql_node  │◄───────────────────────────────│       execute_sql_node          │
        │  (max 5 retries)   │              (error)            └──────────────────────────────┬──┘
        └─────────┬──────────┘                                                                │ (done)
                  │                                           ┌──────────────────────────────▼──┐
                  │ loops back to validate     (exhausted)───►│      format_response_node       │
                  └──────────────────────────────────────────►│                                 │
                                                              └──────────────────────────────┬──┘
                                                                                             │
                                                                                           [END]
"""

import os
import json
import psycopg2
from typing import TypedDict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

load_dotenv()

LLM_MODEL          = "gpt-4o-mini"
EMBED_MODEL        = "text-embedding-3-small"
PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME", "text2sql-rag")
PINECONE_NAMESPACE = "text2sql"
DATABASE_URL       = os.getenv("NEONDB_CONNECTION_STRING")
MAX_RETRIES        = 5

ALL_TABLE_NAMES = [
    "departments", "employees", "warehouses", "suppliers", "categories",
    "products", "product_variants", "inventory", "customers", "addresses",
    "orders", "order_items", "payments", "shipping", "returns",
    "return_items", "reviews", "wishlists", "coupons", "audit_logs",
]

llm         = ChatOpenAI(model=LLM_MODEL, temperature=0)
embeddings  = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings,
    namespace=PINECONE_NAMESPACE,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})


class GraphState(TypedDict):
    question:        str
    route:           str
    enhanced_query:  str
    relevant_tables: str
    generated_sql:   str
    sql_result:      Any
    error_message:   str
    retry_count:     int
    final_answer:    str


def run_sql(sql: str) -> tuple[list[dict], str]:
    try:
        conn   = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(sql)

        if cursor.description:
            col_names = [d[0] for d in cursor.description]
            rows      = [dict(zip(col_names, row)) for row in cursor.fetchall()]
        else:
            conn.commit()
            rows = [{"affected_rows": cursor.rowcount}]

        cursor.close()
        conn.close()
        return rows, ""

    except Exception as e:
        return [], str(e)


class RouteDecision(BaseModel):
    route: str = Field(description="Either 'greeting' or 'text_to_sql'")

router_llm = llm.with_structured_output(RouteDecision)

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Classify the user message into exactly one of:
- "greeting"   : greetings, small talk, jokes, anything NOT a data question
- "text_to_sql": any question that needs querying a database

Return only the JSON with the route field."""),
    ("human", "{question}"),
])

def router_node(state: GraphState) -> GraphState:
    print("\n[router] Classifying question ...")
    decision = router_llm.invoke(ROUTER_PROMPT.format_messages(question=state["question"]))
    print(f"[router] Route = {decision.route}")
    return {**state, "route": decision.route}


GREETING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant. Respond naturally — greet back, be playful, keep it short."),
    ("human", "{question}"),
])

def greeting_node(state: GraphState) -> GraphState:
    print("[greeting] Responding ...")
    response = llm.invoke(GREETING_PROMPT.format_messages(question=state["question"]))
    return {**state, "final_answer": response.content}


ENHANCE_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL query enhancement assistant working with a PostgreSQL e-commerce database.

The database has these tables:
{table_names}

The user asked a question that needs to be answered with SQL.
Your job is to rewrite their question into a more precise, SQL-friendly version that:
- Clearly states what data is needed and from which tables
- Mentions relevant table names where obvious
- Keeps the original intent intact but removes ambiguity
- Is optimized to retrieve the correct data in one query

Return only the enhanced question — no SQL, no explanation."""),
    ("human", "User question: {question}"),
])

def enhance_query_node(state: GraphState) -> GraphState:
    print("[enhance_query] Rewriting question for better SQL generation ...")
    table_names = ", ".join(ALL_TABLE_NAMES)
    response = llm.invoke(ENHANCE_QUERY_PROMPT.format_messages(
        table_names = table_names,
        question    = state["question"],
    ))
    enhanced = response.content.strip()
    print(f"[enhance_query] Enhanced query: {enhanced}")
    return {**state, "enhanced_query": enhanced}


def retrieve_context_node(state: GraphState) -> GraphState:
    print("[retrieve_context] Searching Pinecone for relevant tables ...")
    docs = retriever.invoke(state["enhanced_query"])
    relevant_tables = "\n\n---\n\n".join(doc.page_content for doc in docs)
    print(f"[retrieve_context] {len(docs)} table(s) retrieved")
    return {**state, "relevant_tables": relevant_tables}


SQL_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert PostgreSQL query writer.

The user's original question was enhanced to:
{enhanced_query}

Relevant table schemas retrieved from the vector store:
{relevant_tables}

Rules:
- Write a single valid PostgreSQL SELECT query.
- Use proper JOINs when needed. Use table aliases.
- Return raw SQL only — no markdown, no explanation."""),
    ("human", "Original question: {question}"),
])

def generate_sql_node(state: GraphState) -> GraphState:
    print("[generate_sql] Asking LLM to write SQL ...")
    response = llm.invoke(SQL_GEN_PROMPT.format_messages(
        enhanced_query  = state["enhanced_query"],
        relevant_tables = state["relevant_tables"],
        question        = state["question"],
    ))
    sql = response.content.strip()
    print(f"[generate_sql] SQL:\n{sql}")
    return {**state, "generated_sql": sql, "retry_count": 0, "error_message": ""}


def validate_sql_node(state: GraphState) -> GraphState:
    print("[validate_sql] Running EXPLAIN (syntax check) ...")
    try:
        conn   = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(f"EXPLAIN {state['generated_sql']}")
        cursor.close()
        conn.close()
        print("[validate_sql] Syntax OK.")
        return {**state, "error_message": ""}
    except Exception as e:
        error = str(e)
        print(f"[validate_sql] Syntax error: {error}")
        return {**state, "error_message": f"SQL syntax error: {error}"}


def execute_sql_node(state: GraphState) -> GraphState:
    attempt = state.get("retry_count", 0) + 1
    print(f"[execute_sql] Running query (attempt {attempt}) ...")
    rows, error = run_sql(state["generated_sql"])
    if error:
        print(f"[execute_sql] Error: {error}")
        return {**state, "sql_result": [], "error_message": error}
    print(f"[execute_sql] {len(rows)} row(s) returned.")
    return {**state, "sql_result": rows, "error_message": ""}


SQL_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a PostgreSQL query debugger. The query below failed — fix it.

Enhanced question context:
{enhanced_query}

Relevant table schemas:
{relevant_tables}

Original question: {question}

Failing SQL:
{generated_sql}

Error:
{error_message}

Return ONLY the corrected SQL. No explanation, no markdown."""),
    ("human", "Fix the SQL."),
])

def rewrite_sql_node(state: GraphState) -> GraphState:
    retry_count = state.get("retry_count", 0) + 1
    print(f"[rewrite_sql] Retry #{retry_count} ...")
    response = llm.invoke(SQL_REWRITE_PROMPT.format_messages(
        enhanced_query  = state["enhanced_query"],
        relevant_tables = state["relevant_tables"],
        question        = state["question"],
        generated_sql   = state["generated_sql"],
        error_message   = state["error_message"],
    ))
    new_sql = response.content.strip()
    print(f"[rewrite_sql] New SQL:\n{new_sql}")
    return {**state, "generated_sql": new_sql, "retry_count": retry_count}


FORMAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful data analyst. Present the SQL results clearly in plain English.
Format lists neatly. If the query failed after retries, apologize and explain what went wrong."""),
    ("human", """Question: {question}

SQL used:
{generated_sql}

Results (JSON):
{sql_result}

Error (if any): {error_message}
"""),
])

def format_response_node(state: GraphState) -> GraphState:
    print("[format_response] Formatting final answer ...")
    result_text = json.dumps(state.get("sql_result", []), indent=2, default=str)
    if not result_text or result_text == "[]":
        result_text = "No rows returned."
    response = llm.invoke(FORMAT_PROMPT.format_messages(
        question      = state["question"],
        generated_sql = state["generated_sql"],
        sql_result    = result_text,
        error_message = state.get("error_message", ""),
    ))
    print("[format_response] Done.")
    return {**state, "final_answer": response.content}


def route_after_router(state: GraphState) -> str:
    return "greeting" if state["route"] == "greeting" else "text_to_sql"


def route_after_validation(state: GraphState) -> str:
    has_error   = bool(state.get("error_message"))
    retry_count = state.get("retry_count", 0)
    if has_error and retry_count < MAX_RETRIES:
        print(f"[router] Validation failed — retry {retry_count}/{MAX_RETRIES}")
        return "invalid"
    if has_error:
        print("[router] Retries exhausted — surfacing error")
        return "exhausted"
    return "valid"


def route_after_execution(state: GraphState) -> str:
    has_error   = bool(state.get("error_message"))
    retry_count = state.get("retry_count", 0)
    if has_error and retry_count < MAX_RETRIES:
        print(f"[router] Execution error — retry {retry_count}/{MAX_RETRIES}")
        return "retry"
    return "done"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router_node",           router_node)
    graph.add_node("greeting_node",         greeting_node)
    graph.add_node("enhance_query_node",    enhance_query_node)
    graph.add_node("retrieve_context_node", retrieve_context_node)
    graph.add_node("generate_sql_node",     generate_sql_node)
    graph.add_node("validate_sql_node",     validate_sql_node)
    graph.add_node("execute_sql_node",      execute_sql_node)
    graph.add_node("rewrite_sql_node",      rewrite_sql_node)
    graph.add_node("format_response_node",  format_response_node)

    graph.add_edge(START, "router_node")

    graph.add_conditional_edges(
        "router_node",
        route_after_router,
        {"greeting": "greeting_node", "text_to_sql": "enhance_query_node"},
    )

    graph.add_edge("greeting_node",         END)
    graph.add_edge("enhance_query_node",    "retrieve_context_node")
    graph.add_edge("retrieve_context_node", "generate_sql_node")
    graph.add_edge("generate_sql_node",     "validate_sql_node")

    graph.add_conditional_edges(
        "validate_sql_node",
        route_after_validation,
        {"valid": "execute_sql_node", "invalid": "rewrite_sql_node", "exhausted": "format_response_node"},
    )

    graph.add_conditional_edges(
        "execute_sql_node",
        route_after_execution,
        {"retry": "rewrite_sql_node", "done": "format_response_node"},
    )

    graph.add_edge("rewrite_sql_node",     "validate_sql_node")
    graph.add_edge("format_response_node", END)

    return graph.compile()


def main():
    print("=" * 55)
    print("  Praveen Text2SQL  (NeonDB + Pinecone + GPT-4o-mini)")
    print("  Type 'exit' to quit.")
    print("=" * 55)

    app = build_graph()

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        result = app.invoke({
            "question":        question,
            "route":           "",
            "enhanced_query":  "",
            "relevant_tables": "",
            "generated_sql":   "",
            "sql_result":      [],
            "error_message":   "",
            "retry_count":     0,
            "final_answer":    "",
        })

        print(f"\nAssistant: {result['final_answer']}")


if __name__ == "__main__":
    main()
