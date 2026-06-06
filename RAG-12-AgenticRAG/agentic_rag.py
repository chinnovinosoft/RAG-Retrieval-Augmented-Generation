from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from supabase import create_client
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import psycopg
from psycopg.rows import dict_row
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = PineconeVectorStore(
    index_name="company-policy-index",
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

class GraphState(TypedDict):
    question: str
    datasource: str
    documents: list
    answer: str
    grade: str

class Route(BaseModel):
    datasource: Literal["sql", "rag"]

class Grade(BaseModel):
    binary_score: Literal["yes", "no"]

def router(state: GraphState):
    structured_llm = llm.with_structured_output(Route)

    result = structured_llm.invoke(
        f"""
        Route the question.

        Use sql when question is about:
        employees,
        leave balances,
        insurance coverage.

        Use rag when question is about:
        company policy,
        leave policy,
        insurance policy,
        travel policy,
        reimbursement policy,
        employee handbook.

        Question:
        {state["question"]}
        """
    )

    return {"datasource": result.datasource}

def route_decision(state: GraphState):
    return state["datasource"]


class SQLQuery(BaseModel):
    query: str


def sql_node(state: GraphState):

    schema = """
    Table: employees

    employee_id INTEGER PRIMARY KEY
    employee_code VARCHAR
    full_name VARCHAR
    email VARCHAR
    department VARCHAR
    designation VARCHAR
    manager_name VARCHAR
    joining_date DATE
    employment_type VARCHAR
    location VARCHAR
    annual_salary NUMERIC
    performance_rating NUMERIC
    created_at TIMESTAMP

    Table: leave_balances

    id INTEGER PRIMARY KEY
    employee_id INTEGER
    annual_leave_balance INTEGER
    sick_leave_balance INTEGER
    casual_leave_balance INTEGER
    carry_forward_leave INTEGER
    last_updated TIMESTAMP

    Table: insurance_coverage

    coverage_id INTEGER PRIMARY KEY
    employee_id INTEGER
    spouse_covered BOOLEAN
    children_covered BOOLEAN
    parents_covered BOOLEAN
    coverage_amount NUMERIC

    Relationships:

    leave_balances.employee_id
        -> employees.employee_id

    insurance_coverage.employee_id
        -> employees.employee_id
    """

    DB_URI = "postgresql://neondb_owner:npg_WtlAgc2Jfe0q@ep-plain-truth-apf91b9h-pooler.c-7.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

    sql_generator = llm.with_structured_output(SQLQuery)

    result = sql_generator.invoke(
        f"""
        You are an expert PostgreSQL query generator.

        Generate ONLY a valid PostgreSQL SELECT query.

        Database Schema:

        {schema}

        Rules:
        - Use only the tables and columns provided.
        - Never invent tables or columns.
        - Use joins when needed.
        - Return ONLY SQL.
        - Generate SELECT queries only.
        - Do not use markdown.
        - Do not explain anything.
        - Employee ID is 1 unless the user explicitly specifies another employee.

        User Question:
        {state["question"]}
        """
    )

    generated_sql = result.query.strip().replace("```sql", "").replace("```", "").strip()

    sql_lower = generated_sql.lower()

    forbidden_keywords = [
        "drop ",
        "delete ",
        "truncate ",
        "alter ",
        "update ",
        "insert ",
        "grant ",
        "revoke ",
        "create "
    ]

    if not sql_lower.startswith("select"):
        return {
            "answer": {
                "error": "Only SELECT queries are allowed",
                "generated_sql": generated_sql
            }
        }

    for keyword in forbidden_keywords:
        if keyword in sql_lower:
            return {
                "answer": {
                    "error": f"Unsafe SQL detected: {keyword}",
                    "generated_sql": generated_sql
                }
            }

    try:

        with psycopg.connect(
            DB_URI,
            row_factory=dict_row
        ) as conn:

            with conn.cursor() as cur:

                cur.execute(generated_sql)

                rows = cur.fetchall()

        if not rows:
            return {
                "answer": {
                    "generated_sql": generated_sql,
                    "results": [],
                    "message": "No records found"
                }
            }

        answer_prompt = f"""
        User Question:
        {state["question"]}

        SQL Executed:
        {generated_sql}

        Database Results:
        {rows}

        Provide a concise HR assistant response.
        """

        final_answer = llm.invoke(answer_prompt)

        return {
            "answer": final_answer.content,
            "generated_sql": generated_sql,
            "sql_results": rows
        }

    except Exception as e:

        return {
            "answer": {
                "generated_sql": generated_sql,
                "error": str(e)
            }
        }

def retrieve(state: GraphState):

    docs = retriever.invoke(
        state["question"]
    )

    return {"documents": docs}

def grade_documents(state: GraphState):

    grader = llm.with_structured_output(
        Grade
    )

    result = grader.invoke(
        f"""
        Question:
        {state["question"]}

        Documents:
        {state["documents"]}

        Are these documents relevant enough
        to answer the question?

        Return yes or no.
        """
    )

    return {"grade": result.binary_score}

def grade_decision(state: GraphState):

    if state["grade"] == "yes":
        return "generate"

    return "rewrite"

def rewrite_question(state: GraphState):

    response = llm.invoke(
        f"""
        Rewrite the query to improve retrieval.

        Query:
        {state["question"]}
        """
    )

    return {
        "question": response.content
    }

def generate(state: GraphState):

    response = llm.invoke(
        f"""
        Use only the context.

        Context:
        {state["documents"]}

        Question:
        {state["question"]}
        """
    )

    return {
        "answer": response.content
    }

workflow = StateGraph(GraphState)

workflow.add_node("router", router)
workflow.add_node("sql", sql_node)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_question)
workflow.add_node("generate", generate)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "sql": "sql",
        "rag": "retrieve"
    }
)

workflow.add_edge(
    "retrieve",
    "grade"
)

workflow.add_conditional_edges(
    "grade",
    grade_decision,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

workflow.add_edge(
    "rewrite",
    "retrieve"
)

workflow.add_edge(
    "generate",
    END
)

workflow.add_edge(
    "sql",
    END
)

graph = workflow.compile()

result = graph.invoke(
    {
        "question": "tell me about the attendence details policy?"
    }
)

print(result["answer"])