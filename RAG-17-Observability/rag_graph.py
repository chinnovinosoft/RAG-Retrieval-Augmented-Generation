import os
from typing import Literal, TypedDict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from pinecone import Pinecone

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END
from langsmith import traceable

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
stats = pc.Index(INDEX_NAME).describe_index_stats()
print(f"[Pinecone] index={INDEX_NAME}  total_vectors={stats['total_vector_count']}  namespaces={list(stats['namespaces'].keys())}")

# No LangSmith imports needed — LangChain components auto-trace when LANGSMITH_TRACING=true
# If using raw OpenAI SDK instead: from langsmith.wrappers import wrap_openai; client = wrap_openai(OpenAI())
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace="default"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


class RAGState(TypedDict):
    question: str
    documents: List[Document]
    grade: str
    answer: str


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, 'no' if not")


GRADE_PROMPT = """You are a relevance grader.
Question: {question}
Retrieved Content: {context}
Is the content relevant to the question? Answer only 'yes' or 'no'."""

ANSWER_PROMPT = """You are a helpful assistant. Use ONLY the provided context.
Question: {question}
Context: {context}
Answer:"""

REWRITE_PROMPT = """Rewrite this question to improve retrieval results.
Question: {question}
Rewritten Question:"""


@traceable(name="fetch_documents")
def fetch_documents(question: str) -> list:
    docs = retriever.invoke(question)
    print(f"[Retriever] fetched {len(docs)} docs")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:100]}...")
    return docs


def retrieve(state: RAGState):
    docs = fetch_documents(state["question"])
    return {"documents": docs}


def grade_documents(
    state: RAGState,
) -> Literal["generate_answer", "rewrite_question"]:
    context = "\n\n".join(doc.page_content for doc in state["documents"])

    result = (
        llm
        .with_structured_output(GradeDocuments)
        .invoke(GRADE_PROMPT.format(question=state["question"], context=context))
    )

    return "generate_answer" if result.binary_score.lower() == "yes" else "rewrite_question"


def rewrite_question(state: RAGState):
    rewritten = llm.invoke(REWRITE_PROMPT.format(question=state["question"]))
    return {"question": rewritten.content}


def generate_answer(state: RAGState):
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    answer = llm.invoke(ANSWER_PROMPT.format(question=state["question"], context=context))
    return {"answer": answer.content}


workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("rewrite_question", rewrite_question)

workflow.add_edge(START, "retrieve")
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("rewrite_question", "retrieve")
workflow.add_edge("generate_answer", END)

graph = workflow.compile()
