import logging
import os
import sys
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langmem import ReflectionExecutor, create_memory_store_manager
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEBOUNCE = int(os.getenv("MEMORY_DEBOUNCE_SECONDS", "10"))

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME", "company-policy-index"),
    embedding=embeddings,
)

store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})


# Memory schemas

class UserProfile(BaseModel):
    name: str | None = None
    role: str | None = Field(None, description="job title or area of responsibility")
    expertise: str | None = Field(None, description="novice / intermediate / expert")
    organization: str | None = None
    preferences: list[str] = Field(default_factory=list)


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    context: str | None = None


class Episode(BaseModel):
    observation: str = Field(description="what situation or question was presented")
    thoughts: str = Field(description="reasoning used to approach it")
    action: str = Field(description="what was done or recommended")
    result: str = Field(description="outcome and whether it worked")


# Background memory managers

profile_manager = create_memory_store_manager(
    "openai:gpt-4o-mini",
    schemas=[UserProfile],
    namespace=("profile", "{user_id}"),
    instructions="Extract or update the user's profile. Never create a second profile.",
    enable_inserts=False,
    store=store,
)

semantic_manager = create_memory_store_manager(
    "openai:gpt-4o-mini",
    schemas=[Triple],
    namespace=("semantic", "{user_id}"),
    instructions="Extract factual triples from the conversation. Include context to prevent ambiguity.",
    enable_inserts=True,
    store=store,
)

episodic_manager = create_memory_store_manager(
    "openai:gpt-4o-mini",
    schemas=[Episode],
    namespace=("episodic", "{user_id}"),
    instructions="If this turn shows a successful explanation or problem-solving, capture it as an episode.",
    enable_inserts=True,
    store=store,
)

profile_executor = ReflectionExecutor(profile_manager, store=store)
semantic_executor = ReflectionExecutor(semantic_manager, store=store)
episodic_executor = ReflectionExecutor(episodic_manager, store=store)


# Graph state

class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    query: str
    memory_context: str
    retrieved_docs: list[dict]


# Graph nodes

def load_memory(state: RAGState) -> dict:
    uid = state["user_id"]
    q = state["query"]
    parts = []

    profile = store.search(("profile", uid), query=q, limit=1)
    semantic = store.search(("semantic", uid), query=q, limit=5)
    episodic = store.search(("episodic", uid), query=q, limit=3)

    if profile:
        parts.append(f"[User Profile]\n{profile[0].value}")

    if semantic:
        facts = "\n".join(
            f"  {i.value.get('subject')} {i.value.get('predicate')} {i.value.get('object')}"
            for i in semantic
        )
        parts.append(f"[Known Facts]\n{facts}")

    if episodic:
        eps = "\n".join(
            f"  Situation: {i.value.get('observation')}\n  How solved: {i.value.get('action')}"
            for i in episodic
        )
        parts.append(f"[Past Episodes]\n{eps}")

    return {"memory_context": "\n\n".join(parts) if parts else "No prior memory."}


def retrieve_docs(state: RAGState) -> dict:
    docs = vectorstore.similarity_search(state["query"], k=5)
    return {
        "retrieved_docs": [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "?"),
                "page": doc.metadata.get("page", 0),
            }
            for doc in docs
        ]
    }


def generate_answer(state: RAGState) -> dict:
    kb = "\n\n".join(
        f"[{d['source']} p.{d['page']}]\n{d['text']}" for d in state["retrieved_docs"]
    )
    system = SystemMessage(content=(
        "You are an ISO 27001 compliance advisor.\n\n"
        f"Memory context (past sessions):\n{state['memory_context']}\n\n"
        f"Knowledge base (retrieved documents):\n{kb}\n\n"
        "Weave memory naturally into your response. Cite document sources inline."
    ))
    return {"messages": [llm.invoke([system] + state["messages"])]}


# Build graph

graph = StateGraph(RAGState)
graph.add_node("load_memory", load_memory)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("generate_answer", generate_answer)
graph.add_edge(START, "load_memory")
graph.add_edge("load_memory", "retrieve_docs")
graph.add_edge("retrieve_docs", "generate_answer")
graph.add_edge("generate_answer", END)
graph = graph.compile()


def schedule_background_extraction(user_id: str, user_msg: str, reply: str) -> None:
    payload = {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
    }
    cfg = {"configurable": {"user_id": user_id}}
    profile_executor.submit(payload, after_seconds=DEBOUNCE, config=cfg)
    semantic_executor.submit(payload, after_seconds=DEBOUNCE, config=cfg)
    episodic_executor.submit(payload, after_seconds=DEBOUNCE, config=cfg)
    log.info(f"Background extraction queued (fires in ~{DEBOUNCE}s, user={user_id})")


def show_memory(user_id: str) -> None:
    for label, ns in [
        ("Profile", ("profile", user_id)),
        ("Semantic", ("semantic", user_id)),
        ("Episodic", ("episodic", user_id)),
    ]:
        items = store.search(ns, limit=20)
        print(f"\n[{label}] ({len(items)} record(s))")
        for item in items:
            print(f"  {item.value}")
    print()


def main(user_id: str = "user_001") -> None:
    print(f"\nRAG Agent + Background Memory | user={user_id} | commands: memory / flush / exit\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "memory":
            show_memory(user_id)
            continue
        if query.lower() == "flush":
            print(f"Waiting {DEBOUNCE + 5}s for background extraction...")
            time.sleep(DEBOUNCE + 5)
            show_memory(user_id)
            continue

        result = graph.invoke({
            "messages": [HumanMessage(content=query)],
            "user_id": user_id,
            "query": query,
            "memory_context": "",
            "retrieved_docs": [],
        })

        reply = result["messages"][-1].content
        print(f"\nAssistant: {reply}\n")

        schedule_background_extraction(user_id, query, reply)
        print(f"  [bg] memory extraction queued (~{DEBOUNCE}s)\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "user_001")
