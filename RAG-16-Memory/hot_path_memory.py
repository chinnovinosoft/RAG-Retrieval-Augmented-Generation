import logging
import os
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME", "company-policy-index"),
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})
checkpointer = InMemorySaver()


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


# RAG tool

@tool
def search_knowledge_base(query: str) -> str:
    """Search the ISO 27001 knowledge base for relevant documents."""
    docs = vectorstore.similarity_search(query, k=5)
    return "\n\n".join(
        f"[{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


# All memory tools — search + update, all three namespaces

tools = [
    search_knowledge_base,

    create_search_memory_tool(namespace=("profile",  "{user_id}"), name="search_user_profile",   store=store),
    create_search_memory_tool(namespace=("semantic", "{user_id}"), name="search_semantic_memory", store=store),
    create_search_memory_tool(namespace=("episodic", "{user_id}"), name="search_episodic_memory", store=store),

    create_manage_memory_tool(namespace=("profile",  "{user_id}"), name="update_user_profile",   schema=UserProfile, store=store),
    create_manage_memory_tool(namespace=("semantic", "{user_id}"), name="update_semantic_memory", schema=Triple,      store=store),
    create_manage_memory_tool(namespace=("episodic", "{user_id}"), name="update_episodic_memory", schema=Episode,     store=store),
]

llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

SYSTEM = SystemMessage(content=(
    "You are an ISO 27001 compliance advisor with a knowledge base and three memory stores.\n\n"
    "Each turn follow this order:\n"
    "1. search_user_profile    — recall who this user is\n"
    "2. search_semantic_memory — recall relevant facts\n"
    "3. search_episodic_memory — recall how similar problems were solved before\n"
    "4. search_knowledge_base  — retrieve ISO 27001 documents from Pinecone\n"
    "5. Answer using all four sources; cite document references inline\n"
    "6. update_user_profile    — save new facts learned about the user\n"
    "7. update_semantic_memory — save new factual triples from this turn\n"
    "8. update_episodic_memory — save this turn as an episode if it solved something well\n\n"
    "IMPORTANT memory write rules:\n"
    "- Use action='create' when no existing record was found in search results.\n"
    "- Use action='update' ONLY when you have an existing memory ID from a prior search result; pass that ID as the 'id' field.\n"
    "- Never call update without an 'id'.\n\n"
    "Reference recalled memories naturally. Do not quote raw memory records."
))


# Graph state and nodes

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent(state: AgentState) -> dict:
    response = llm.invoke([SYSTEM] + state["messages"])
    if response.tool_calls:
        for tc in response.tool_calls:
            log.info(f"Tool call: {tc['name']}  args={tc['args']}")
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    return "tools" if state["messages"][-1].tool_calls else END


# Build graph

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
graph = graph.compile(store=store, checkpointer=checkpointer)


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
    config = {"configurable": {"thread_id": user_id, "user_id": user_id}}
    print(f"\nRAG Agent + Hot-Path Memory | user={user_id} | commands: memory / exit\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "memory":
            show_memory(user_id)
            continue

        result = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
        reply = result["messages"][-1].content
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "user_001")
