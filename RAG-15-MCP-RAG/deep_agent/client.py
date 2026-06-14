"""
Deep Agent Client
=================
Uses LangChain Deep Agents (`create_deep_agent`) with:
  - Five MCP search tools from the FastMCP RAG server (via langchain-mcp-adapters)
  - Five SKILL.md files that teach the agent HOW to do RAG

Architecture
------------
  MCP Server (FastMCP + Pinecone)
        ↕  Streamable-HTTP / MCP protocol
  MultiServerMCPClient  →  LangChain BaseTool objects
        ↕
  create_deep_agent()   ←  skills/  (SKILL.md files)
        ↕
  FilesystemBackend (reads skills from disk)

The MCP session is held open for the full duration of agent.invoke() so that
all tool calls inside the agent share one connection.

Usage
-----
  # In an async context:
  async with RAGDeepAgentSession() as session:
      answer = await session.run("Compare GraphRAG vs Vector RAG")
      print(answer)

  # Or one-shot:
  answer = await run_query("What is hybrid search?")
  print(answer)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.callbacks import BaseCallbackHandler

try:
    from .config import settings       # when imported as part of the package
except ImportError:
    from config import settings        # when run directly: python client.py

logger = logging.getLogger(__name__)

# ── Skill-usage logger ────────────────────────────────────────────────────────

class SkillUsageLogger(BaseCallbackHandler):
    """
    Intercepts every tool call the agent makes.
    When it calls read_file on a SKILL.md path, we log which skill was activated.
    When it calls one of the MCP search tools, we log the tool + query.
    """

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "unknown")

        if tool_name == "read_file" and "SKILL.md" in input_str:
            # Extract skill name from path  e.g. ".../query-classification/SKILL.md"
            import re
            match = re.search(r"skills[/\\]([\w-]+)[/\\]SKILL\.md", input_str)
            skill = match.group(1) if match else input_str
            logger.info("SKILL ACTIVATED ▶ %s", skill)

        elif tool_name in {"semantic_search", "keyword_search",
                           "hybrid_search", "metadata_search", "multi_search"}:
            logger.info("MCP TOOL CALLED  ▶ %s | input: %.120s", tool_name, input_str)

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent   # RAG-15-MCP-RAG/
_SKILLS_DIR   = Path(__file__).parent / "skills"  # deep_agent/skills/

# ── MCP server config ─────────────────────────────────────────────────────────
_MCP_SERVERS: Dict[str, Any] = {
    "rag-search": {
        "url": settings.mcp_server_url,
        "transport": "streamable_http",
    }
}

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a Deep Research Agent specialised in Retrieval-Augmented Generation (RAG).

You have access to five MCP search tools that query a Pinecone vector database:
  • semantic_search   — dense vector similarity (conceptual queries)
  • keyword_search    — sparse BM25 (exact terms, names, acronyms)
  • hybrid_search     — dense + sparse combined (general default)
  • metadata_search   — filter by document attributes (date, category, author, tag)
  • multi_search      — all strategies in parallel (maximum recall)

You also have five skills that define HOW to use these tools effectively:
  1. query-classification   — classify every query before retrieving
  2. question-decomposition — break complex queries into sub-questions
  3. retrieval-planning     — choose the right tool per sub-question
  4. parallel-retrieval     — call multiple tools concurrently
  5. answer-synthesis       — merge results into one structured answer

Always follow this pipeline:
  classify → [decompose] → plan → retrieve → synthesise

Never skip classification. Never hallucinate facts not in retrieved documents.\
"""


# ── Session class ─────────────────────────────────────────────────────────────

class RAGDeepAgentSession:
    """
    Async context manager that holds an open MCP connection and a ready agent.

    async with RAGDeepAgentSession() as session:
        answer = await session.run("Explain GraphRAG")
    """

    def __init__(self) -> None:
        self._agent = None

    async def __aenter__(self) -> "RAGDeepAgentSession":
        logger.info("Opening MCP session → %s", settings.mcp_server_url)
        self._mcp_cm = MultiServerMCPClient(_MCP_SERVERS)

        tools: List = await self._mcp_cm.get_tools()
        logger.info("MCP tools loaded: %s", [t.name for t in tools])

        backend = FilesystemBackend(root_dir=str(_PROJECT_ROOT), virtual_mode=False)

        # ── Log which skills will be loaded ───────────────────────────────────
        skill_dirs = sorted(Path(_SKILLS_DIR).iterdir()) if Path(_SKILLS_DIR).exists() else []
        loaded_skills = [d.name for d in skill_dirs if (d / "SKILL.md").exists()]
        logger.info("Skills found (%d): %s", len(loaded_skills), loaded_skills)

        # Enable deepagents internal debug logs so skill injection is visible
        logging.getLogger("deepagents").setLevel(logging.DEBUG)

        self._agent = create_deep_agent(
            model=f"{settings.llm_provider}:{settings.llm_model}",
            tools=tools,
            backend=backend,
            skills=[str(_SKILLS_DIR) + "/"],
            system_prompt=_SYSTEM_PROMPT,
        )
        logger.info("Deep agent ready (model=%s:%s) — skills injected into system prompt",
                    settings.llm_provider, settings.llm_model)
        return self

    async def __aexit__(self, *exc_info) -> None:
        logger.info("MCP session closed")

    async def run(self, query: str) -> str:
        """
        Run the deep agent RAG pipeline for a user query.
        Returns the final synthesised answer as a string.
        """
        if self._agent is None:
            raise RuntimeError("Session not started — use `async with RAGDeepAgentSession()`")

        logger.info("Running deep agent | query='%.80s…'", query)
        result = await self._agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"callbacks": [SkillUsageLogger()]},
        )
        last_msg = result["messages"][-1]
        content = last_msg.content

        # content can be a string or a list of content blocks from the API
        if isinstance(content, list):
            answer = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        else:
            answer = str(content).strip()

        logger.info("Deep agent finished | answer_len=%d chars", len(answer))
        return answer

    async def stream(self, query: str):
        """
        Stream the agent's response token-by-token.
        Yields string chunks as they arrive.
        """
        if self._agent is None:
            raise RuntimeError("Session not started — use `async with RAGDeepAgentSession()`")

        logger.info("Streaming deep agent | query='%.80s…'", query)
        async for chunk in self._agent.astream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        yield msg.content


# ── Convenience one-shot function ─────────────────────────────────────────────

async def run_query(query: str) -> str:
    """One-shot helper: opens a session, runs the query, closes the session."""
    async with RAGDeepAgentSession() as session:
        return await session.run(query)


# ── Standalone entry-point  (python client.py) ───────────────────────────────

if __name__ == "__main__":
    import asyncio
    import sys
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(Path(__file__).parent.parent / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    async def _chat() -> None:
        print("\n" + "=" * 60)
        print("  Deep RAG Agent  —  type 'exit' to quit")
        print("=" * 60 + "\n")
        async with RAGDeepAgentSession() as session:
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
                    answer = await session.run(q)
                    print(f"Agent:\n{answer}\n")
                except Exception as exc:
                    print(f"[Error] {exc}\n")

    # Allow an optional one-shot query: python client.py "my question"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        asyncio.run(run_query(query))
    else:
        asyncio.run(_chat())
