"""Runnable demo: a LangGraph ReAct agent wired to the two context-the-agent tools.

Usage (from the repo root):

    pip install -e .
    pip install -r examples/requirements.txt
    cp examples/.env.example examples/.env      # fill in OPENAI_API_KEY
    python examples/demo_agent.py "What endpoints exist?"    # one-shot
    python examples/demo_agent.py                             # REPL (/exit to quit)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from context_the_agent import build_tools

HERE = Path(__file__).resolve().parent

SYSTEM_PROMPT = """You are a knowledge agent backed by a markdown corpus.

You have two tools:
- `ingest(filepath, filename, content)` — save new knowledge as a markdown file. Use it whenever the user gives you something to remember or asks you to record notes. Pick short descriptive filenames ending in `.md`.
- `retrieve(query)` — answer a question by consulting the corpus. The tool itself picks relevant files and synthesizes an answer with citations. Prefer it over guessing from your own training data whenever the corpus could plausibly cover the topic.

Rules:
- Never invent answers when retrieve returns a 'no relevant files' message — tell the user plainly.
- When calling retrieve, pass the user's question as-is or lightly rephrased; do not pre-filter.
- When ingesting, choose a filepath like `"notes"` or `""` (root) and a filename like `auth.md`.
"""


def build_demo_agent():
    """Construct a LangGraph ReAct agent wired to the two macro-tools."""
    load_dotenv(HERE / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in examples/.env")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    context_dir = Path(os.environ.get("CONTEXT_DIR", str(HERE / "context"))).resolve()

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    tools = build_tools(context_dir, llm=llm)
    return create_agent(llm, tools=tools, prompt=SYSTEM_PROMPT)


def _last_ai_text(result) -> str:
    """Extract the final assistant reply from a LangGraph invocation result."""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                return "".join(parts).strip()
            return str(content).strip()
    return ""


def _ask(agent, question: str) -> str:
    result = agent.invoke({"messages": [("user", question)]})
    return _last_ai_text(result)


def main() -> int:
    agent = build_demo_agent()
    args = sys.argv[1:]
    if args:
        print(_ask(agent, " ".join(args)))
        return 0
    print("demo-agent REPL — type /exit to quit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question in ("/exit", "/quit"):
            return 0
        print(_ask(agent, question))


if __name__ == "__main__":
    sys.exit(main())
