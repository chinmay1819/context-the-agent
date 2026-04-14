"""Runnable demo: a LangGraph ReAct agent wired to context-the-agent tools.

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
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from context_the_agent import build_tools, load_index_prompt, write_index

HERE = Path(__file__).resolve().parent

SYSTEM_TEMPLATE = """You are a knowledge agent. Your knowledge lives as markdown files under a context directory.

Start from the INDEX below. When a question needs more detail than the summaries give you, call read_file on the relevant entry. Use glob_files, grep_content, and search_headings for discovery. Use write_file to record notes when the user asks you to.

Rules:
- Prefer answering from files you have actually read; do not guess.
- When citing, reference the file path (e.g. `api.md`).
- Only write files with the .md extension, and keep them under the context directory.

=== INDEX.md ===
{index}
=== END INDEX ==="""


def build_demo_agent():
    """Construct a LangGraph ReAct agent wired to the sample context corpus.

    Loads ``examples/.env`` for credentials, refreshes ``INDEX.md`` via
    :func:`write_index`, embeds it into the system prompt, and returns a
    compiled agent that can answer questions by calling the
    ``context_the_agent`` tools.

    Args:
        (none)

    Returns:
        A compiled LangGraph agent (the return value of
        ``langgraph.prebuilt.create_react_agent``) ready to be invoked with
        ``{"messages": [("user", question)]}``.

    Raises:
        RuntimeError: If ``OPENAI_API_KEY`` is not present in
            ``examples/.env`` or the process environment.
    """
    load_dotenv(HERE / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in examples/.env")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    context_dir = Path(os.environ.get("CONTEXT_DIR", str(HERE / "context"))).resolve()

    write_index(context_dir)
    system = SYSTEM_TEMPLATE.format(index=load_index_prompt(context_dir))

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    tools = build_tools(context_dir, writable=True)
    return create_react_agent(llm, tools=tools, prompt=system)


def _last_ai_text(result) -> str:
    """Extract the final assistant reply from a LangGraph invocation result.

    Handles both plain-string ``content`` and the structured
    ``[{"type": "text", "text": "..."}]`` shape some providers emit.

    Args:
        result: The dict returned by ``agent.invoke(...)`` — expected to
            contain a ``"messages"`` key whose value is a list of
            LangChain message objects.

    Returns:
        The stripped text of the most recent ``AIMessage``, or an empty
        string if no ``AIMessage`` is present in the result.
    """
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                return "".join(parts).strip()
            return str(content).strip()
    return ""


def _ask(agent, question: str) -> str:
    """Send a single user turn to the agent and return the final answer.

    Args:
        agent: A compiled LangGraph agent from :func:`build_demo_agent`.
        question: The user's question as plain text.

    Returns:
        The text of the last ``AIMessage`` produced by the agent.
    """
    result = agent.invoke({"messages": [("user", question)]})
    return _last_ai_text(result)


def main() -> int:
    """CLI entrypoint: one-shot question from argv, or interactive REPL.

    If command-line arguments are supplied, they are joined into a single
    question, answered once, and the process exits. Otherwise a REPL is
    started; ``/exit`` or ``/quit`` (or EOF / Ctrl-C) ends it.

    Args:
        (none — reads ``sys.argv``)

    Returns:
        Process exit code. ``0`` on success.
    """
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
