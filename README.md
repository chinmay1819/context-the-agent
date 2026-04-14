# context-the-agent

Markdown-as-context tools for LangChain agents. No embeddings, no vector DB — your knowledge base is a directory of `.md` files plus a generated `INDEX.md`, and the agent navigates it with plain filesystem tools.

## Install

Install directly from GitHub (pin to a tag for reproducible builds):

```bash
pip install git+https://github.com/chinmay1819/context-the-agent.git@v0.1.0
```

Or track `main`:

```bash
pip install git+https://github.com/chinmay1819/context-the-agent.git
```

In a `requirements.txt`:

```
context-the-agent @ git+https://github.com/chinmay1819/context-the-agent.git@v0.1.0
```

## Usage

```python
from pathlib import Path
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from context_the_agent import build_tools, write_index, load_index_prompt

docs = Path("./docs")

write_index(docs)                              # generates docs/INDEX.md
index = load_index_prompt(docs)                # read back for your system prompt

tools = build_tools(docs)                      # 4 read-only tools
# tools = build_tools(docs, writable=True)     # add write_file (5 tools total)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(
    llm,
    tools=tools,
    prompt=f"You are a docs assistant. Knowledge index:\n{index}",
)

result = agent.invoke({"messages": [("user", "What endpoints exist?")]})
print(result["messages"][-1].content)
```

## Public API

| Symbol | Purpose |
| --- | --- |
| `build_tools(root, *, writable=False)` | Returns LangChain `BaseTool` list: `read_file`, `glob_files`, `grep_content`, `search_headings` (+ `write_file` if `writable=True`). |
| `build_index(root)` | Returns the rendered `INDEX.md` text (pure, no write). |
| `write_index(root)` | Builds the index and writes `root/INDEX.md`. Returns the path. |
| `load_index_prompt(root)` | Reads `root/INDEX.md` and returns its contents. |

All filesystem operations are guarded against path-traversal — tool calls outside the context root return an `ERROR: ...` string rather than raising.

## Ingestion format

`write_index` walks `*.md` under `root` and renders a bullet per file. The summary for each file is:
1. The `summary:` field from YAML frontmatter, if present.
2. Otherwise, the first non-heading paragraph (truncated to 200 chars).

Re-run `write_index` whenever the corpus changes.

## Example

See [`examples/demo_agent.py`](examples/demo_agent.py) for a runnable LangGraph ReAct agent wired to these tools.
