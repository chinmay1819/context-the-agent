# context-the-agent

Markdown-as-context tools for LangChain agents. No embeddings, no vector DB — your knowledge base is a directory of `.md` files plus a generated `INDEX.md`, and the agent navigates it with plain filesystem tools.

## Why use this?

Most "chat with your docs" stacks ship a full RAG pipeline: a chunker, an embedding model, a vector database, and a retriever. That is a lot of moving parts for a lot of real projects that only have a few dozen markdown files.

This package takes a different shortcut: **keep the docs as plain `.md` files, let the LLM read them directly.** On startup you generate a short `INDEX.md` listing every file and a one-line summary. You paste that index into the agent's system prompt. The agent then decides which files to open using four small filesystem tools (`read_file`, `glob_files`, `grep_content`, `search_headings`) and optionally a `write_file` tool if you want it to record notes.

You should reach for this package when:

- Your knowledge base is **small to medium** — runbooks, product specs, architecture docs, personal notes, onboarding guides. Anywhere from a handful to a few hundred markdown files.
- You want **no infrastructure** — no vector DB to host, no embedding API to pay for, no re-index job to schedule. `git pull` is your sync.
- You care about **auditability** — every answer the agent gives is traceable to a file path it actually opened. No "the embedding said so."
- You want **easy editing** — update a `.md` file in your editor, re-run `write_index`, done. No re-chunking, no re-embedding.
- You want a **library, not a framework** — you bring your own LLM, your own agent loop, your own prompt. This package ships tools and an index builder; nothing else.

## Scope and limitations

This package is deliberately small. Before adopting it, know what it does **not** do:

- **No semantic search.** If a user asks about "login" and your file is titled "authentication", the match depends on whether the word "login" appears in the file's summary or body. There are no embeddings to bridge vocabulary gaps. Good summaries and good filenames matter a lot.
- **No sub-document retrieval.** The unit of reading is a whole file (capped at 200 KB per `read_file` call). If your files are huge, break them up yourself.
- **Index must fit in the prompt.** You embed `INDEX.md` into the system prompt verbatim. A few hundred bullets is fine. Tens of thousands is not — this is not a substitute for a vector DB at large scale.
- **Freshness is manual.** `write_index` runs when you call it. It does not watch the filesystem. If you add a file and forget to re-ingest, the agent will not know it exists (though `glob_files` may still find it).
- **Writes are unconditional.** With `writable=True`, `write_file` overwrites files without prompting. Keep the context directory under version control, and do not point it at anything you are not prepared to let the agent clobber.
- **Latency cost.** Each tool call is an extra LLM round-trip. A question that needs to read three files takes three extra turns. Fine for a CLI or async workflow, a problem for a sub-second chatbot.
- **LangChain-coupled.** The tools are `langchain_core.tools.BaseTool`. If you are on LlamaIndex, raw OpenAI function-calling, or a non-LangChain agent framework, this package is not a drop-in fit.

If any of those are showstoppers for your use case, you probably want a real RAG stack (LlamaIndex, Haystack, a vector DB). If none of them are, you will likely find this is 5% of the code for 80% of the value.

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
