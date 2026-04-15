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
- **No sub-document retrieval.** The unit of reading is a whole file (capped at 50 KB per file inside `retrieve`). If your files are huge, break them up yourself.
- **Index must fit in the LLM prompt inside `retrieve`.** A few hundred bullets is fine. Tens of thousands is not — this is not a substitute for a vector DB at large scale.
- **Tool-turn chattiness.** `retrieve` is a pure read tool, so the agent spends at least two turns per factual question: one for `INDEX.md`, one for the picked files. That's cheaper than nested LLM calls inside a tool, but each turn is still a round-trip to the LLM provider.
- **Writes are unconditional.** `ingest` overwrites existing files at the same path without prompting. Keep the context directory under version control, and do not point it at anything you are not prepared to let the agent clobber.
- **Freshness is automatic for agent-written files, manual for external edits.** `ingest` refreshes `INDEX.md` after every write. If you edit markdown in your own editor, re-run `write_index` to resync.
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
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from context_the_agent import build_tools

docs = Path("./docs")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = build_tools(docs, llm=llm)             # -> [ingest, ingest_document, retrieve]
agent = create_agent(
    llm,
    tools=tools,
    prompt="You are a docs assistant. Use `retrieve` to answer questions, `ingest` to save notes.",
)

result = agent.invoke({"messages": [("user", "What endpoints exist?")]})
print(result["messages"][-1].content)
```

`INDEX.md` is built automatically the first time `retrieve` is called on an un-indexed corpus (one LLM summary per file, cached). For large pre-existing corpora you can pre-warm it at startup with `write_index(docs, llm=llm)` so the first user question doesn't pay the bootstrap cost.

## Public API

| Symbol | Purpose |
| --- | --- |
| `build_tools(root, llm)` | Returns a three-tool list: `ingest` (write markdown), `ingest_document` (convert PDF/DOCX/PPTX/etc. via MarkItDown + write), and `retrieve` (read markdown files — the tool's docstring teaches the agent to read `INDEX.md` first, then the relevant files). Auto-builds `INDEX.md` on first use. |
| `write_index(root, *, llm=None)` | Optional pre-warm / resync helper — explicitly rebuild `INDEX.md`. Call this after editing markdown files outside the agent. With `llm`, summaries come from the LLM (cached); without, an offline heuristic is used. |

All filesystem operations are guarded against path-traversal — tool calls outside the context root return an `ERROR: ...` string rather than raising.

## How the tools work

**`ingest(filepath, filename, content)`** — writes the markdown file at `<root>/<filepath>/<filename>` (creating parent dirs as needed), then rebuilds `INDEX.md`. Summaries are generated by a single LLM call per file, cached by content hash in `<root>/.context_cache.json` so unchanged files are not re-summarized on later calls.

**`ingest_document(source_path, target_filepath="", target_filename="")`** — same as `ingest`, but takes a path to a non-markdown file (PDF, DOCX, PPTX, XLSX, HTML, image, etc.) and runs it through Microsoft's [MarkItDown](https://github.com/microsoft/markitdown) first. Local, free, no API keys — bundled as a core dependency. If `target_filename` is empty, it's derived from the source's stem (`report.pdf` → `report.md`).

**`retrieve(paths)`** — a pure read tool: given a list of relative `.md` paths, reads each file and returns the concatenated contents wrapped in `=== \`path.md\` ===` headers. No LLM calls inside the tool.

The agent follows a four-step flow taught by the tool's own docstring:
1. Call `retrieve(["INDEX.md"])` first — this is the one-line-per-file summary of the whole corpus.
2. From the summaries, pick the paths likely to answer the question.
3. Call `retrieve([...those paths...])` to read them.
4. Answer using only the returned contents, citing paths in backticks.

This keeps every file the agent reads visible in the outer transcript (full auditability), skips two nested LLM calls per retrieval, and lets the agent adapt (read more files, re-read, follow references between files) instead of being locked into a fixed pipeline.

Commit `.context_cache.json` to version control if you want reproducible summaries across machines; otherwise add it to `.gitignore`.

## Example

See [`examples/demo_agent.py`](examples/demo_agent.py) for a runnable LangGraph ReAct agent wired to these tools.
