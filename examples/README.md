# demo_agent

Minimal LangGraph ReAct agent wired to `context-the-agent`'s tools and markdown index.

## Run

```bash
pip install -e ..
pip install -r requirements.txt
cp .env.example .env    # fill in OPENAI_API_KEY
python demo_agent.py "What endpoints exist?"
```

`demo_agent.py` calls `write_index()` on startup to keep `context/INDEX.md` fresh, loads it into the system prompt via `load_index_prompt()`, and builds a writable tool set via `build_tools(context_dir, writable=True)`.
