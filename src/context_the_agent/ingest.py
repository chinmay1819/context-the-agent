"""Internal INDEX.md management: walk the corpus, summarize files, render the index.

Only :func:`write_index` is re-exported from the package. Everything else is
private helpers used by the :mod:`context_the_agent.tools` macro-tools.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
SUMMARY_RE = re.compile(r"^summary:\s*(.+)$", re.MULTILINE)
SUMMARY_MAX = 200
SUMMARY_TRUNCATE = 8_000
INDEX_FILENAME = "INDEX.md"
CACHE_FILENAME = ".context_cache.json"

SUMMARIZE_PROMPT = """Summarize this markdown file in ONE sentence (max ~25 words) for an index that helps an agent decide whether to read it.

File path: `{path}`
File contents:
{content}

Output only the summary sentence. No preamble, no quotes, no bullet markers."""


def _heuristic_summary(text: str) -> str:
    """Offline summary: YAML frontmatter ``summary:`` or first non-heading paragraph."""
    body = text
    m = FRONTMATTER_RE.match(text)
    if m:
        fm = m.group(1)
        s = SUMMARY_RE.search(fm)
        if s:
            return s.group(1).strip().strip("\"'")
        body = text[m.end():]
    for raw in body.split("\n\n"):
        para = raw.strip()
        if not para or para.startswith("#"):
            continue
        para = " ".join(para.splitlines()).strip()
        if len(para) > SUMMARY_MAX:
            para = para[: SUMMARY_MAX - 1].rstrip() + "…"
        return para
    return "(no summary)"


def _message_text(msg: BaseMessage | Any) -> str:
    """Extract plain text from a ``BaseMessage``, handling list content shapes."""
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )
    return str(content)


def _llm_summary(llm: BaseChatModel, rel: str, content: str) -> str | None:
    """One LLM call → one-line summary. Returns None on any failure."""
    try:
        prompt = SUMMARIZE_PROMPT.format(
            path=rel, content=content[:SUMMARY_TRUNCATE]
        )
        result = llm.invoke(prompt)
    except Exception:
        return None
    text = _message_text(result).strip()
    if not text:
        return None
    return text.splitlines()[0].strip() or None


def _load_cache(root: Path) -> dict:
    path = root / CACHE_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(root: Path, cache: dict) -> None:
    """Best-effort cache write — failures are swallowed."""
    try:
        (root / CACHE_FILENAME).write_text(
            json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8"
        )
    except OSError:
        pass


def _render_index(entries: list[tuple[str, str]]) -> str:
    lines = ["# Context Index", ""]
    if not entries:
        lines.append("_(no markdown files found)_")
    else:
        for rel, summary in entries:
            lines.append(f"- `{rel}` — {summary}")
    lines.append("")
    return "\n".join(lines)


def build_index(context_dir: Path, *, llm: BaseChatModel | None = None) -> str:
    """Render the ``INDEX.md`` text for a context directory without writing it.

    Args:
        context_dir: Directory to scan. Every ``*.md`` under it (except
            ``INDEX.md`` itself) becomes one bullet.
        llm: Optional LangChain chat model. When provided, each file is
            summarized by a single LLM call (not an agent loop), and the
            results are cached by SHA-256 in ``<context_dir>/.context_cache.json``
            so unchanged files don't re-call the LLM on subsequent runs.
            When ``None``, the offline heuristic is used.

    Returns:
        The complete ``INDEX.md`` text, starting with ``# Context Index``.
    """
    root = Path(context_dir)
    cache = _load_cache(root) if llm is not None else {}
    new_cache: dict = {}
    entries: list[tuple[str, str]] = []

    for p in sorted(root.rglob("*.md")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.name == INDEX_FILENAME:
            continue
        relstr = str(rel)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            entries.append((relstr, f"(unreadable: {e})"))
            continue

        heuristic = _heuristic_summary(text)
        if llm is not None:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            prev = cache.get(relstr)
            if (
                isinstance(prev, dict)
                and prev.get("hash") == digest
                and prev.get("summary")
            ):
                summary = prev["summary"]
            else:
                llm_summary = _llm_summary(llm, relstr, text)
                summary = llm_summary if llm_summary else heuristic
            new_cache[relstr] = {"hash": digest, "summary": summary}
        else:
            summary = heuristic
        entries.append((relstr, summary))

    if llm is not None:
        _save_cache(root, new_cache)
    return _render_index(entries)


def write_index(context_dir: Path, *, llm: BaseChatModel | None = None) -> Path:
    """Build the index for ``context_dir`` and write it to ``INDEX.md``.

    Use this once at startup to bootstrap an existing markdown corpus so
    the :func:`~context_the_agent.build_tools` ``retrieve`` tool has an
    index to consult. After that, the ``ingest`` tool keeps ``INDEX.md``
    current whenever the agent writes a new file.

    Args:
        context_dir: Directory containing the markdown corpus. Must exist.
        llm: Optional LangChain chat model. When provided, summaries come
            from the LLM (with hash-based caching); otherwise the offline
            heuristic is used.

    Returns:
        The path of the written ``INDEX.md``.

    Raises:
        FileNotFoundError: If ``context_dir`` does not exist.
    """
    root = Path(context_dir)
    if not root.exists():
        raise FileNotFoundError(f"context directory does not exist: {root}")
    out = root / INDEX_FILENAME
    out.write_text(build_index(root, llm=llm), encoding="utf-8")
    return out


def load_index(context_dir: Path) -> str:
    """Read the existing ``INDEX.md`` — raises if missing."""
    path = Path(context_dir) / INDEX_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Call write_index({context_dir!r}) first."
        )
    return path.read_text(encoding="utf-8")
