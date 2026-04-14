"""Public macro-tools exposed to LangChain agents: ``ingest`` and ``retrieve``."""
from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from markitdown import MarkItDown

from ._security import resolve_safe_path
from .ingest import (
    INDEX_FILENAME,
    _message_text,
    build_index,
    load_index,
)

MAX_FILES_PER_RETRIEVE = 8
MAX_FILE_BYTES_PER_RETRIEVE = 50_000

PICK_PROMPT = """You are helping answer a user's question by choosing which markdown files to open from a knowledge corpus.

User query: {query}

Index of available files (path — one-line summary):
{index}

List the file paths (exactly as they appear in the index, without backticks) that are most likely to contain the answer. One path per line. At most {max_files}. If no file looks relevant, output nothing. No explanations, no bullets."""

ANSWER_PROMPT = """Answer the user's query using ONLY the markdown file contents below. Cite the file path in backticks (e.g. `api.md`) whenever you draw on a file. If the provided files do not answer the query, say so plainly — do not fall back on general knowledge.

User query: {query}

Files:
{files}

Answer:"""


def build_tools(context_root: Path, llm: BaseChatModel) -> list[BaseTool]:
    """Build the macro-tool set bound to a markdown context directory.

    All tools are closures over ``context_root`` and ``llm``. Every
    filesystem operation against the corpus is guarded against path
    traversal so the agent cannot escape the directory.

    Args:
        context_root: Directory the tools may read and write within. The
            path is ``.resolve()``-d once at build time.
        llm: A LangChain chat model (e.g. ``ChatOpenAI``). Used internally
            to generate file summaries (``ingest`` / ``ingest_document``)
            and to select-then-synthesize answers (``retrieve``).

    Returns:
        A list of three ``langchain_core.tools.BaseTool`` instances:
        ``ingest`` (write markdown content the agent already has),
        ``ingest_document`` (convert a PDF/DOCX/PPTX/etc. to markdown via
        MarkItDown and write it), and ``retrieve`` (answer
        natural-language questions from the corpus).
    """
    root = context_root.resolve()

    def _save_markdown(filepath: str, filename: str, content: str, *, verb: str) -> str:
        """Shared write + index-refresh used by ingest and ingest_document."""
        if not filename.endswith(".md"):
            return "ERROR: filename must end in .md"
        rel_dir = filepath.strip("/")
        rel = f"{rel_dir}/{filename}" if rel_dir else filename
        try:
            p = resolve_safe_path(rel, root)
        except ValueError as e:
            return f"ERROR: {e}"
        data = content.encode("utf-8")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        except OSError as e:
            return f"ERROR: could not write {rel}: {e}"
        try:
            (root / INDEX_FILENAME).write_text(
                build_index(root, llm=llm), encoding="utf-8"
            )
        except OSError as e:
            return f"{verb} {len(data)} bytes to {rel} (index update failed: {e})"
        return f"{verb} {len(data)} bytes to {rel} and updated INDEX.md"

    @tool
    def ingest(filepath: str, filename: str, content: str) -> str:
        """Write a markdown file into the corpus and refresh INDEX.md with a summary.

        Call this whenever the user asks you to save, record, or remember
        new information. An existing file at the target path is
        overwritten. After the write, ``INDEX.md`` is regenerated so
        subsequent ``retrieve`` calls can find the new file.

        Args:
            filepath: Directory (relative to the context root) to write
                into. Use ``""`` to write at the root itself. Intermediate
                directories are created as needed.
            filename: The file name. Must end in ``".md"``.
            content: Full UTF-8 markdown body to write.

        Returns:
            A status string such as ``"wrote N bytes to <rel> and
            updated INDEX.md"``, or a string beginning with ``"ERROR: "``
            if the write was rejected (bad extension, path escape, OS
            error).
        """
        return _save_markdown(filepath, filename, content, verb="wrote")

    @tool
    def ingest_document(
        source_path: str,
        target_filepath: str = "",
        target_filename: str = "",
    ) -> str:
        """Convert an external document (PDF, DOCX, PPTX, XLSX, HTML, image, etc.)
        to markdown and save it into the corpus.

        Uses Microsoft's ``MarkItDown`` (local, free) under the hood.

        Call this whenever the user points at a non-markdown file they want
        remembered (a PDF, a Word doc, a slide deck, a spreadsheet). After
        conversion, the flow is identical to ``ingest`` — file is written to
        the corpus and ``INDEX.md`` is refreshed.

        Args:
            source_path: Absolute or user-relative path to the file on disk.
                Supports formats MarkItDown handles (PDF, DOCX, PPTX, XLSX,
                HTML, images, and more).
            target_filepath: Directory under the context root to save into.
                ``""`` writes at the root. Intermediate directories are
                created as needed.
            target_filename: Markdown file name (must end in ``".md"``). If
                left empty, derived from the source file's stem
                (e.g. ``report.pdf`` → ``report.md``).

        Returns:
            A status string such as ``"converted N bytes to <rel> and
            updated INDEX.md"``, or a string beginning with ``"ERROR: "``
            if the source is unreadable, conversion fails, or the write is
            rejected.
        """
        src = Path(source_path).expanduser()
        if not src.exists() or not src.is_file():
            return f"ERROR: source file not found: {source_path}"

        filename = target_filename or f"{src.stem}.md"
        try:
            result = MarkItDown().convert(str(src))
            content = result.text_content or ""
        except Exception as e:
            return f"ERROR: conversion failed: {e}"
        if not content.strip():
            return f"ERROR: conversion produced empty markdown for {source_path}"

        return _save_markdown(target_filepath, filename, content, verb="converted")

    @tool
    def retrieve(query: str) -> str:
        """Answer a natural-language question using the markdown corpus.

        Looks at ``INDEX.md``, asks the LLM which files are relevant,
        reads them, and asks the LLM to synthesize a natural-language
        answer with file-path citations.

        Prefer this tool over answering from prior knowledge whenever the
        corpus could plausibly cover the topic.

        Args:
            query: The user's question in plain English.

        Returns:
            A synthesized answer string (with ``path.md`` citations in
            backticks), a friendly "no relevant files" message, or a
            string beginning with ``"ERROR: "`` if the LLM call fails.
            If ``INDEX.md`` does not exist yet, it is built automatically
            on first use.
        """
        index_text = _ensure_index(root, llm)

        pick_prompt = PICK_PROMPT.format(
            query=query, index=index_text, max_files=MAX_FILES_PER_RETRIEVE
        )
        try:
            picked_msg = llm.invoke(pick_prompt)
        except Exception as e:
            return f"ERROR: LLM call failed during file selection: {e}"
        candidates = _parse_paths(_message_text(picked_msg), root)
        if not candidates:
            return (
                "I could not find any relevant files in the corpus for that question."
            )

        file_blocks: list[str] = []
        for rel in candidates[:MAX_FILES_PER_RETRIEVE]:
            try:
                p = resolve_safe_path(rel, root)
            except ValueError:
                continue
            if not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text) > MAX_FILE_BYTES_PER_RETRIEVE:
                text = text[:MAX_FILE_BYTES_PER_RETRIEVE] + "\n[...truncated]"
            file_blocks.append(f"=== `{rel}` ===\n{text}")

        if not file_blocks:
            return "Picked files were unreadable or outside the context root."

        answer_prompt = ANSWER_PROMPT.format(
            query=query, files="\n\n".join(file_blocks)
        )
        try:
            answer_msg = llm.invoke(answer_prompt)
        except Exception as e:
            return f"ERROR: LLM call failed during answer synthesis: {e}"
        answer = _message_text(answer_msg).strip()
        return answer or "(no answer produced)"

    return [ingest, ingest_document, retrieve]


def _ensure_index(root: Path, llm: BaseChatModel) -> str:
    """Return ``INDEX.md`` text, building and persisting it lazily if missing.

    Called on every ``retrieve`` invocation. The first call on an
    un-indexed corpus pays the bootstrap cost (one LLM summary per
    ``*.md`` file, cached by content hash); subsequent calls just read
    the existing file.
    """
    try:
        return load_index(root)
    except FileNotFoundError:
        text = build_index(root, llm=llm)
        try:
            (root / INDEX_FILENAME).write_text(text, encoding="utf-8")
        except OSError:
            pass
        return text


def _parse_paths(text: str, root: Path) -> list[str]:
    """Pull file paths out of LLM output; drop anything that isn't a real .md file under root."""
    paths: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip().lstrip("-•*").strip().strip("`").strip()
        if not line or line in seen:
            continue
        try:
            p = resolve_safe_path(line, root)
        except ValueError:
            continue
        if p.is_file() and p.suffix == ".md":
            seen.add(line)
            paths.append(line)
    return paths
