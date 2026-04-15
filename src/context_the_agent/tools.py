"""Public macro-tools exposed to LangChain agents: ``ingest`` and ``retrieve``."""
from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from markitdown import MarkItDown

from ._security import resolve_safe_path
from .ingest import (
    INDEX_FILENAME,
    build_index,
    load_index,
)

MAX_FILE_BYTES_PER_RETRIEVE = 50_000


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
    def retrieve(paths: list[str]) -> str:
        """Read one or more markdown files from the knowledge corpus.

        REQUIRED USAGE FLOW — follow these steps in order every time you
        need to answer a factual question from this corpus:

        Step 1 — Always call ``retrieve(paths=["INDEX.md"])`` FIRST on any
        new question. ``INDEX.md`` is a one-line-per-file summary of every
        markdown file in the corpus; it tells you which files exist and
        what each one covers. Skipping this step is not allowed — you
        cannot know what the corpus contains without reading the index.

        Step 2 — Read the summaries in ``INDEX.md`` and pick the file
        paths most likely to contain the answer. Copy the paths exactly
        as they appear in the index.

        Step 3 — Call ``retrieve(paths=[...])`` again with the picked
        paths. You can pass multiple paths in one call to save round
        trips (e.g. ``retrieve(paths=["api.md", "notes/auth.md"])``).

        Step 4 — Write your answer using ONLY the file contents returned.
        Cite file paths in backticks (e.g. `api.md`) whenever you draw on
        a file. If none of the files answer the user's question, say so
        plainly — do not fall back on prior knowledge, do not guess.

        If the user's question shifts to a different topic mid-conversation,
        or you have not consulted the index yet in this session, re-read
        ``INDEX.md`` before answering.

        Args:
            paths: Relative paths to markdown files under the context
                root, e.g. ``["INDEX.md"]`` or
                ``["api.md", "notes/auth.md"]``. Each path must end in
                ``.md``. Paths that escape the root, don't exist, or fail
                to read are reported as ``=== ERROR: ... ===`` sections
                inline — other files in the same call still return
                normally.

        Returns:
            The concatenated contents of the requested files, each
            wrapped in a ``=== `rel/path.md` ===`` header. Files larger
            than 50 KB are truncated with a ``[...truncated]`` marker.
            Call with ``paths=["INDEX.md"]`` first if you have not yet
            seen the corpus index.
        """
        _ensure_index(root, llm)
        if not paths:
            return "=== ERROR: no paths provided — pass at least one, e.g. ['INDEX.md'] ==="

        blocks: list[str] = []
        for rel in paths:
            if not rel.endswith(".md"):
                blocks.append(f"=== ERROR: path must end in .md: {rel} ===")
                continue
            try:
                p = resolve_safe_path(rel, root)
            except ValueError as e:
                blocks.append(f"=== ERROR: {e} ===")
                continue
            if not p.is_file():
                blocks.append(f"=== ERROR: not a file: {rel} ===")
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                blocks.append(f"=== ERROR: could not read {rel}: {e} ===")
                continue
            if len(text) > MAX_FILE_BYTES_PER_RETRIEVE:
                text = text[:MAX_FILE_BYTES_PER_RETRIEVE] + "\n[...truncated]"
            blocks.append(f"=== `{rel}` ===\n{text}")
        return "\n\n".join(blocks)

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


