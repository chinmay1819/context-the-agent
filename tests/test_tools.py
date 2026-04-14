from pathlib import Path
from typing import Callable

import pytest
from langchain_core.messages import AIMessage

from context_the_agent import build_tools, write_index


class _ScriptedLLM:
    """Minimal stand-in for a LangChain chat model.

    ``responder`` receives the raw prompt string and returns the text the
    fake LLM should emit. Calls are recorded for assertions.
    """

    def __init__(self, responder: Callable[[str], str]) -> None:
        self._responder = responder
        self.calls: list[str] = []

    def invoke(self, prompt):
        if isinstance(prompt, str):
            text = prompt
        else:
            text = str(prompt)
        self.calls.append(text)
        return AIMessage(content=self._responder(text))


def _summary_llm(summary: str = "auto summary for {path}") -> _ScriptedLLM:
    """LLM that always emits a summary sentence for _llm_summary prompts."""
    def _respond(prompt: str) -> str:
        # Heuristic: find the `{path}` line in the SUMMARIZE_PROMPT format.
        for line in prompt.splitlines():
            if line.startswith("File path:"):
                path = line.split("`")[1] if "`" in line else "?"
                return summary.format(path=path)
        return "generic summary"
    return _ScriptedLLM(_respond)


def _by_name(tools):
    return {t.name: t for t in tools}


def test_build_tools_returns_three_tools(tmp_path: Path) -> None:
    tools = build_tools(tmp_path, llm=_summary_llm())
    assert [t.name for t in tools] == ["ingest", "ingest_document", "retrieve"]


def test_ingest_writes_file_and_refreshes_index(tmp_path: Path) -> None:
    ingest = _by_name(build_tools(tmp_path, llm=_summary_llm()))["ingest"]
    out = ingest.invoke(
        {"filepath": "", "filename": "auth.md", "content": "# Auth\n\nJWT only.\n"}
    )
    assert out.startswith("wrote ")
    assert "updated INDEX.md" in out
    assert (tmp_path / "auth.md").read_text().startswith("# Auth")
    assert "`auth.md`" in (tmp_path / "INDEX.md").read_text()


def test_ingest_creates_nested_dirs(tmp_path: Path) -> None:
    ingest = _by_name(build_tools(tmp_path, llm=_summary_llm()))["ingest"]
    ingest.invoke(
        {"filepath": "notes/2026", "filename": "q2.md", "content": "quarterly"}
    )
    assert (tmp_path / "notes" / "2026" / "q2.md").read_text() == "quarterly"


def test_ingest_rejects_non_markdown(tmp_path: Path) -> None:
    ingest = _by_name(build_tools(tmp_path, llm=_summary_llm()))["ingest"]
    out = ingest.invoke({"filepath": "", "filename": "evil.sh", "content": "x"})
    assert out.startswith("ERROR: filename must end in .md")


def test_ingest_rejects_traversal(tmp_path: Path) -> None:
    ingest = _by_name(build_tools(tmp_path, llm=_summary_llm()))["ingest"]
    out = ingest.invoke({"filepath": "../..", "filename": "pwn.md", "content": "x"})
    assert out.startswith("ERROR:")


def test_ingest_os_error_on_write(tmp_path: Path, monkeypatch) -> None:
    ingest = _by_name(build_tools(tmp_path, llm=_summary_llm()))["ingest"]

    def _boom(self, *a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", _boom)
    out = ingest.invoke({"filepath": "", "filename": "a.md", "content": "x"})
    assert out.startswith("ERROR: could not write")


def test_retrieve_auto_bootstraps_index(tmp_path: Path) -> None:
    """retrieve should build INDEX.md on first call if missing."""
    (tmp_path / "api.md").write_text("# API\n\nPOST /jobs.\n")

    def _respond(prompt: str) -> str:
        # Summarize call (during auto-bootstrap)
        if prompt.startswith("Summarize this markdown file"):
            return "api summary"
        # Pick call
        if "choosing which markdown files" in prompt:
            return "api.md"
        # Answer call
        return "POST /jobs (`api.md`)"

    retrieve = _by_name(build_tools(tmp_path, llm=_ScriptedLLM(_respond)))["retrieve"]
    out = retrieve.invoke({"query": "endpoints?"})
    assert "POST /jobs" in out
    assert (tmp_path / "INDEX.md").exists()


def test_retrieve_empty_corpus_auto_bootstrap(tmp_path: Path) -> None:
    """retrieve on an empty dir should bootstrap an empty index and report no matches."""
    def _respond(prompt: str) -> str:
        if "choosing which markdown files" in prompt:
            return ""
        return ""

    retrieve = _by_name(build_tools(tmp_path, llm=_ScriptedLLM(_respond)))["retrieve"]
    out = retrieve.invoke({"query": "anything"})
    assert "could not find any relevant files" in out.lower()
    assert (tmp_path / "INDEX.md").exists()


def test_retrieve_no_relevant_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    write_index(tmp_path)  # heuristic, no llm needed

    # LLM returns empty pick -> "no relevant files" message.
    def _respond(prompt: str) -> str:
        if "User query:" in prompt and "Index of available files" in prompt:
            return ""
        return ""

    retrieve = _by_name(build_tools(tmp_path, llm=_ScriptedLLM(_respond)))["retrieve"]
    out = retrieve.invoke({"query": "tell me about zebras"})
    assert "could not find any relevant files" in out.lower()


def test_retrieve_happy_path(tmp_path: Path) -> None:
    (tmp_path / "api.md").write_text("# API\n\nPOST /jobs creates a job.\n")
    (tmp_path / "misc.md").write_text("# Misc\n\nUnrelated.\n")
    write_index(tmp_path)

    def _respond(prompt: str) -> str:
        if "Answer the user's query" in prompt:
            assert "POST /jobs" in prompt  # chosen file was included
            return "The API exposes POST /jobs (`api.md`)."
        if "choosing which markdown files" in prompt:
            return "api.md\n"
        return ""

    llm = _ScriptedLLM(_respond)
    retrieve = _by_name(build_tools(tmp_path, llm=llm))["retrieve"]
    out = retrieve.invoke({"query": "what endpoints exist?"})
    assert "POST /jobs" in out
    assert "`api.md`" in out
    # Two LLM calls: pick, then synthesize.
    assert len(llm.calls) == 2


def test_retrieve_strips_bullets_and_backticks_from_picks(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text("# X\n\nContent.\n")
    write_index(tmp_path)

    def _respond(prompt: str) -> str:
        if "Answer the user's query" in prompt:
            return "answer"
        return "- `x.md`\n* `nonexistent.md`\n"

    retrieve = _by_name(build_tools(tmp_path, llm=_ScriptedLLM(_respond)))["retrieve"]
    out = retrieve.invoke({"query": "x"})
    assert out == "answer"


def test_retrieve_llm_failure_during_pick(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n")
    write_index(tmp_path)

    def _boom(prompt):
        raise RuntimeError("API down")

    class _ExplodingLLM:
        calls: list = []
        def invoke(self, prompt):
            raise RuntimeError("API down")

    retrieve = _by_name(build_tools(tmp_path, llm=_ExplodingLLM()))["retrieve"]
    out = retrieve.invoke({"query": "x"})
    assert out.startswith("ERROR:") and "file selection" in out


class _FakeMarkItDown:
    """Stand-in for markitdown.MarkItDown."""

    class _Result:
        def __init__(self, text: str) -> None:
            self.text_content = text

    def __init__(self, text_by_suffix: dict | None = None) -> None:
        self._by_suffix = text_by_suffix or {}

    def convert(self, src: str):
        suffix = Path(src).suffix.lower()
        if suffix in self._by_suffix:
            return self._Result(self._by_suffix[suffix])
        return self._Result(f"# Converted {Path(src).name}\n\nBody from {suffix}.\n")


def _install_fake_markitdown(monkeypatch, fake=None) -> None:
    """Patch ``tools.MarkItDown`` so ``ingest_document`` uses a stub converter."""
    from context_the_agent import tools as tools_mod

    monkeypatch.setattr(
        tools_mod, "MarkItDown", lambda: (fake or _FakeMarkItDown())
    )


def test_ingest_document_missing_source(tmp_path: Path, monkeypatch) -> None:
    _install_fake_markitdown(monkeypatch)
    tools = _by_name(build_tools(tmp_path, llm=_summary_llm()))
    out = tools["ingest_document"].invoke(
        {"source_path": str(tmp_path / "nope.pdf")}
    )
    assert out.startswith("ERROR: source file not found")


def test_ingest_document_happy_path(tmp_path: Path, monkeypatch) -> None:
    _install_fake_markitdown(monkeypatch)
    src = tmp_path / "report.pdf"
    src.write_bytes(b"%PDF-fake")

    tools = _by_name(build_tools(tmp_path, llm=_summary_llm()))
    out = tools["ingest_document"].invoke({"source_path": str(src)})
    assert out.startswith("converted ")
    assert "updated INDEX.md" in out
    # Derived filename: report.pdf -> report.md
    written = tmp_path / "report.md"
    assert written.exists()
    assert "Converted report.pdf" in written.read_text()
    assert "`report.md`" in (tmp_path / "INDEX.md").read_text()


def test_ingest_document_explicit_target(tmp_path: Path, monkeypatch) -> None:
    _install_fake_markitdown(monkeypatch)
    src = tmp_path / "report.pdf"
    src.write_bytes(b"%PDF-fake")

    tools = _by_name(build_tools(tmp_path, llm=_summary_llm()))
    out = tools["ingest_document"].invoke(
        {
            "source_path": str(src),
            "target_filepath": "docs",
            "target_filename": "q2.md",
        }
    )
    assert out.startswith("converted ")
    assert (tmp_path / "docs" / "q2.md").exists()


def test_ingest_document_rejects_non_markdown_target(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_markitdown(monkeypatch)
    src = tmp_path / "x.pdf"
    src.write_bytes(b"x")
    tools = _by_name(build_tools(tmp_path, llm=_summary_llm()))
    out = tools["ingest_document"].invoke(
        {"source_path": str(src), "target_filename": "bad.txt"}
    )
    assert out.startswith("ERROR: filename must end in .md")


def test_ingest_document_conversion_failure(tmp_path: Path, monkeypatch) -> None:
    class _Boom:
        def convert(self, src: str):
            raise RuntimeError("bad pdf")

    _install_fake_markitdown(monkeypatch, fake=_Boom())
    src = tmp_path / "x.pdf"
    src.write_bytes(b"x")
    tools = _by_name(build_tools(tmp_path, llm=_summary_llm()))
    out = tools["ingest_document"].invoke({"source_path": str(src)})
    assert out.startswith("ERROR: conversion failed")


def test_ingest_document_empty_conversion(tmp_path: Path, monkeypatch) -> None:
    class _Empty:
        class _R:
            text_content = ""
        def convert(self, src: str):
            return self._R()

    _install_fake_markitdown(monkeypatch, fake=_Empty())
    src = tmp_path / "x.pdf"
    src.write_bytes(b"x")
    tools = _by_name(build_tools(tmp_path, llm=_summary_llm()))
    out = tools["ingest_document"].invoke({"source_path": str(src)})
    assert "empty markdown" in out


def test_retrieve_llm_failure_during_synthesis(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n")
    write_index(tmp_path)

    calls = {"n": 0}

    def _respond(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "a.md"
        raise RuntimeError("synth boom")

    retrieve = _by_name(build_tools(tmp_path, llm=_ScriptedLLM(_respond)))["retrieve"]
    out = retrieve.invoke({"query": "x"})
    assert out.startswith("ERROR:") and "answer synthesis" in out
