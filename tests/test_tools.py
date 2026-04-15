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


def test_retrieve_reads_index(tmp_path: Path) -> None:
    """retrieve(['INDEX.md']) on a fresh corpus bootstraps and returns the index."""
    (tmp_path / "api.md").write_text("# API\n\nPOST /jobs.\n")

    retrieve = _by_name(build_tools(tmp_path, llm=_summary_llm()))["retrieve"]
    out = retrieve.invoke({"paths": ["INDEX.md"]})
    assert "=== `INDEX.md` ===" in out
    assert "`api.md`" in out
    assert (tmp_path / "INDEX.md").exists()


def test_retrieve_reads_multiple_files(tmp_path: Path) -> None:
    (tmp_path / "api.md").write_text("# API\n\nPOST /jobs creates a job.\n")
    (tmp_path / "auth.md").write_text("# Auth\n\nJWT tokens.\n")

    retrieve = _by_name(build_tools(tmp_path, llm=_summary_llm()))["retrieve"]
    out = retrieve.invoke({"paths": ["api.md", "auth.md"]})
    assert "=== `api.md` ===" in out
    assert "POST /jobs" in out
    assert "=== `auth.md` ===" in out
    assert "JWT tokens" in out


def test_retrieve_reports_bad_paths_inline(tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text("# Good\n")

    retrieve = _by_name(build_tools(tmp_path, llm=_summary_llm()))["retrieve"]
    out = retrieve.invoke(
        {"paths": ["good.md", "missing.md", "../../etc/passwd.md", "bad.txt"]}
    )
    # Good file renders normally
    assert "=== `good.md` ===" in out
    # Missing file reported
    assert "=== ERROR: not a file: missing.md ===" in out
    # Traversal reported
    assert "escapes context root" in out
    # Non-markdown rejected
    assert "path must end in .md: bad.txt" in out


def test_retrieve_empty_paths(tmp_path: Path) -> None:
    retrieve = _by_name(build_tools(tmp_path, llm=_summary_llm()))["retrieve"]
    out = retrieve.invoke({"paths": []})
    assert "no paths provided" in out
    assert "INDEX.md" in out


def test_retrieve_truncates_large_file(tmp_path: Path) -> None:
    (tmp_path / "big.md").write_text("x" * 60_000)
    retrieve = _by_name(build_tools(tmp_path, llm=_summary_llm()))["retrieve"]
    out = retrieve.invoke({"paths": ["big.md"]})
    assert "[...truncated]" in out


def test_retrieve_read_os_error(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.md").write_text("# A\n")
    retrieve = _by_name(build_tools(tmp_path, llm=_summary_llm()))["retrieve"]

    real = Path.read_text

    def _maybe(self, *a, **kw):
        if self.name == "a.md":
            raise OSError("permission denied")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", _maybe)
    out = retrieve.invoke({"paths": ["a.md"]})
    assert "could not read a.md" in out


def test_retrieve_makes_no_llm_calls_on_read(tmp_path: Path) -> None:
    """After INDEX.md exists, retrieve must not invoke the LLM."""
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    llm = _summary_llm()
    tools = _by_name(build_tools(tmp_path, llm=llm))
    # Bootstrap (LLM used for per-file summary)
    tools["retrieve"].invoke({"paths": ["INDEX.md"]})
    calls_after_bootstrap = len(llm.calls)
    # Subsequent reads should add zero LLM calls.
    tools["retrieve"].invoke({"paths": ["a.md"]})
    tools["retrieve"].invoke({"paths": ["INDEX.md", "a.md"]})
    assert len(llm.calls) == calls_after_bootstrap


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


