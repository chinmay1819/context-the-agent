import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from context_the_agent import write_index
from context_the_agent.ingest import (
    CACHE_FILENAME,
    INDEX_FILENAME,
    _heuristic_summary,
    build_index,
    load_index,
)


class _FakeLLM:
    def __init__(self):
        self.calls: list[str] = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        for line in prompt.splitlines():
            if line.startswith("File path:") and "`" in line:
                return AIMessage(content=f"summary of {line.split('`')[1]}")
        return AIMessage(content="summary")


class _ExplodingLLM:
    def invoke(self, prompt):
        raise RuntimeError("LLM down")


def test_heuristic_prefers_frontmatter() -> None:
    text = "---\nsummary: Hello world\n---\n# T\n\nBody.\n"
    assert _heuristic_summary(text) == "Hello world"


def test_heuristic_strips_quotes() -> None:
    text = "---\nsummary: \"quoted\"\n---\n# T\n"
    assert _heuristic_summary(text) == "quoted"


def test_heuristic_first_paragraph_fallback() -> None:
    text = "# Title\n\nFirst paragraph.\n\nSecond.\n"
    assert _heuristic_summary(text) == "First paragraph."


def test_heuristic_skips_headings() -> None:
    text = "# H1\n\n## H2\n\nBody here.\n"
    assert _heuristic_summary(text) == "Body here."


def test_heuristic_truncates_long() -> None:
    body = "x" * 500
    out = _heuristic_summary(f"# T\n\n{body}\n")
    assert out.endswith("…") and len(out) <= 200


def test_heuristic_empty() -> None:
    assert _heuristic_summary("") == "(no summary)"


def test_build_index_heuristic(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    (tmp_path / "b.md").write_text("# B\n\nBeta.\n")
    text = build_index(tmp_path)
    assert "`a.md` — Alpha." in text
    assert "`b.md` — Beta." in text


def test_build_index_skips_existing_index(tmp_path: Path) -> None:
    (tmp_path / INDEX_FILENAME).write_text("# stale\n")
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    assert "INDEX.md" not in build_index(tmp_path)


def test_build_index_empty(tmp_path: Path) -> None:
    assert "no markdown files found" in build_index(tmp_path)


def test_write_index_persists(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    out = write_index(tmp_path)
    assert out == tmp_path / INDEX_FILENAME
    assert "`a.md`" in out.read_text()


def test_write_index_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_index(tmp_path / "nope")


def test_load_index_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    write_index(tmp_path)
    assert "# Context Index" in load_index(tmp_path)


def test_load_index_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="write_index"):
        load_index(tmp_path)


def test_build_index_with_llm_invokes_once_per_file(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    (tmp_path / "b.md").write_text("# B\n\nBeta.\n")
    llm = _FakeLLM()
    text = build_index(tmp_path, llm=llm)
    assert "summary of a.md" in text
    assert "summary of b.md" in text
    assert len(llm.calls) == 2


def test_cache_written(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    build_index(tmp_path, llm=_FakeLLM())
    cache = json.loads((tmp_path / CACHE_FILENAME).read_text())
    assert "a.md" in cache and "hash" in cache["a.md"]


def test_cache_reused_when_unchanged(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    llm = _FakeLLM()
    build_index(tmp_path, llm=llm)
    build_index(tmp_path, llm=llm)
    assert len(llm.calls) == 1


def test_cache_invalidated_on_change(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    llm = _FakeLLM()
    build_index(tmp_path, llm=llm)
    (tmp_path / "a.md").write_text("# A\n\nAlpha v2.\n")
    build_index(tmp_path, llm=llm)
    assert len(llm.calls) == 2


def test_llm_failure_falls_back_to_heuristic(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha content.\n")
    text = build_index(tmp_path, llm=_ExplodingLLM())
    assert "Alpha content." in text  # heuristic survived


def test_unreadable_file_entry(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "ok.md").write_text("# ok\n\nBody.\n")
    (tmp_path / "bad.md").write_text("# bad\n")
    real = Path.read_text

    def _maybe(self, *a, **kw):
        if self.name == "bad.md":
            raise OSError("denied")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", _maybe)
    text = build_index(tmp_path)
    assert "`ok.md`" in text
    assert "`bad.md` — (unreadable:" in text


def test_cache_save_failure_swallowed(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")

    real_write = Path.write_text

    def _maybe(self, *a, **kw):
        if self.name == CACHE_FILENAME:
            raise OSError("ro filesystem")
        return real_write(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", _maybe)
    # Must not raise
    text = build_index(tmp_path, llm=_FakeLLM())
    assert "`a.md`" in text
