from pathlib import Path

import pytest

from context_the_agent._security import resolve_safe_path


def test_resolves_relative_path(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("x")
    resolved = resolve_safe_path("a.md", tmp_path)
    assert resolved == (tmp_path / "a.md").resolve()


def test_resolves_nested_path(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("x")
    resolved = resolve_safe_path("sub/b.md", tmp_path)
    assert resolved == (tmp_path / "sub" / "b.md").resolve()


def test_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes context root"):
        resolve_safe_path("../etc/passwd", tmp_path)


def test_rejects_deep_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes context root"):
        resolve_safe_path("../../../etc/passwd", tmp_path)


def test_rejects_absolute_outside_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes context root"):
        resolve_safe_path("/etc/passwd", tmp_path)


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.md"
    outside.write_text("secret")
    try:
        link = tmp_path / "link.md"
        link.symlink_to(outside)
        with pytest.raises(ValueError, match="escapes context root"):
            resolve_safe_path("link.md", tmp_path)
    finally:
        outside.unlink(missing_ok=True)
