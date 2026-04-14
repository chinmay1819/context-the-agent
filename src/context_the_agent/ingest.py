"""Generate a markdown INDEX.md that summarizes the files in a context directory.

The index becomes the agent's navigation map: each bullet lists a file path and a
one-line summary (from optional YAML frontmatter `summary:` or the first paragraph).
"""
from pathlib import Path
import re

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
SUMMARY_RE = re.compile(r"^summary:\s*(.+)$", re.MULTILINE)
SUMMARY_MAX = 200
INDEX_FILENAME = "INDEX.md"


def _extract_summary(text: str) -> str:
    """Derive a one-line summary for a markdown document.

    Resolution order:
        1. If the file begins with a YAML frontmatter block (``---\\n...\\n---``)
           containing a ``summary:`` field, return that value (stripped of
           surrounding quotes).
        2. Otherwise, return the first non-empty, non-heading paragraph from
           the remaining body, with newlines collapsed and truncated to
           ``SUMMARY_MAX`` characters (ending with ``…``).

    Args:
        text: Full markdown file contents as a string.

    Returns:
        The extracted summary, or ``"(no summary)"`` if no suitable
        paragraph could be found.
    """
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
        if not para:
            continue
        if para.startswith("#"):
            continue
        para = " ".join(para.splitlines()).strip()
        if len(para) > SUMMARY_MAX:
            para = para[: SUMMARY_MAX - 1].rstrip() + "…"
        return para
    return "(no summary)"


def build_index(context_dir: Path) -> str:
    """Render the ``INDEX.md`` text for a context directory without writing it.

    Walks every ``*.md`` file under ``context_dir`` (recursively, sorted by
    path), skipping any existing ``INDEX.md``, and renders one bullet per
    file in the form ``- `relpath` — summary`` where the summary comes from
    :func:`_extract_summary`.

    Args:
        context_dir: Directory to scan. Files outside this directory are
            never touched; the resulting index uses paths relative to it.

    Returns:
        The complete ``INDEX.md`` text, starting with ``# Context Index``
        and ending with a trailing newline. If no markdown files exist
        under ``context_dir``, returns an index with a single
        ``_(no markdown files found)_`` line.
    """
    root = Path(context_dir)
    entries: list[tuple[str, str]] = []
    for p in sorted(root.rglob("*.md")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.name == INDEX_FILENAME:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        entries.append((str(rel), _extract_summary(text)))

    lines = ["# Context Index", ""]
    if not entries:
        lines.append("_(no markdown files found)_")
    else:
        for rel, summary in entries:
            lines.append(f"- `{rel}` — {summary}")
    lines.append("")
    return "\n".join(lines)


def write_index(context_dir: Path) -> Path:
    """Build the index for ``context_dir`` and write it to ``INDEX.md``.

    Thin wrapper around :func:`build_index` that persists the output to
    ``<context_dir>/INDEX.md``, overwriting any existing index. Call this
    whenever the corpus changes.

    Args:
        context_dir: Directory containing the markdown corpus. Must exist.

    Returns:
        The ``Path`` of the written ``INDEX.md`` file.

    Raises:
        FileNotFoundError: If ``context_dir`` does not exist.
    """
    root = Path(context_dir)
    if not root.exists():
        raise FileNotFoundError(f"context directory does not exist: {root}")
    out = root / INDEX_FILENAME
    out.write_text(build_index(root), encoding="utf-8")
    return out


def load_index_prompt(context_dir: Path) -> str:
    """Read the existing ``INDEX.md`` from ``context_dir`` for embedding in a prompt.

    Intended to be spliced into an agent's system prompt so the LLM knows
    what files exist and what each one covers. Callers should have run
    :func:`write_index` at least once before calling this.

    Args:
        context_dir: Directory containing a previously generated ``INDEX.md``.

    Returns:
        The contents of ``<context_dir>/INDEX.md`` as a string.

    Raises:
        FileNotFoundError: If ``INDEX.md`` does not exist in ``context_dir``.
            The error message includes a hint to run :func:`write_index`.
    """
    path = Path(context_dir) / INDEX_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Call write_index({context_dir!r}) to generate it."
        )
    return path.read_text(encoding="utf-8")
