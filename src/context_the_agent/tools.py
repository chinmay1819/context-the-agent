from pathlib import Path
import re

from langchain_core.tools import BaseTool, tool

from ._security import resolve_safe_path

MAX_READ_BYTES = 200_000
MAX_GREP_HITS = 100
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")


def build_tools(context_root: Path, *, writable: bool = False) -> list[BaseTool]:
    """Build the LangChain tool set bound to a markdown context directory.

    Each returned tool is a closure over ``context_root``; all filesystem
    operations are routed through a path-traversal guard so the agent can
    never escape the directory.

    Args:
        context_root: Directory the tools may read (and optionally write).
            The path is ``.resolve()``-d once at build time.
        writable: If ``True``, include ``write_file`` in the returned list.
            Defaults to ``False`` (read-only) — callers must opt in to
            destructive behavior explicitly.

    Returns:
        A list of ``langchain_core.tools.BaseTool`` instances ready to pass
        to any LangChain agent. Read-only set (4 tools): ``read_file``,
        ``glob_files``, ``grep_content``, ``search_headings``. With
        ``writable=True``, ``write_file`` is appended (5 tools).
    """
    root = context_root.resolve()

    @tool
    def read_file(path: str) -> str:
        """Read a markdown file under the context directory and return its contents.

        Args:
            path: Path relative to the context root, e.g. ``"api.md"`` or
                ``"notes/auth.md"``.

        Returns:
            The file's UTF-8 contents. If the file exceeds ``MAX_READ_BYTES``
            (200 KB) it is truncated with a ``[...truncated, file is N bytes]``
            marker appended. If the path escapes the root or is not a file,
            returns a string starting with ``"ERROR: "`` rather than raising.
        """
        try:
            p = resolve_safe_path(path, root)
        except ValueError as e:
            return f"ERROR: {e}"
        if not p.exists() or not p.is_file():
            return f"ERROR: not a file: {path}"
        data = p.read_bytes()
        if len(data) > MAX_READ_BYTES:
            return (
                data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
                + f"\n\n[...truncated, file is {len(data)} bytes]"
            )
        return data.decode("utf-8", errors="replace")

    @tool
    def glob_files(pattern: str) -> list[str]:
        """List files under the context directory matching a glob pattern.

        Args:
            pattern: Glob pattern evaluated with ``pathlib.Path.rglob``,
                e.g. ``"**/*.md"`` or ``"notes/*.md"``.

        Returns:
            Sorted list of relative paths (as strings) for every file that
            matched the pattern. Empty list if nothing matched.
        """
        results = []
        for p in root.rglob(pattern):
            if p.is_file():
                results.append(str(p.relative_to(root)))
        results.sort()
        return results

    @tool
    def grep_content(pattern: str, path: str | None = None) -> list[str]:
        """Regex-search the text of markdown files under the context directory.

        Args:
            pattern: Python regular expression compiled with ``re.compile``.
            path: Optional relative file or directory to limit the search.
                Defaults to ``None`` which scans every ``*.md`` under the
                context root.

        Returns:
            A list of ``"rel/path.md:LINE: matched text"`` strings, capped at
            ``MAX_GREP_HITS`` (100) entries — a trailing
            ``"[...truncated at 100 hits]"`` marker is appended when the cap
            is reached. On invalid regex or an out-of-root ``path``, returns
            a single-element list whose string starts with ``"ERROR: "``.
        """
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return [f"ERROR: invalid regex: {e}"]

        if path:
            try:
                start = resolve_safe_path(path, root)
            except ValueError as e:
                return [f"ERROR: {e}"]
            if not start.exists():
                return [f"ERROR: not found: {path}"]
            files = [start] if start.is_file() else list(start.rglob("*.md"))
        else:
            files = list(root.rglob("*.md"))

        hits: list[str] = []
        for f in files:
            if not f.is_file():
                continue
            try:
                for lineno, line in enumerate(
                    f.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
                ):
                    if rx.search(line):
                        rel = f.relative_to(root)
                        hits.append(f"{rel}:{lineno}: {line.strip()}")
                        if len(hits) >= MAX_GREP_HITS:
                            hits.append(f"[...truncated at {MAX_GREP_HITS} hits]")
                            return hits
            except OSError as e:
                hits.append(f"ERROR reading {f}: {e}")
        return hits

    @tool
    def search_headings(query: str) -> list[str]:
        """Search H1/H2/H3 markdown headings across the context directory.

        Uses a case-insensitive substring match against the heading text
        (the portion after the ``#`` markers).

        Args:
            query: Substring to look for in heading text. Case-insensitive.

        Returns:
            A list of ``"rel/path.md:LINE: # Heading"`` strings, one per
            matching heading. Empty list if nothing matched.
        """
        q = query.lower()
        out: list[str] = []
        for f in root.rglob("*.md"):
            if not f.is_file():
                continue
            try:
                for lineno, line in enumerate(
                    f.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
                ):
                    m = HEADING_RE.match(line)
                    if m and q in m.group(2).lower():
                        rel = f.relative_to(root)
                        out.append(f"{rel}:{lineno}: {line.strip()}")
            except OSError as e:
                out.append(f"ERROR reading {f}: {e}")
        return out

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a markdown file under the context directory.

        Parent directories are created as needed. The write is unconditional
        — any existing file at ``path`` is replaced.

        Args:
            path: Relative path ending in ``.md``.
            content: Full UTF-8 file contents to write.

        Returns:
            A human-readable status string: ``"wrote N bytes to <path>"`` on
            success, or a string starting with ``"ERROR: "`` if the extension
            is wrong or the path escapes the context root.
        """
        if not path.endswith(".md"):
            return "ERROR: path must end in .md"
        try:
            p = resolve_safe_path(path, root)
        except ValueError as e:
            return f"ERROR: {e}"
        p.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        p.write_bytes(data)
        return f"wrote {len(data)} bytes to {path}"

    tools: list[BaseTool] = [read_file, glob_files, grep_content, search_headings]
    if writable:
        tools.append(write_file)
    return tools
