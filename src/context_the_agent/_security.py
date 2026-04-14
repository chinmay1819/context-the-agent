from pathlib import Path


def resolve_safe_path(rel: str, context_root: Path) -> Path:
    """Resolve a relative path under ``context_root`` and reject any escape attempt.

    Protects every filesystem-facing tool from path-traversal inputs (``../..``,
    absolute paths, symlinks that point outside the root). Both ``context_root``
    and the candidate are ``.resolve()``-d before comparison so symlinks can't
    be used to sneak out.

    Args:
        rel: Path relative to the context root (e.g. ``"api.md"`` or
            ``"notes/auth.md"``). Absolute paths will also be rejected if they
            resolve outside the root.
        context_root: The directory the caller is allowed to read/write within.

    Returns:
        The fully resolved absolute ``Path`` of the candidate, guaranteed to
        live under ``context_root``.

    Raises:
        ValueError: If the resolved candidate is not inside ``context_root``.
    """
    root = context_root.resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError(f"path escapes context root: {rel}") from e
    return candidate
