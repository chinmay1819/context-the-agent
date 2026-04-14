from pathlib import Path

import pytest


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A small markdown corpus used across tool and ingest tests."""
    (tmp_path / "api.md").write_text(
        "---\nsummary: REST endpoints for the dispatch service.\n---\n"
        "# API\n\n## Endpoints\n\nPOST /jobs creates a new job.\n",
        encoding="utf-8",
    )
    (tmp_path / "overview.md").write_text(
        "# Overview\n\nOrbit coordinates field technicians.\n\n"
        "It handles scheduling and routing.\n",
        encoding="utf-8",
    )
    sub = tmp_path / "notes"
    sub.mkdir()
    (sub / "auth.md").write_text(
        "# Auth Notes\n\nJWT tokens expire after 1 hour.\n",
        encoding="utf-8",
    )
    return tmp_path
