from pathlib import Path


def get_project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).resolve().parents[1]
