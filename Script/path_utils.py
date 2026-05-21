from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_repo_path(path_like: Path) -> Path:
    if path_like.is_absolute():
        return path_like
    root = repo_root()
    direct = Path(path_like)
    if direct.exists():
        return direct
    root_path = root / path_like
    if root_path.exists():
        return root_path
    parent_path = Path("..") / path_like
    if parent_path.exists():
        return parent_path
    return root_path
