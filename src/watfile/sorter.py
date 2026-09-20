"""Move classified files into category folders."""

import shutil
from dataclasses import dataclass
from pathlib import Path

_UNCATEGORIZED = "uncategorized"


@dataclass(frozen=True)
class MoveResult:
    source: Path
    destination: Path
    moved: bool  # False in dry-run or on failure


def sanitize_category(name: str) -> str:
    """Make a category safe as a folder name."""
    keep = "-_. ()"
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in name.strip())
    return cleaned.strip(". ") or _UNCATEGORIZED


def _resolve_collision(destination: Path) -> Path:
    """Return a non-existing variant of *destination* by appending a counter."""
    if not destination.exists():
        return destination
    stem, suffix = destination.stem, destination.suffix
    for i in range(1, 1000):
        candidate = destination.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find free name for {destination}")


def place_file(
    source: Path,
    category: str,
    target_root: Path,
    *,
    dry_run: bool = False,
    copy: bool = False,
) -> MoveResult:
    """Place *source* into target_root/<category>/. Creates folders as needed."""
    category_dir = target_root / sanitize_category(category)
    destination = _resolve_collision(category_dir / source.name)
    if dry_run:
        return MoveResult(source=source, destination=destination, moved=False)
    category_dir.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, destination)
    else:
        shutil.move(str(source), destination)
    return MoveResult(source=source, destination=destination, moved=True)
