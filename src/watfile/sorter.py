"""Move classified files into category folders."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

_UNCATEGORIZED = "uncategorized"


@dataclass(frozen=True)
class MoveResult:
    source: Path
    destination: Path
    moved: bool  # False in dry-run or on failure


#: How a classified file is placed into its category folder.
PLACEMENT_SYMLINK = "symlink"  # default: leave original in place, link in category folder
PLACEMENT_MOVE = "move"
PLACEMENT_COPY = "copy"


def sanitize_category(name: str) -> str:
    """Make a category safe as a folder name."""
    keep = "-_. ()"
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in name.strip())
    return cleaned.strip(". ") or _UNCATEGORIZED


def _resolve_collision(destination: Path) -> Path:
    """Return a non-existing variant of *destination* by appending a counter.

    Uses lexists so dangling symlinks also count as occupied.
    """
    if not os.path.lexists(destination):
        return destination
    stem, suffix = destination.stem, destination.suffix
    for i in range(1, 1000):
        candidate = destination.with_name(f"{stem}_{i}{suffix}")
        if not os.path.lexists(candidate):
            return candidate
    raise FileExistsError(f"could not find free name for {destination}")


def place_file(
    source: Path,
    category: str,
    target_root: Path,
    *,
    dry_run: bool = False,
    placement: str = PLACEMENT_SYMLINK,
) -> MoveResult:
    """Place *source* into target_root/<category>/ as symlink (default), move, or copy.

    Symlinks point at the absolute original location; move leaves the original
    gone; copy duplicates the file.
    """
    category_dir = target_root / sanitize_category(category)
    destination = _resolve_collision(category_dir / source.name)
    if dry_run:
        return MoveResult(source=source, destination=destination, moved=False)
    category_dir.mkdir(parents=True, exist_ok=True)
    if placement == PLACEMENT_SYMLINK:
        os.symlink(source.resolve(), destination)
    elif placement == PLACEMENT_COPY:
        shutil.copy2(source, destination)
    elif placement == PLACEMENT_MOVE:
        shutil.move(str(source), destination)
    else:
        raise ValueError(f"unknown placement: {placement}")
    return MoveResult(source=source, destination=destination, moved=True)
