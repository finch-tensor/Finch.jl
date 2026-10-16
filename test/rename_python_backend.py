"""Point legacy Python wrapper checkouts at Galley for integration testing."""

from pathlib import Path
import re
import subprocess
import sys


def rename_backend(root):
    root = Path(root)
    # Restrict edits to tracked sources, excluding installed dependencies and caches.
    paths = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"]
    ).decode().split("\0")
    for name in filter(None, paths):
        path = root / name
        if path.suffix not in {".py", ".jl", ".json"}:
            continue
        original = path.read_text()
        updated = original.replace("Finch.Galley", "Galley.Optimizer")
        # Keep compiler API names such as FinchNotation and @finch unchanged.
        updated = re.sub(r"\bFinch\b", "Galley", updated)
        if updated != original:
            path.write_text(updated)


if __name__ == "__main__":
    rename_backend(sys.argv[1])
