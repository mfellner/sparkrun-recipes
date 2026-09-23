"""Copy only manifest-listed TP3 build artifacts; never copy model weights."""
from pathlib import Path
import shutil
import sys

from manifest import verify_artifacts


def stage(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    data = verify_artifacts(source)
    if source == destination:
        return
    for relative in (*data["files"], "manifest.json", "LICENSE.MIT", "LICENSE.upstream-AGPL-3.0"):
        src, dst = source / relative, destination / relative
        if not dst.resolve().is_relative_to(destination):
            raise ValueError("destination escapes bundle")
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    verify_artifacts(destination)


if __name__ == "__main__":
    stage(*sys.argv[1:])
