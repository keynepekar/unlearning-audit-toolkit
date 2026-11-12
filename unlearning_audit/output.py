"""Write audit artifacts without leaving partially written files."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_artifact(path: Path, text: str, *, inputs: tuple[Path, ...] = ()) -> None:
    destination = path.resolve()
    if any(
        destination == source.resolve()
        or (destination.exists() and destination.samefile(source))
        for source in inputs
    ):
        raise ValueError("output path must differ from every input path")

    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
