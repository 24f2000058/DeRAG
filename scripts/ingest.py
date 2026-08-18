import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdf

_TEXT_SUFFIXES = {".txt", ".pdf"}


def _extract_text(path: Path) -> str:
    if path.suffix == ".pdf":
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.iterdir() if p.is_file() and p.suffix in _TEXT_SUFFIXES
    )
