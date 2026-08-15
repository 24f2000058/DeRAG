import re
import unicodedata

_KEEP_WHITESPACE = "\t\n\r"

_REMOVE_TABLE = {
    codepoint: None
    for codepoint in range(0x110000)
    if unicodedata.category(chr(codepoint)) in ("Cc", "Cf")
    and chr(codepoint) not in _KEEP_WHITESPACE
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = text.translate(_REMOVE_TABLE)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.lower()
