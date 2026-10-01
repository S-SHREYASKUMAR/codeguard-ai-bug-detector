"""Heuristic language detection for analyzer input."""

from __future__ import annotations

from dataclasses import dataclass
import re

_C_HINTS = re.compile(
    r"\b(#include|#define|typedef|struct|union|enum|sizeof|malloc|printf|"
    r"strcpy|NULL|int\s+\w+\s*\(|void\s+\w+\s*\(|char\s+\*?)\b",
    re.IGNORECASE,
)
_CPP_HINTS = re.compile(
    r"\b(std::|namespace|template\s*<|class\s+\w+|new\s+\w+|delete\s+|cout|cin)\b"
)
_PYTHON_HINTS = re.compile(
    r"^\s*(def |class |import |from |print\(|if __name__)",
    re.MULTILINE,
)
_JAVA_HINTS = re.compile(
    r"\b(public\s+class|System\.out|import java\.|package\s+\w+)\b"
)
_JS_HINTS = re.compile(
    r"\b(function |const |let |=>|console\.log|export |import .* from)\b"
)


@dataclass(frozen=True)
class LanguageInfo:
    """Detected language plus optional user-facing warning."""

    detected: str
    warning: str | None
    looks_like_c: bool


def detect_language(code: str) -> LanguageInfo:
    """Guess whether *code* looks like C/C++ vs Python/Java/JavaScript.

    Non-C input still proceeds in best-effort mode with a warning.
    """
    sample = (code or "")[:8000]
    if not sample.strip():
        return LanguageInfo(detected="unknown", warning=None, looks_like_c=False)

    py = bool(_PYTHON_HINTS.search(sample)) and "def " in sample
    java = bool(_JAVA_HINTS.search(sample))
    js = (
        (bool(_JS_HINTS.search(sample)) and "console.log" in sample)
        or ("function " in sample and "=>" in sample and "#include" not in sample)
    )
    c_like = bool(_C_HINTS.search(sample) or _CPP_HINTS.search(sample))

    if py and not c_like:
        return LanguageInfo(
            detected="python",
            warning="This analyzer is built for C; results may be unreliable",
            looks_like_c=False,
        )
    if java and not c_like:
        return LanguageInfo(
            detected="java",
            warning="This analyzer is built for C; results may be unreliable",
            looks_like_c=False,
        )
    if js and not c_like:
        return LanguageInfo(
            detected="javascript",
            warning="This analyzer is built for C; results may be unreliable",
            looks_like_c=False,
        )
    if _CPP_HINTS.search(sample) and not _C_HINTS.search(sample):
        return LanguageInfo(detected="c++", warning=None, looks_like_c=True)
    if c_like:
        kind = "c++" if _CPP_HINTS.search(sample) else "c"
        return LanguageInfo(detected=kind, warning=None, looks_like_c=True)

    return LanguageInfo(
        detected="unknown",
        warning="This analyzer is built for C; results may be unreliable",
        looks_like_c=False,
    )
