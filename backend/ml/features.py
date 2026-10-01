"""Shared feature extraction for training and inference."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

TOKEN_PATTERN = (
    r"(?u)[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|->|\+\+|--|&&|\|\|"
    r"|[+\-*/%<>&|^~!=?:;.,(){}\[\]]"
)
_TOKEN_RE = re.compile(TOKEN_PATTERN)
_COMMENT_RE = re.compile(r"/\*.*?\*/|//.*?$|#(?!include\b).*?$", re.MULTILINE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
MAX_CHARS = 4000
NUMERIC_NAMES = [
    "loc",
    "ptr_deref",
    "alloc_free",
    "dangerous_api",
    "if_count",
    "ptr_uses",
    "returns",
    "max_nesting",
    "gotos",
    "array_index",
    "casts",
    "check_alloc_ratio",
    "cyclomatic",
]


def preprocess(code: str, max_chars: int = MAX_CHARS) -> str:
    """Strip comments, normalize whitespace, truncate."""
    text = code or ""
    text = _COMMENT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def tokenize(code: str) -> list[str]:
    return _TOKEN_RE.findall(preprocess(code))


def numeric_features(code: str) -> np.ndarray:
    """Handcrafted numeric features for one function/file."""
    raw = code or ""
    pre = preprocess(raw)
    loc = max(raw.count("\n") + 1, 1)
    ptr_deref = pre.count("->") + len(re.findall(r"(?<![\w])\*(?!=)", pre))
    alloc_free = len(re.findall(r"\b(malloc|calloc|realloc|free)\s*\(", pre))
    dangerous_api = len(
        re.findall(r"\b(strcpy|strcat|sprintf|gets|memcpy|scanf)\s*\(", pre)
    )
    if_count = len(re.findall(r"\bif\s*\(", pre))
    ptr_uses = ptr_deref + len(re.findall(r"\b\w+\s*\[", pre))
    returns = len(re.findall(r"\breturn\b", pre))
    max_nesting = _max_brace_depth(raw)
    gotos = len(re.findall(r"\bgoto\b", pre))
    array_index = len(re.findall(r"\[[^\]]+\]", pre))
    casts = len(re.findall(r"\(\s*(?:const\s+)?(?:unsigned\s+|signed\s+)?(?:struct\s+)?\w+\s*\*?\s*\)", pre))
    check_alloc_ratio = if_count / (alloc_free + 1.0)
    cyclomatic = 1 + len(re.findall(r"\b(if|for|while|case|&&|\|\||catch)\b", pre))
    return np.array(
        [
            loc,
            ptr_deref,
            alloc_free,
            dangerous_api,
            if_count,
            ptr_uses,
            returns,
            max_nesting,
            gotos,
            array_index,
            casts,
            check_alloc_ratio,
            cyclomatic,
        ],
        dtype=np.float64,
    )


def _max_brace_depth(code: str) -> int:
    depth = 0
    best = 0
    for ch in code:
        if ch == "{":
            depth += 1
            best = max(best, depth)
        elif ch == "}":
            depth = max(0, depth - 1)
    return best


def numeric_matrix(codes: Sequence[str]) -> np.ndarray:
    return np.vstack([numeric_features(c) for c in codes])


@dataclass
class FeatureBundle:
    """Fitted vectorizers plus scaler for the numeric block."""

    word: TfidfVectorizer
    char: TfidfVectorizer
    scaler: StandardScaler

    def transform(self, codes: Sequence[str]) -> sparse.csr_matrix:
        docs = [preprocess(c) for c in codes]
        word_x = self.word.transform(docs)
        char_x = self.char.transform(docs)
        num = self.scaler.transform(numeric_matrix(codes))
        return sparse.hstack([word_x, char_x, sparse.csr_matrix(num)]).tocsr()

    def word_transform(self, codes: Sequence[str]) -> sparse.csr_matrix:
        docs = [preprocess(c) for c in codes]
        return self.word.transform(docs)


def fit_features(
    codes: Sequence[str],
    *,
    word_max_features: int = 100_000,
    char_max_features: int = 50_000,
) -> FeatureBundle:
    """Fit word/char TF-IDF and a numeric scaler on *codes*."""
    docs = [preprocess(c) for c in codes]
    word = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 3),
        max_features=word_max_features,
        sublinear_tf=True,
        min_df=2,
        lowercase=False,
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=char_max_features,
        sublinear_tf=True,
        min_df=2,
        lowercase=False,
    )
    word.fit(docs)
    char.fit(docs)
    scaler = StandardScaler()
    scaler.fit(numeric_matrix(codes))
    return FeatureBundle(word=word, char=char, scaler=scaler)


def iter_token_spans(code: str) -> Iterable[tuple[str, int, int, int]]:
    """Yield (token, line, start_col, end_col) in original source (1-based line)."""
    for match in _TOKEN_RE.finditer(code or ""):
        start = match.start()
        line = (code or "")[:start].count("\n") + 1
        line_start = (code or "").rfind("\n", 0, start) + 1
        col = start - line_start
        yield match.group(), line, col, col + len(match.group())
