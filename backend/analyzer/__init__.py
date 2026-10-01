"""Static analysis engine (tree-sitter C parser + conservative bug rules)."""

from backend.analyzer.language import detect_language
from backend.analyzer.parser import ParsedFile, extract_functions, parse_source
from backend.analyzer.rules import Finding, analyze_code

__all__ = [
    "Finding",
    "ParsedFile",
    "analyze_code",
    "detect_language",
    "extract_functions",
    "parse_source",
]
