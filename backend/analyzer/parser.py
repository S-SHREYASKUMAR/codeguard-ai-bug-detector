"""tree-sitter C parsing and function extraction with regex fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_C_LANGUAGE: Any | None = None
_PARSER: Any | None = None
_TREE_SITTER_READY = False


def _init_parser() -> None:
    global _C_LANGUAGE, _PARSER, _TREE_SITTER_READY
    if _PARSER is not None:
        return
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_c as tsc

        lang_capsule = tsc.language()
        try:
            _C_LANGUAGE = Language(lang_capsule)
        except TypeError:
            _C_LANGUAGE = Language(lang_capsule, "c")
        try:
            _PARSER = Parser(_C_LANGUAGE)
        except TypeError:
            _PARSER = Parser()
            if hasattr(_PARSER, "language"):
                _PARSER.language = _C_LANGUAGE
            else:
                _PARSER.set_language(_C_LANGUAGE)
        _TREE_SITTER_READY = True
    except Exception as exc:  # pragma: no cover - environment-specific
        logger.warning("tree-sitter unavailable, using regex fallback: %s", exc)
        _TREE_SITTER_READY = False
        _PARSER = False  # sentinel so we do not retry forever


@dataclass
class FunctionInfo:
    """A C function extracted from source."""

    name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    source: str
    node: Any | None = None
    is_regex_fallback: bool = False


@dataclass
class ParsedFile:
    """Parse result for a source buffer."""

    source: str
    source_bytes: bytes
    tree: Any | None
    functions: list[FunctionInfo] = field(default_factory=list)
    used_fallback: bool = False
    root: Any | None = None


def line_of(source: str, byte_offset: int) -> int:
    """1-based line number for a byte offset into UTF-8 source."""
    if byte_offset <= 0:
        return 1
    encoded = source.encode("utf-8", errors="replace")
    clipped = min(byte_offset, len(encoded))
    return encoded[:clipped].count(b"\n") + 1


def node_text(source_bytes: bytes, node: Any) -> str:
    """Decode the exact source span of a tree-sitter node."""
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def walk(node: Any) -> Iterator[Any]:
    """Pre-order walk of a tree-sitter subtree."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        children = list(current.children)
        for child in reversed(children):
            stack.append(child)


def parse_source(code: str) -> ParsedFile:
    """Parse C source. Falls back to regex function extraction on failure."""
    _init_parser()
    source = code or ""
    source_bytes = source.encode("utf-8", errors="replace")
    parsed = ParsedFile(source=source, source_bytes=source_bytes, tree=None)

    if _TREE_SITTER_READY and _PARSER:
        try:
            tree = _PARSER.parse(source_bytes)
            parsed.tree = tree
            parsed.root = tree.root_node
            parsed.functions = _functions_from_tree(source, source_bytes, tree.root_node)
            if parsed.functions:
                return parsed
            # Empty function list on clearly function-like input → fallback.
            if re.search(r"\w+\s*\([^;]*\)\s*\{", source):
                logger.info("Parser found no functions; using regex fallback")
            else:
                return parsed
        except Exception as exc:
            logger.warning("tree-sitter parse failed: %s", exc)

    parsed.used_fallback = True
    parsed.functions = _functions_from_regex(source)
    return parsed


def extract_functions(code: str) -> list[FunctionInfo]:
    """Convenience wrapper around :func:`parse_source`."""
    return parse_source(code).functions


def _functions_from_tree(source: str, source_bytes: bytes, root: Any) -> list[FunctionInfo]:
    functions: list[FunctionInfo] = []
    for node in walk(root):
        if node.type != "function_definition":
            continue
        name = _function_name(source_bytes, node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        functions.append(
            FunctionInfo(
                name=name,
                start_line=start_line,
                end_line=end_line,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                source=node_text(source_bytes, node),
                node=node,
            )
        )
    functions.sort(key=lambda f: f.start_line)
    return functions


def _function_name(source_bytes: bytes, fn_node: Any) -> str:
    declarator = None
    for child in fn_node.children:
        if child.type == "function_declarator":
            declarator = child
            break
        if child.type == "pointer_declarator":
            for inner in walk(child):
                if inner.type == "function_declarator":
                    declarator = inner
                    break
    if declarator is None:
        for inner in walk(fn_node):
            if inner.type == "function_declarator":
                declarator = inner
                break
    if declarator is not None:
        for child in declarator.children:
            if child.type == "identifier":
                return node_text(source_bytes, child)
            if child.type == "parenthesized_declarator":
                for inner in walk(child):
                    if inner.type == "identifier":
                        return node_text(source_bytes, inner)
    return "<anonymous>"


_FN_REGEX = re.compile(
    r"(?:^|\n)[ \t]*(?:[\w\*]+\s+)+\*?(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
    re.MULTILINE,
)


def _functions_from_regex(source: str) -> list[FunctionInfo]:
    """Brace-matching function extractor used when tree-sitter fails."""
    functions: list[FunctionInfo] = []
    for match in _FN_REGEX.finditer(source):
        name = match.group("name")
        if name in {"if", "for", "while", "switch", "return"}:
            continue
        brace_start = match.end() - 1
        end = _matching_brace(source, brace_start)
        if end is None:
            end = len(source) - 1
        start_line = source[: match.start()].count("\n") + 1
        if match.start() > 0 and source[match.start()] == "\n":
            start_line = source[: match.start() + 1].count("\n") + 1
        end_line = source[: end + 1].count("\n") + 1
        snippet = source[match.start() : end + 1].lstrip("\n")
        functions.append(
            FunctionInfo(
                name=name,
                start_line=start_line,
                end_line=end_line,
                start_byte=match.start(),
                end_byte=end + 1,
                source=snippet,
                node=None,
                is_regex_fallback=True,
            )
        )
    if not functions and source.strip():
        functions.append(
            FunctionInfo(
                name="<file>",
                start_line=1,
                end_line=source.count("\n") + 1,
                start_byte=0,
                end_byte=len(source),
                source=source,
                node=None,
                is_regex_fallback=True,
            )
        )
    return functions


def _matching_brace(source: str, open_idx: int) -> int | None:
    depth = 0
    i = open_idx
    n = len(source)
    in_str = False
    str_ch = ""
    in_char = False
    in_line = False
    in_block = False
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if in_line:
            if ch == "\n":
                in_line = False
            i += 1
            continue
        if in_block:
            if ch == "*" and nxt == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == str_ch:
                in_str = False
            i += 1
            continue
        if in_char:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "'":
                in_char = False
            i += 1
            continue
        if ch == "/" and nxt == "/":
            in_line = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block = True
            i += 2
            continue
        if ch in {'"', "'"}:
            if ch == '"':
                in_str = True
                str_ch = '"'
            else:
                in_char = True
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None
