"""Conservative C bug rules over a tree-sitter AST (regex fallback if needed)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable

from backend.analyzer.parser import (
    FunctionInfo,
    ParsedFile,
    node_text,
    parse_source,
    walk,
)

ALLOC_FUNCS = frozenset(
    {"malloc", "calloc", "realloc", "strdup", "fopen", "fopen_s", "aligned_alloc"}
)
FREE_FUNCS = frozenset({"free", "fclose"})
UNBOUNDED_COPY = frozenset({"strcpy", "strcat", "sprintf", "vsprintf", "gets"})
FORMAT_FUNCS = {
    "printf": 0,
    "sprintf": 1,
    "vsprintf": 1,
    "syslog": 1,
    "fprintf": 1,
    "snprintf": 2,
    "vsnprintf": 2,
}
DANGEROUS = {
    "gets": ("Use fgets with a bounded buffer.", "CWE-242"),
    "tmpnam": ("Use mkstemp or tmpfile instead.", "CWE-377"),
    "mktemp": ("Use mkstemp instead.", "CWE-377"),
    "atoi": ("Use strtol and check the full conversion.", "CWE-190"),
}
CHECKED_RETURNS = frozenset(
    {
        "malloc",
        "calloc",
        "realloc",
        "fread",
        "fwrite",
        "read",
        "write",
        "snprintf",
        "fopen",
        "strdup",
    }
)
PTHREAD_PREFIX = "pthread_"


@dataclass
class Finding:
    """A single static-analysis finding."""

    rule_id: str
    bug_type: str
    cwe: str
    severity: str
    line: int
    end_line: int
    function: str
    message: str
    why_it_is_a_bug: str
    fix_suggestion: str
    fixed_code_snippet: str
    confidence: float
    mode: str = "ast"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_code(code: str) -> list[Finding]:
    """Run all rules on *code* and return conservative findings."""
    parsed = parse_source(code or "")
    findings: list[Finding] = []
    if not parsed.functions:
        return findings
    for fn in parsed.functions:
        if fn.node is not None and not parsed.used_fallback:
            findings.extend(_analyze_function_ast(parsed, fn))
        else:
            findings.extend(_analyze_function_regex(fn))
    return _dedupe(findings)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, int, str]] = set()
    out: list[Finding] = []
    for item in findings:
        key = (item.rule_id, item.line, item.function)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _analyze_function_ast(parsed: ParsedFile, fn: FunctionInfo) -> list[Finding]:
    ctx = FunctionContext.build(parsed, fn)
    findings: list[Finding] = []
    findings.extend(rule_buf_overflow(ctx))
    findings.extend(rule_null_deref(ctx))
    findings.extend(rule_use_after_free(ctx))
    findings.extend(rule_double_free(ctx))
    findings.extend(rule_memory_leak(ctx))
    findings.extend(rule_int_overflow(ctx))
    findings.extend(rule_format_string(ctx))
    findings.extend(rule_uninit_var(ctx))
    findings.extend(rule_div_zero(ctx))
    findings.extend(rule_unchecked_return(ctx))
    findings.extend(rule_dangerous_func(ctx))
    findings.extend(rule_infinite_or_missing_return(ctx))
    findings.extend(rule_stack_array(ctx))
    findings.extend(rule_assign_in_condition(ctx))
    if ctx.fallback_hint:
        for item in findings:
            item.confidence = min(item.confidence, 0.55)
            item.mode = "fallback"
    return findings


def _analyze_function_regex(fn: FunctionInfo) -> list[Finding]:
    """Line-oriented rules when the C parser cannot produce an AST."""
    ctx_source = fn.source
    fake = FunctionContext(
        parsed=ParsedFile(source=ctx_source, source_bytes=b"", tree=None),
        fn=fn,
        source=ctx_source,
        source_bytes=ctx_source.encode("utf-8", errors="replace"),
        fallback_hint=True,
    )
    # Reuse regex scans that do not need a node.
    return _regex_rules(fake)


# ---------------------------------------------------------------------------
# Function context / AST helpers
# ---------------------------------------------------------------------------


@dataclass
class LocalVar:
    name: str
    line: int
    is_pointer: bool
    is_array: bool
    array_size: int | None
    has_init: bool
    is_param: bool
    node: Any | None = None


@dataclass
class CallSite:
    name: str
    node: Any
    args: list[Any]
    line: int
    assigned_to: str | None
    used: bool  # return value consumed (assign, if, return, arg)


@dataclass
class FunctionContext:
    parsed: ParsedFile
    fn: FunctionInfo
    source: str
    source_bytes: bytes
    locals: dict[str, LocalVar] = field(default_factory=dict)
    calls: list[CallSite] = field(default_factory=list)
    fallback_hint: bool = False

    @property
    def name(self) -> str:
        return self.fn.name

    @classmethod
    def build(cls, parsed: ParsedFile, fn: FunctionInfo) -> FunctionContext:
        ctx = cls(
            parsed=parsed,
            fn=fn,
            source=parsed.source,
            source_bytes=parsed.source_bytes,
        )
        if fn.node is None:
            ctx.fallback_hint = True
            return ctx
        ctx._collect_params()
        ctx._collect_locals()
        ctx._collect_calls()
        return ctx

    def text(self, node: Any) -> str:
        return node_text(self.source_bytes, node)

    def line(self, node: Any) -> int:
        return int(node.start_point[0]) + 1

    def end_line(self, node: Any) -> int:
        return int(node.end_point[0]) + 1

    def _collect_params(self) -> None:
        if self.fn.node is None:
            return
        for node in walk(self.fn.node):
            if node.type != "parameter_declaration":
                continue
            name, is_ptr, is_arr, size = _declarator_info(self, node)
            if not name:
                continue
            self.locals[name] = LocalVar(
                name=name,
                line=self.line(node),
                is_pointer=is_ptr or is_arr,
                is_array=is_arr,
                array_size=size,
                has_init=True,
                is_param=True,
                node=node,
            )

    def _collect_locals(self) -> None:
        body = _function_body(self.fn.node)
        if body is None:
            return
        for node in walk(body):
            if node.type != "declaration":
                continue
            # Skip nested function-like; still handle locals in blocks.
            name, is_ptr, is_arr, size = _declarator_info(self, node)
            if not name or name in self.locals and self.locals[name].is_param:
                if name and self.locals.get(name) and self.locals[name].is_param:
                    continue
            if not name:
                # Multiple declarators: walk init_declarator
                for child in walk(node):
                    if child.type != "init_declarator" and child.type != "array_declarator":
                        continue
                names = _declaration_names(self, node)
                for n, ip, ia, sz, init in names:
                    self.locals[n] = LocalVar(
                        name=n,
                        line=self.line(node),
                        is_pointer=ip or ia,
                        is_array=ia,
                        array_size=sz,
                        has_init=init,
                        is_param=False,
                        node=node,
                    )
                continue
            has_init = any(c.type == "init_declarator" for c in node.children) or "=" in self.text(
                node
            ).split(";")[0]
            # Prefer structured names
            names = _declaration_names(self, node)
            if names:
                for n, ip, ia, sz, init in names:
                    if n in self.locals and self.locals[n].is_param:
                        continue
                    self.locals[n] = LocalVar(
                        name=n,
                        line=self.line(node),
                        is_pointer=ip or ia,
                        is_array=ia,
                        array_size=sz,
                        has_init=init,
                        is_param=False,
                        node=node,
                    )
            else:
                self.locals[name] = LocalVar(
                    name=name,
                    line=self.line(node),
                    is_pointer=is_ptr or is_arr,
                    is_array=is_arr,
                    array_size=size,
                    has_init=has_init,
                    is_param=False,
                    node=node,
                )

    def _collect_calls(self) -> None:
        if self.fn.node is None:
            return
        for node in walk(self.fn.node):
            if node.type != "call_expression":
                continue
            name = _callee_name(self, node)
            if not name:
                continue
            args_node = None
            for child in node.children:
                if child.type == "argument_list":
                    args_node = child
            args = _call_args(args_node)
            parent = node.parent
            assigned = None
            used = False
            if parent is not None:
                if parent.type == "init_declarator":
                    used = True
                    assigned = _init_declarator_name(self, parent)
                elif parent.type == "assignment_expression":
                    used = True
                    left = parent.child_by_field_name("left") if hasattr(parent, "child_by_field_name") else None
                    if left is None and len(parent.children) >= 1:
                        left = parent.children[0]
                    if left is not None and left.type == "identifier":
                        assigned = self.text(left)
                elif parent.type in {
                    "if_statement",
                    "parenthesized_expression",
                    "binary_expression",
                    "unary_expression",
                    "return_statement",
                    "argument_list",
                    "cast_expression",
                    "conditional_expression",
                    "expression_statement",
                }:
                    if parent.type == "expression_statement":
                        used = False
                    else:
                        used = True
                if parent.type == "expression_statement":
                    used = False
            self.calls.append(
                CallSite(
                    name=name,
                    node=node,
                    args=args,
                    line=self.line(node),
                    assigned_to=assigned,
                    used=used,
                )
            )


def _function_body(fn_node: Any) -> Any | None:
    if fn_node is None:
        return None
    for child in fn_node.children:
        if child.type == "compound_statement":
            return child
    return None


def _callee_name(ctx: FunctionContext, call: Any) -> str:
    fn_node = call.child_by_field_name("function") if hasattr(call, "child_by_field_name") else None
    if fn_node is None and call.children:
        fn_node = call.children[0]
    if fn_node is None:
        return ""
    if fn_node.type == "identifier":
        return ctx.text(fn_node)
    if fn_node.type == "field_expression":
        field = fn_node.child_by_field_name("field") if hasattr(fn_node, "child_by_field_name") else None
        if field is not None:
            return ctx.text(field)
    return ctx.text(fn_node).split("(")[0].strip()


def _call_args(args_node: Any) -> list[Any]:
    if args_node is None:
        return []
    out = []
    for child in args_node.children:
        if child.type in {"(", ")", ","}:
            continue
        out.append(child)
    return out


def _identifier_of(ctx: FunctionContext, node: Any) -> str | None:
    if node is None:
        return None
    if node.type == "identifier":
        return ctx.text(node)
    if node.type in {"pointer_expression", "unary_expression", "parenthesized_expression", "cast_expression"}:
        for child in node.children:
            ident = _identifier_of(ctx, child)
            if ident:
                return ident
    if node.type in {"field_expression", "subscript_expression"}:
        arg = node.child_by_field_name("argument") if hasattr(node, "child_by_field_name") else None
        if arg is None and node.children:
            arg = node.children[0]
        return _identifier_of(ctx, arg)
    return None


def _declarator_info(ctx: FunctionContext, node: Any) -> tuple[str | None, bool, bool, int | None]:
    names = _declaration_names(ctx, node)
    if not names:
        return None, False, False, None
    n, ip, ia, sz, _init = names[0]
    return n, ip, ia, sz


def _declaration_names(
    ctx: FunctionContext, node: Any
) -> list[tuple[str, bool, bool, int | None, bool]]:
    results: list[tuple[str, bool, bool, int | None, bool]] = []
    for child in walk(node):
        if child.type in {"parameter_list", "compound_statement", "function_declarator"}:
            # Do not walk into nested functions / parameter lists of function pointers deeply
            continue
    # Direct scan of declarators under this declaration
    stack = list(node.children)
    while stack:
        cur = stack.pop(0)
        if cur.type in {"compound_statement", "parameter_list"}:
            continue
        if cur.type == "init_declarator":
            decl = cur.child_by_field_name("declarator") if hasattr(cur, "child_by_field_name") else cur.children[0]
            value = cur.child_by_field_name("value") if hasattr(cur, "child_by_field_name") else None
            name, is_ptr, is_arr, size = _parse_declarator(ctx, decl)
            if name:
                results.append((name, is_ptr, is_arr, size, value is not None or "=" in ctx.text(cur)))
            continue
        if cur.type in {"identifier", "pointer_declarator", "array_declarator"}:
            # bare declaration without init
            if cur.type == "identifier" and cur.parent is not None and cur.parent.type == "declaration":
                results.append((ctx.text(cur), False, False, None, False))
                continue
            name, is_ptr, is_arr, size = _parse_declarator(ctx, cur)
            if name and cur.parent is not None and cur.parent.type in {
                "declaration",
                "parameter_declaration",
            }:
                results.append((name, is_ptr, is_arr, size, False))
                continue
        if cur.type in {"pointer_declarator", "array_declarator", "declaration", "parameter_declaration"}:
            stack.extend(cur.children)
    return results


def _parse_declarator(ctx: FunctionContext, node: Any | None) -> tuple[str | None, bool, bool, int | None]:
    if node is None:
        return None, False, False, None
    is_ptr = False
    is_arr = False
    size: int | None = None
    current = node
    name: str | None = None
    guard = 0
    while current is not None and guard < 12:
        guard += 1
        if current.type == "identifier":
            name = ctx.text(current)
            break
        if current.type == "pointer_declarator":
            is_ptr = True
        if current.type == "array_declarator":
            is_arr = True
            size_node = current.child_by_field_name("size") if hasattr(current, "child_by_field_name") else None
            if size_node is None:
                for ch in current.children:
                    if ch.type == "number_literal":
                        size_node = ch
            if size_node is not None and size_node.type == "number_literal":
                try:
                    size = int(ctx.text(size_node), 0)
                except ValueError:
                    size = None
        # descend
        nxt = None
        if hasattr(current, "child_by_field_name"):
            nxt = current.child_by_field_name("declarator")
        if nxt is None:
            for ch in current.children:
                if ch.type in {
                    "identifier",
                    "pointer_declarator",
                    "array_declarator",
                    "parenthesized_declarator",
                }:
                    nxt = ch
                    break
        current = nxt
    return name, is_ptr, is_arr, size


def _init_declarator_name(ctx: FunctionContext, node: Any) -> str | None:
    decl = node.child_by_field_name("declarator") if hasattr(node, "child_by_field_name") else None
    name, *_rest = _parse_declarator(ctx, decl if decl is not None else node)
    return name


def _finding(
    ctx: FunctionContext,
    *,
    rule_id: str,
    bug_type: str,
    cwe: str,
    severity: str,
    node: Any | None,
    line: int | None = None,
    message: str,
    why: str,
    fix: str,
    snippet: str,
    confidence: float,
) -> Finding:
    ln = line if line is not None else (ctx.line(node) if node is not None else ctx.fn.start_line)
    el = ctx.end_line(node) if node is not None else ln
    return Finding(
        rule_id=rule_id,
        bug_type=bug_type,
        cwe=cwe,
        severity=severity,
        line=ln,
        end_line=el,
        function=ctx.name,
        message=message,
        why_it_is_a_bug=why,
        fix_suggestion=fix,
        fixed_code_snippet=snippet,
        confidence=confidence,
        mode="ast",
    )


def _null_checked(ctx: FunctionContext, var: str) -> bool:
    pattern = re.compile(
        rf"\b((if|while|for)\s*\([^)]*\b{re.escape(var)}\b[^)]*\)|{re.escape(var)}\s*!=\s*NULL|NULL\s*!=\s*{re.escape(var)}|{re.escape(var)}\s*==\s*NULL|!\s*{re.escape(var)}\b)",
    )
    return bool(pattern.search(ctx.fn.source))


def _assigned_after(ctx: FunctionContext, var: str, after: int | Any) -> bool:
    for node in walk(ctx.fn.node):
        if node.type != "assignment_expression":
            continue
        if isinstance(after, int):
            if ctx.line(node) <= after:
                continue
        else:
            if getattr(node, "start_byte", 0) <= getattr(after, "end_byte", 0):
                continue
        left = node.child_by_field_name("left") if hasattr(node, "child_by_field_name") else node.children[0]
        if left is not None and left.type == "identifier" and ctx.text(left) == var:
            return True
    return False


def _uses_of(ctx: FunctionContext, var: str) -> list[Any]:
    uses = []
    if ctx.fn.node is None:
        return uses
    for node in walk(ctx.fn.node):
        if node.type == "identifier" and ctx.text(node) == var:
            parent = node.parent
            if parent is not None and parent.type in {
                "init_declarator",
                "pointer_declarator",
                "array_declarator",
                "declaration",
                "parameter_declaration",
                "function_declarator",
            }:
                continue
            uses.append(node)
    return uses


def _is_assignment_lhs(node: Any) -> bool:
    parent = node.parent
    if parent is None:
        return False
    if parent.type != "assignment_expression":
        return False
    left = parent.child_by_field_name("left") if hasattr(parent, "child_by_field_name") else None
    if left is None and parent.children:
        left = parent.children[0]
    if left is None:
        return False
    if left is node or left.start_byte == node.start_byte:
        return True
    return node.start_byte >= left.start_byte and node.end_byte <= left.end_byte


def _is_deref_context(node: Any) -> bool:
    parent = node.parent
    if parent is None:
        return False
    if parent.type in {"pointer_expression", "field_expression", "subscript_expression"}:
        return True
    if parent.type == "unary_expression":
        return True
    return False


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def rule_buf_overflow(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    for call in ctx.calls:
        if call.name in UNBOUNDED_COPY:
            if call.name == "gets":
                findings.append(
                    _finding(
                        ctx,
                        rule_id="BUF-OVERFLOW",
                        bug_type="Buffer Overflow",
                        cwe="CWE-120",
                        severity="critical",
                        node=call.node,
                        message="Unbounded gets() writes past any destination buffer.",
                        why="gets() cannot limit how many bytes it reads, so a long input overflows the stack buffer.",
                        fix="Replace gets() with fgets(buf, sizeof(buf), stdin).",
                        snippet="fgets(buf, sizeof(buf), stdin);",
                        confidence=0.95,
                    )
                )
                continue
            dest = _identifier_of(ctx, call.args[0]) if call.args else None
            findings.append(
                _finding(
                    ctx,
                    rule_id="BUF-OVERFLOW",
                    bug_type="Buffer Overflow",
                    cwe="CWE-120",
                    severity="critical",
                    node=call.node,
                    message=f"Unbounded {call.name}() can overflow the destination buffer.",
                    why=f"{call.name}() copies until a NUL byte and does not know the destination size.",
                    fix=f"Use a bounded API (snprintf/strncpy_s) and pass sizeof({dest or 'dest'}).",
                    snippet=f"snprintf({dest or 'dest'}, sizeof({dest or 'dest'}), \"%s\", src);",
                    confidence=0.92,
                )
            )
        if call.name == "scanf" and call.args:
            fmt = ctx.text(call.args[0])
            if "%s" in fmt and not re.search(r"%\d+s", fmt):
                findings.append(
                    _finding(
                        ctx,
                        rule_id="BUF-OVERFLOW",
                        bug_type="Buffer Overflow",
                        cwe="CWE-120",
                        severity="high",
                        node=call.node,
                        message='scanf("%s") has no field width and can overflow the target array.',
                        why="Without a width, scanf writes an unbounded string into a fixed buffer.",
                        fix='Use scanf("%15s", buf) or fgets with an explicit size.',
                        snippet='if (scanf("%15s", buf) != 1) { return -1; }',
                        confidence=0.85,
                    )
                )
        if call.name in {"memcpy", "strncpy", "memmove"} and len(call.args) >= 3:
            dest = call.args[0]
            n_arg = call.args[2]
            dest_name = _identifier_of(ctx, dest)
            dest_var = ctx.locals.get(dest_name or "")
            n_text = ctx.text(n_arg)
            bounded = "sizeof" in n_text
            if dest_var and dest_var.is_array and dest_var.array_size and not bounded:
                const_n = _int_literal(n_text)
                if const_n is not None and const_n <= dest_var.array_size:
                    continue
                findings.append(
                    _finding(
                        ctx,
                        rule_id="BUF-OVERFLOW",
                        bug_type="Buffer Overflow",
                        cwe="CWE-120",
                        severity="high",
                        node=call.node,
                        message=f"{call.name} length is not clearly bounded by the destination size.",
                        why="A copy longer than the destination array writes out of bounds.",
                        fix=f"Copy min(n, sizeof({dest_name})) bytes only.",
                        snippet=f"{call.name}({dest_name}, src, sizeof({dest_name}));",
                        confidence=0.7,
                    )
                )
    # constant OOB index
    if ctx.fn.node is not None:
        for node in walk(ctx.fn.node):
            if node.type != "subscript_expression":
                continue
            arg = node.child_by_field_name("argument") if hasattr(node, "child_by_field_name") else None
            idx = node.child_by_field_name("index") if hasattr(node, "child_by_field_name") else None
            if arg is None or idx is None:
                children = [c for c in node.children if c.type not in {"[", "]"}]
                if len(children) >= 2:
                    arg, idx = children[0], children[1]
            if arg is None or idx is None:
                continue
            name = _identifier_of(ctx, arg)
            local = ctx.locals.get(name or "")
            if not local or not local.array_size:
                continue
            const_i = _int_literal(ctx.text(idx))
            if const_i is not None and const_i >= local.array_size:
                findings.append(
                    _finding(
                        ctx,
                        rule_id="BUF-OVERFLOW",
                        bug_type="Out-of-bounds Write",
                        cwe="CWE-121",
                        severity="high",
                        node=node,
                        message=f"Constant index {const_i} is outside array {name}[{local.array_size}].",
                        why="A compile-time index at or past the array length is an overflow.",
                        fix=f"Index only in 0..{local.array_size - 1} or enlarge the array.",
                        snippet=f"{name}[{min(const_i, local.array_size - 1)}] = value;",
                        confidence=0.9,
                    )
                )
        # off-by-one: for (i = 0; i <= N; i++) with array[N]
        for node in walk(ctx.fn.node):
            if node.type != "for_statement":
                continue
            cond = node.child_by_field_name("condition") if hasattr(node, "child_by_field_name") else None
            if cond is None:
                continue
            cond_text = ctx.text(cond)
            m = re.search(r"(\w+)\s*<=\s*(\d+)", cond_text)
            if not m:
                continue
            _ivar, n_s = m.group(1), m.group(2)
            n_val = int(n_s)
            body = node.child_by_field_name("body") if hasattr(node, "child_by_field_name") else None
            body_text = ctx.text(body) if body is not None else ctx.text(node)
            for vname, local in ctx.locals.items():
                if local.array_size == n_val and re.search(rf"\b{vname}\s*\[", body_text):
                    findings.append(
                        _finding(
                            ctx,
                            rule_id="BUF-OVERFLOW",
                            bug_type="Off-by-one Buffer Overflow",
                            cwe="CWE-193",
                            severity="high",
                            node=node,
                            message=f"Loop uses <= {n_val} while array {vname} has length {n_val}.",
                            why="The last iteration writes index N of an N-element array (valid 0..N-1).",
                            fix=f"Iterate with i < {n_val} (or i < sizeof({vname})/sizeof({vname}[0])).",
                            snippet=f"for (int i = 0; i < {n_val}; i++) {{ {vname}[i] = 0; }}",
                            confidence=0.8,
                        )
                    )
    return findings


def rule_null_deref(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    alloc_vars: dict[str, CallSite] = {}
    for call in ctx.calls:
        if call.name in ALLOC_FUNCS and call.assigned_to:
            alloc_vars[call.assigned_to] = call
    for var, call in alloc_vars.items():
        if _null_checked(ctx, var):
            continue
        for use in _uses_of(ctx, var):
            if ctx.line(use) <= call.line:
                continue
            if _is_deref_context(use) or (
                use.parent is not None and use.parent.type == "argument_list"
            ):
                # assignment of the pointer itself is not a deref
                if use.parent is not None and use.parent.type == "assignment_expression":
                    left = use.parent.child_by_field_name("left") if hasattr(use.parent, "child_by_field_name") else None
                    if left is use:
                        continue
                findings.append(
                    _finding(
                        ctx,
                        rule_id="NULL-DEREF",
                        bug_type="NULL Pointer Dereference",
                        cwe="CWE-476",
                        severity="high",
                        node=use,
                        message=f"Pointer '{var}' from {call.name}() is used without a NULL check.",
                        why=f"{call.name}() may return NULL on failure; dereferencing it is undefined behavior.",
                        fix=f"if ({var} == NULL) {{ return NULL; }} before any use.",
                        snippet=f"if ({var} == NULL) {{\n    return NULL;\n}}\n{var}[0] = 1;",
                        confidence=0.88,
                    )
                )
                break
    # pointer parameters dereferenced with no check (low confidence)
    for name, local in ctx.locals.items():
        if not local.is_param or not local.is_pointer or local.is_array:
            continue
        if _null_checked(ctx, name):
            continue
        for use in _uses_of(ctx, name):
            if _is_deref_context(use):
                findings.append(
                    _finding(
                        ctx,
                        rule_id="NULL-DEREF",
                        bug_type="NULL Pointer Dereference",
                        cwe="CWE-476",
                        severity="low",
                        node=use,
                        message=f"Pointer parameter '{name}' is dereferenced with no NULL check.",
                        why="Callers may pass NULL; the function never tests the pointer.",
                        fix=f"if ({name} == NULL) return; before dereferencing.",
                        snippet=f"if ({name} == NULL) {{\n    return;\n}}",
                        confidence=0.4,
                    )
                )
                break
    return findings


def rule_use_after_free(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    for call in ctx.calls:
        if call.name not in {"free"} or not call.args:
            continue
        var = _identifier_of(ctx, call.args[0])
        if not var:
            continue
        call_end_byte = getattr(call.node, "end_byte", 0)
        for use in _uses_of(ctx, var):
            use_start_byte = getattr(use, "start_byte", 0)
            if ctx.line(use) < call.line:
                continue
            if ctx.line(use) == call.line and use_start_byte <= call_end_byte:
                continue
            if _is_assignment_lhs(use):
                continue
            if _assigned_after(ctx, var, call.node) and use_start_byte > _assign_byte(ctx, var, call.node):
                continue
            parent = use.parent
            if parent is not None and parent.type == "call_expression":
                continue
            if parent is not None and parent.type == "argument_list":
                # free(var) itself
                if use_start_byte <= call_end_byte:
                    continue
            findings.append(
                _finding(
                    ctx,
                    rule_id="USE-AFTER-FREE",
                    bug_type="Use After Free",
                    cwe="CWE-416",
                    severity="critical",
                    node=use,
                    message=f"'{var}' is used after free({var}).",
                    why="The object is no longer valid; any read or write is undefined behavior.",
                    fix=f"Do not use {var} after free; set {var} = NULL after freeing.",
                    snippet=f"free({var});\n{var} = NULL;",
                    confidence=0.9,
                )
            )
            break
    return findings


def _assign_byte(ctx: FunctionContext, var: str, after: Any) -> int:
    after_byte = getattr(after, "end_byte", 0) if not isinstance(after, int) else 0
    for node in walk(ctx.fn.node):
        if node.type != "assignment_expression":
            continue
        if getattr(node, "start_byte", 0) <= after_byte:
            continue
        left = node.child_by_field_name("left") if hasattr(node, "child_by_field_name") else node.children[0]
        if left is not None and left.type == "identifier" and ctx.text(left) == var:
            return getattr(node, "start_byte", 10**9)
    return 10**9


def _assign_line(ctx: FunctionContext, var: str, after: int) -> int:
    for node in walk(ctx.fn.node):
        if node.type != "assignment_expression":
            continue
        if ctx.line(node) <= after:
            continue
        left = node.child_by_field_name("left") if hasattr(node, "child_by_field_name") else node.children[0]
        if left is not None and left.type == "identifier" and ctx.text(left) == var:
            return ctx.line(node)
    return 10**9


def rule_double_free(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    last_free: dict[str, int] = {}
    for call in sorted(ctx.calls, key=lambda c: c.line):
        if call.name != "free" or not call.args:
            continue
        var = _identifier_of(ctx, call.args[0])
        if not var:
            continue
        prev = last_free.get(var)
        if prev is not None and not _assigned_after(ctx, var, prev):
            findings.append(
                _finding(
                    ctx,
                    rule_id="DOUBLE-FREE",
                    bug_type="Double Free",
                    cwe="CWE-415",
                    severity="critical",
                    node=call.node,
                    message=f"'{var}' is passed to free() twice without reassignment.",
                    why="Freeing the same allocation twice corrupts the heap.",
                    fix=f"free({var}); {var} = NULL; and skip a second free.",
                    snippet=f"free({var});\n{var} = NULL;",
                    confidence=0.9,
                )
            )
        last_free[var] = call.line
    return findings


def rule_memory_leak(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    body_text = ctx.fn.source
    for call in ctx.calls:
        if call.name not in ALLOC_FUNCS or not call.assigned_to:
            continue
        var = call.assigned_to
        local = ctx.locals.get(var)
        if local is None or local.is_param:
            continue
        # returned?
        if re.search(rf"\breturn\s+{re.escape(var)}\b", body_text):
            continue
        # stored to struct / global-looking assignment var-> or *out =
        if re.search(rf"(->|\.)\s*\w+\s*=\s*{re.escape(var)}\b", body_text):
            continue
        if re.search(rf"\*\s*\w+\s*=\s*{re.escape(var)}\b", body_text):
            continue
        # passed to another function (possible ownership transfer), except strcpy-like
        passed = False
        for other in ctx.calls:
            if other is call:
                continue
            for arg in other.args:
                if _identifier_of(ctx, arg) == var and other.name not in UNBOUNDED_COPY | {
                    "printf",
                    "fprintf",
                    "snprintf",
                    "memcpy",
                    "memmove",
                    "strncpy",
                    "strlen",
                    "strcmp",
                }:
                    passed = True
        if passed:
            continue
        if call.name in {"fopen", "fopen_s"}:
            if not any(c.name == "fclose" for c in ctx.calls):
                findings.append(
                    _finding(
                        ctx,
                        rule_id="MEMORY-LEAK",
                        bug_type="Resource Leak",
                        cwe="CWE-775",
                        severity="medium",
                        node=call.node,
                        message=f"FILE* '{var}' from fopen is never fclose()'d.",
                        why="Leaving a FILE open leaks descriptors and buffers.",
                        fix=f"fclose({var}); after you finish I/O.",
                        snippet=f"FILE *{var} = fopen(path, \"r\");\nif (!{var}) return;\n/* use */\nfclose({var});",
                        confidence=0.8,
                    )
                )
            continue
        if any(c.name == "free" and c.args and _identifier_of(ctx, c.args[0]) == var for c in ctx.calls):
            # still flag early return before free if obvious
            if _early_return_before_free(ctx, var, call.line):
                findings.append(
                    _finding(
                        ctx,
                        rule_id="MEMORY-LEAK",
                        bug_type="Memory Leak",
                        cwe="CWE-401",
                        severity="medium",
                        node=call.node,
                        message=f"'{var}' is allocated then returned on an early path without free().",
                        why="Some control-flow paths leave the allocation unreachable.",
                        fix="Free on every error path, or use a single cleanup label.",
                        snippet=f"if (err) {{\n    free({var});\n    return;\n}}",
                        confidence=0.65,
                    )
                )
            continue
        findings.append(
            _finding(
                ctx,
                rule_id="MEMORY-LEAK",
                bug_type="Memory Leak",
                cwe="CWE-401",
                severity="medium",
                node=call.node,
                message=f"Local pointer '{var}' is allocated and never freed or returned.",
                why="The heap block becomes unreachable when the function returns.",
                fix=f"free({var}); before returning, or return {var} to the caller.",
                snippet=f"char *{var} = malloc(100);\nif (!{var}) return;\nfree({var});",
                confidence=0.75,
            )
        )
    return findings


def _early_return_before_free(ctx: FunctionContext, var: str, alloc_line: int) -> bool:
    free_line = None
    for call in ctx.calls:
        if call.name == "free" and call.args and _identifier_of(ctx, call.args[0]) == var:
            free_line = call.line
            break
    if free_line is None or ctx.fn.node is None:
        return False
    for node in walk(ctx.fn.node):
        if node.type == "return_statement":
            ln = ctx.line(node)
            if alloc_line < ln < free_line:
                return True
    return False


def rule_int_overflow(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    for call in ctx.calls:
        if call.name not in {"malloc", "calloc", "realloc", "aligned_alloc"} or not call.args:
            continue
        arg = call.args[0] if call.name != "calloc" else (call.args[1] if len(call.args) > 1 else None)
        if arg is None:
            continue
        text = ctx.text(arg)
        if "*" in text and "sizeof" in text:
            # n * sizeof(x) with n not a literal
            if re.search(r"\bsizeof\b", text) and re.search(r"[A-Za-z_]\w*", text.replace("sizeof", "")):
                if re.fullmatch(r"\s*sizeof\s*\([^)]+\)\s*", text):
                    continue
                # constant * sizeof is ok
                if re.match(r"\s*\d+\s*\*\s*sizeof", text) or re.match(r"\s*sizeof[^)]*\)\s*\*\s*\d+", text):
                    continue
                if re.search(r"(SIZE_MAX|overflow|n >|n <)", ctx.fn.source):
                    continue
                findings.append(
                    _finding(
                        ctx,
                        rule_id="INT-OVERFLOW",
                        bug_type="Integer Overflow",
                        cwe="CWE-190",
                        severity="medium",
                        node=call.node,
                        message="Allocation size uses multiplication without an overflow check.",
                        why="n * sizeof(T) can wrap a size_t, producing a tiny allocation and later overflow.",
                        fix="Check n <= SIZE_MAX / sizeof(T) before multiplying.",
                        snippet="if (n > SIZE_MAX / sizeof(int)) return NULL;\np = malloc(n * sizeof(int));",
                        confidence=0.7,
                    )
                )
        if "+" in text and not re.search(r"\d+\s*\+\s*\d+", text) and any(c.isalpha() for c in text):
            findings.append(
                _finding(
                    ctx,
                    rule_id="INT-OVERFLOW",
                    bug_type="Integer Overflow",
                    cwe="CWE-190",
                    severity="low",
                    node=call.node,
                    message="Allocation size adds non-constant values without an overflow check.",
                    why="Addition of sizes can wrap and request a too-small buffer.",
                    fix="Use saturating/checked addition before allocating.",
                    snippet="if (a > SIZE_MAX - b) return NULL;\np = malloc(a + b);",
                    confidence=0.55,
                )
            )
    # signed vs unsigned size check
    if re.search(r"\(\s*int\s*\)\s*\w+\s*[<>]=?\s*(sizeof|size_t)", ctx.fn.source) or re.search(
        r"\bint\s+\w+[^;]*;\s*.*size_t", ctx.fn.source
    ):
        pass  # too noisy; only flag explicit int vs size_t compare in condition
    for node in walk(ctx.fn.node) if ctx.fn.node is not None else []:
        if node.type != "cast_expression":
            continue
        text = ctx.text(node)
        if not re.search(r"\(\s*(int|ssize_t)\s*\)", text):
            continue
        parent = node.parent
        parent_text = ctx.text(parent) if parent is not None else text
        if parent is not None and parent.type == "binary_expression" and re.search(
            r"\b(sizeof|size_t)\b", parent_text
        ):
            findings.append(
                _finding(
                    ctx,
                    rule_id="INT-OVERFLOW",
                    bug_type="Signed/Unsigned Comparison",
                    cwe="CWE-190",
                    severity="low",
                    node=node,
                    message="Size check mixes a signed cast with sizeof/size_t.",
                    why="A negative signed value becomes a huge unsigned size after conversion.",
                    fix="Compare using size_t throughout.",
                    snippet="if (n > sizeof(buf)) return -1;",
                    confidence=0.5,
                )
            )
    return findings


def rule_format_string(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    for call in ctx.calls:
        if call.name not in FORMAT_FUNCS:
            continue
        idx = FORMAT_FUNCS[call.name]
        if len(call.args) <= idx:
            continue
        fmt = call.args[idx]
        if fmt.type == "string_literal":
            continue
        if fmt.type == "identifier" or fmt.type not in {"string_literal"}:
            # concatenations of literals are ok
            if fmt.type in {"concatenated_string"}:
                continue
            if "string_literal" in {c.type for c in walk(fmt)} and fmt.type != "identifier":
                continue
            findings.append(
                _finding(
                    ctx,
                    rule_id="FORMAT-STRING",
                    bug_type="Format String",
                    cwe="CWE-134",
                    severity="high",
                    node=call.node,
                    message=f"{call.name}() uses a non-literal format string.",
                    why="If an attacker controls the format, they can read/write the stack via %n and similar.",
                    fix=f'{call.name}("%s", {ctx.text(fmt)});',
                    snippet=f'{call.name}("%s", user_input);',
                    confidence=0.9,
                )
            )
    return findings


def rule_uninit_var(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    for name, local in ctx.locals.items():
        if local.is_param or local.has_init or local.is_array or local.is_pointer:
            continue
        assigned_line: int | None = None
        for node in walk(ctx.fn.node) if ctx.fn.node else []:
            if node.type == "assignment_expression":
                left = node.child_by_field_name("left") if hasattr(node, "child_by_field_name") else None
                if left is not None and left.type == "identifier" and ctx.text(left) == name:
                    assigned_line = ctx.line(node)
                    break
        for use in _uses_of(ctx, name):
            if assigned_line is not None and ctx.line(use) >= assigned_line:
                continue
            if ctx.line(use) <= local.line:
                continue
            parent = use.parent
            if parent is not None and parent.type == "assignment_expression":
                left = parent.child_by_field_name("left") if hasattr(parent, "child_by_field_name") else parent.children[0]
                if left is use:
                    continue
            if parent is not None and parent.type in {"update_expression"}:
                pass
            findings.append(
                _finding(
                    ctx,
                    rule_id="UNINIT-VAR",
                    bug_type="Uninitialized Variable",
                    cwe="CWE-457",
                    severity="medium",
                    node=use,
                    message=f"Local '{name}' is read before it is assigned.",
                    why="Automatic storage is not zeroed; using it is undefined behavior.",
                    fix=f"Initialize {name} at declaration.",
                    snippet=f"int {name} = 0;",
                    confidence=0.7,
                )
            )
            break
    return findings


def rule_div_zero(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    if ctx.fn.node is None:
        return findings
    for node in walk(ctx.fn.node):
        if node.type != "binary_expression":
            continue
        op = None
        if hasattr(node, "child_by_field_name"):
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            for ch in node.children:
                if ctx.text(ch) in {"/", "%"}:
                    op = ctx.text(ch)
        else:
            left = right = None
        if op is None:
            text = ctx.text(node)
            if "/" not in text and "%" not in text:
                continue
            # skip pointer casts etc; require simple ident divisor
        if op not in {"/", "%"} and not (
            hasattr(node, "children")
            and any(ctx.text(c) in {"/", "%"} for c in node.children)
        ):
            continue
        if op is None:
            for ch in node.children:
                if ctx.text(ch) in {"/", "%"}:
                    op = ctx.text(ch)
            left = node.child_by_field_name("left") if hasattr(node, "child_by_field_name") else None
            right = node.child_by_field_name("right") if hasattr(node, "child_by_field_name") else None
        if right is None:
            kids = [c for c in node.children if c.type not in {"/", "%"}]
            if len(kids) >= 2:
                right = kids[1]
        if right is None:
            continue
        if right.type != "identifier":
            continue
        divisor = ctx.text(right)
        if re.search(rf"\b{re.escape(divisor)}\s*(==|!=)\s*0\b|\b0\s*(==|!=)\s*{re.escape(divisor)}\b", ctx.fn.source):
            continue
        if re.search(rf"if\s*\(\s*{re.escape(divisor)}\s*\)", ctx.fn.source):
            continue
        findings.append(
            _finding(
                ctx,
                rule_id="DIV-ZERO",
                bug_type="Division by Zero",
                cwe="CWE-369",
                severity="high",
                node=node,
                message=f"Division uses '{divisor}' with no zero check in this function.",
                why="A zero divisor raises SIGFPE / undefined behavior.",
                fix=f"if ({divisor} == 0) return 0; before dividing.",
                snippet=f"if ({divisor} == 0) {{\n    return 0;\n}}\nreturn n / {divisor};",
                confidence=0.7,
            )
        )
    return findings


def rule_unchecked_return(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    for call in ctx.calls:
        name = call.name
        interesting = name in CHECKED_RETURNS or name.startswith(PTHREAD_PREFIX)
        if not interesting:
            continue
        if call.used:
            continue
        findings.append(
            _finding(
                ctx,
                rule_id="UNCHECKED-RETURN",
                bug_type="Unchecked Return Value",
                cwe="CWE-252",
                severity="medium",
                node=call.node,
                message=f"Return value of {name}() is ignored.",
                why="Failures (NULL, short I/O, pthread errors) go unnoticed.",
                fix=f"Capture and test the return value of {name}().",
                snippet=f"void *p = {name}(size);\nif (!p) return;",
                confidence=0.75,
            )
        )
    return findings


def rule_dangerous_func(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    for call in ctx.calls:
        if call.name not in DANGEROUS:
            continue
        fix, cwe = DANGEROUS[call.name]
        findings.append(
            _finding(
                ctx,
                rule_id="DANGEROUS-FUNC",
                bug_type="Dangerous Function",
                cwe=cwe,
                severity="high" if call.name == "gets" else "medium",
                node=call.node,
                message=f"{call.name}() is unsafe or deprecated.",
                why="This API is easy to misuse and is banned in secure coding standards.",
                fix=fix,
                snippet="/* use the safer alternative described in the fix */",
                confidence=0.85,
            )
        )
    return findings


def rule_infinite_or_missing_return(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    if ctx.fn.node is None:
        return findings
    for node in walk(ctx.fn.node):
        if node.type != "while_statement":
            continue
        cond = node.child_by_field_name("condition") if hasattr(node, "child_by_field_name") else None
        cond_text = ctx.text(cond) if cond is not None else ""
        if not re.search(r"\b(1|true)\b", cond_text):
            continue
        body = node.child_by_field_name("body") if hasattr(node, "child_by_field_name") else node
        body_text = ctx.text(body)
        if re.search(r"\b(break|return|exit|goto)\b", body_text):
            continue
        findings.append(
            _finding(
                ctx,
                rule_id="INFINITE-LOOP",
                bug_type="Infinite Loop",
                cwe="CWE-835",
                severity="medium",
                node=node,
                message="while(1) loop has no break/return/exit.",
                why="The loop never terminates, hanging the program.",
                fix="Add a break/return condition, or use a bounded loop.",
                snippet="while (running) {\n    if (done) break;\n}",
                confidence=0.75,
            )
        )
    # missing return: non-void function without any return
    ret_type = _return_type(ctx)
    if ret_type and ret_type not in {"void"}:
        has_return = any(n.type == "return_statement" for n in walk(ctx.fn.node))
        if not has_return:
            findings.append(
                _finding(
                    ctx,
                    rule_id="MISSING-RETURN",
                    bug_type="Missing Return",
                    cwe="CWE-754",
                    severity="medium",
                    node=ctx.fn.node,
                    line=ctx.fn.start_line,
                    message=f"Non-void function '{ctx.name}' has no return statement.",
                    why="Falling off the end of a non-void function leaves a garbage return value.",
                    fix="Return a value on every path.",
                    snippet=f"{ret_type} {ctx.name}(void) {{\n    return 0;\n}}",
                    confidence=0.8,
                )
            )
    return findings


def _return_type(ctx: FunctionContext) -> str | None:
    if ctx.fn.node is None:
        return None
    for child in ctx.fn.node.children:
        if child.type in {"primitive_type", "type_identifier", "sized_type_specifier"}:
            return ctx.text(child)
        if child.type == "storage_class_specifier":
            continue
    text = ctx.fn.source.split("{", 1)[0]
    m = re.search(r"\b(void|int|char|long|short|float|double|size_t|unsigned|signed|bool)\b", text)
    return m.group(1) if m else None


def rule_stack_array(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    for name, local in ctx.locals.items():
        if local.is_array and local.array_size and local.array_size >= 65536:
            findings.append(
                _finding(
                    ctx,
                    rule_id="STACK-ARRAY-LARGE",
                    bug_type="Large Stack Array",
                    cwe="CWE-770",
                    severity="medium",
                    node=local.node,
                    line=local.line,
                    message=f"Stack array '{name}[{local.array_size}]' is very large.",
                    why="Huge VLAs/stack buffers can overflow the thread stack.",
                    fix="Allocate on the heap with malloc and free it.",
                    snippet=f"char *{name} = malloc({local.array_size});\nif (!{name}) return;\nfree({name});",
                    confidence=0.7,
                )
            )
    for call in ctx.calls:
        if call.name == "alloca" and call.args:
            arg = ctx.text(call.args[0])
            if re.search(r"[A-Za-z_]\w*", arg):
                findings.append(
                    _finding(
                        ctx,
                        rule_id="STACK-ARRAY-LARGE",
                        bug_type="Unbounded alloca",
                        cwe="CWE-770",
                        severity="high",
                        node=call.node,
                        message="alloca() size comes from a variable (possibly user-controlled).",
                        why="A large or attacker-controlled alloca can smash the stack.",
                        fix="Use malloc with a size cap, then free.",
                        snippet="if (n > MAX) return NULL;\np = malloc(n);",
                        confidence=0.8,
                    )
                )
    # VLA: array_declarator size is identifier
    if ctx.fn.node is not None:
        for node in walk(ctx.fn.node):
            if node.type != "array_declarator":
                continue
            size_node = node.child_by_field_name("size") if hasattr(node, "child_by_field_name") else None
            if size_node is None:
                continue
            if size_node.type == "identifier":
                findings.append(
                    _finding(
                        ctx,
                        rule_id="STACK-ARRAY-LARGE",
                        bug_type="Variable Length Array",
                        cwe="CWE-770",
                        severity="medium",
                        node=node,
                        message="VLA size is a variable and may be user-controlled.",
                        why="A huge VLA overflows the stack; VLAs are also banned in some standards.",
                        fix="Cap the size and use malloc.",
                        snippet="if (n > MAX || n == 0) return;\nchar *buf = malloc(n);",
                        confidence=0.7,
                    )
                )
    return findings


def rule_assign_in_condition(ctx: FunctionContext) -> list[Finding]:
    findings: list[Finding] = []
    if ctx.fn.node is None:
        return findings
    for node in walk(ctx.fn.node):
        if node.type not in {"if_statement", "while_statement"}:
            continue
        cond = node.child_by_field_name("condition") if hasattr(node, "child_by_field_name") else None
        if cond is None:
            continue
        # look for assignment_expression as the (only) condition expression
        inner = cond
        if cond.type == "parenthesized_expression" and cond.children:
            for ch in cond.children:
                if ch.type not in {"(", ")"}:
                    inner = ch
                    break
        if inner.type == "assignment_expression":
            findings.append(
                _finding(
                    ctx,
                    rule_id="ASSIGN-IN-CONDITION",
                    bug_type="Assignment in Condition",
                    cwe="CWE-481",
                    severity="low",
                    node=inner,
                    message="Condition uses assignment '=' instead of a comparison.",
                    why="This is often a typo for == and changes control flow unexpectedly.",
                    fix="Use == if you meant to compare, or assign before the if.",
                    snippet="if (x == y) {\n    /* ... */\n}",
                    confidence=0.8,
                )
            )
    return findings


def _int_literal(text: str) -> int | None:
    text = text.strip()
    try:
        return int(text, 0)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Regex fallback (parser failure)
# ---------------------------------------------------------------------------


def _regex_rules(ctx: FunctionContext) -> list[Finding]:
    src = ctx.fn.source
    findings: list[Finding] = []
    base = ctx.fn.start_line

    def ln(match: re.Match[str]) -> int:
        return base + src[: match.start()].count("\n")

    for rx, rule_id, bug, cwe, sev, msg, why, fix, snip, conf in _REGEX_PATTERNS:
        for m in rx.finditer(src):
            findings.append(
                _finding(
                    ctx,
                    rule_id=rule_id,
                    bug_type=bug,
                    cwe=cwe,
                    severity=sev,
                    node=None,
                    line=ln(m),
                    message=msg,
                    why=why,
                    fix=fix,
                    snippet=snip,
                    confidence=conf,
                )
            )
    for item in findings:
        item.mode = "fallback"
        item.confidence = min(item.confidence, 0.6)
    return findings


_REGEX_PATTERNS: list[tuple] = [
    (
        re.compile(r"\bstrcpy\s*\("),
        "BUF-OVERFLOW",
        "Buffer Overflow",
        "CWE-120",
        "critical",
        "Unbounded strcpy() can overflow the destination buffer.",
        "strcpy copies until NUL without a size.",
        "Use snprintf or memcpy with sizeof(dest).",
        'snprintf(dest, sizeof(dest), "%s", src);',
        0.7,
    ),
    (
        re.compile(r"\bstrcat\s*\("),
        "BUF-OVERFLOW",
        "Buffer Overflow",
        "CWE-120",
        "critical",
        "Unbounded strcat() can overflow the destination buffer.",
        "strcat appends until NUL without a size.",
        "Use strncat with remaining space or snprintf.",
        "snprintf(dest, sizeof(dest), \"%s%s\", dest, src);",
        0.7,
    ),
    (
        re.compile(r"\bsprintf\s*\("),
        "BUF-OVERFLOW",
        "Buffer Overflow",
        "CWE-120",
        "critical",
        "Unbounded sprintf() can overflow the destination buffer.",
        "sprintf does not take a destination size.",
        "Use snprintf.",
        'snprintf(dest, sizeof(dest), "%s", src);',
        0.7,
    ),
    (
        re.compile(r"\bgets\s*\("),
        "BUF-OVERFLOW",
        "Buffer Overflow",
        "CWE-120",
        "critical",
        "Unbounded gets() writes past any destination buffer.",
        "gets cannot bound its write.",
        "Use fgets.",
        "fgets(buf, sizeof(buf), stdin);",
        0.7,
    ),
    (
        re.compile(r"\bfree\s*\(\s*(\w+)\s*\)\s*;[^\n]*\n[^\n]*\1\s*\["),
        "USE-AFTER-FREE",
        "Use After Free",
        "CWE-416",
        "critical",
        "Pointer is used after free().",
        "The heap object is no longer valid.",
        "Do not use the pointer after free.",
        "free(p);\np = NULL;",
        0.65,
    ),
    (
        re.compile(r"\bfree\s*\(\s*(\w+)\s*\)\s*;[\s\S]{0,400}?\bfree\s*\(\s*\1\s*\)"),
        "DOUBLE-FREE",
        "Double Free",
        "CWE-415",
        "critical",
        "Pointer is freed twice.",
        "Double free corrupts the heap.",
        "Set the pointer to NULL after free.",
        "free(p);\np = NULL;",
        0.65,
    ),
    (
        re.compile(r"\bprintf\s*\(\s*([A-Za-z_]\w*)\s*\)"),
        "FORMAT-STRING",
        "Format String",
        "CWE-134",
        "high",
        "printf() uses a non-literal format string.",
        "Attacker-controlled format strings are dangerous.",
        'printf("%s", user_input);',
        'printf("%s", user_input);',
        0.7,
    ),
    (
        re.compile(r"\bif\s*\(\s*\w+\s*=\s*[^=]"),
        "ASSIGN-IN-CONDITION",
        "Assignment in Condition",
        "CWE-481",
        "low",
        "Condition uses assignment '=' instead of a comparison.",
        "Often a typo for ==.",
        "Use == to compare.",
        "if (x == y) { }",
        0.6,
    ),
]


def _iter_ident_nodes(node: Any) -> Iterable[Any]:
    if node is None:
        return
    yield from walk(node)
