"""Unit tests: each static rule has a buggy snippet and a clean counterpart."""

from __future__ import annotations

import pytest

from backend.analyzer.language import detect_language
from backend.analyzer.parser import parse_source
from backend.analyzer.rules import Finding, analyze_code


def ids(code: str) -> set[str]:
    return {f.rule_id for f in analyze_code(code)}


def findings_for(code: str, rule_id: str) -> list[Finding]:
    return [f for f in analyze_code(code) if f.rule_id == rule_id]


def assert_rule_fires(buggy: str, fixed: str, rule_id: str) -> None:
    bads = findings_for(buggy, rule_id)
    goods = findings_for(fixed, rule_id)
    assert bads, f"{rule_id} should fire on buggy snippet; got {ids(buggy)}"
    assert not goods, f"{rule_id} should be silent on fixed snippet; got {goods}"
    for item in bads:
        assert 0.0 <= item.confidence <= 1.0
        assert item.line >= 1
        assert item.cwe.startswith("CWE-")
        assert item.severity in {"critical", "high", "medium", "low"}


BUF_BUGGY = """
#include <string.h>
void greet(char *name) {
    char buf[16];
    strcpy(buf, name);
}
"""

BUF_FIXED = """
#include <stdio.h>
void greet(char *name) {
    char buf[16];
    if (name == NULL) return;
    snprintf(buf, sizeof(buf), "%s", name);
}
"""

NULL_BUGGY = """
#include <stdlib.h>
int *make(void) {
    int *p = malloc(8);
    p[0] = 1;
    return p;
}
"""

NULL_FIXED = """
#include <stdlib.h>
int *make(void) {
    int *p = malloc(8);
    if (p == NULL) {
        return NULL;
    }
    p[0] = 1;
    return p;
}
"""

UAF_BUGGY = """
#include <stdlib.h>
void uaf(void) {
    char *p = malloc(8);
    if (p == NULL) return;
    free(p);
    p[0] = 1;
}
"""

UAF_FIXED = """
#include <stdlib.h>
void uaf(void) {
    char *p = malloc(8);
    if (p == NULL) return;
    p[0] = 1;
    free(p);
    p = NULL;
}
"""

DF_BUGGY = """
#include <stdlib.h>
void df(void) {
    char *p = malloc(8);
    if (p == NULL) return;
    free(p);
    free(p);
}
"""

DF_FIXED = """
#include <stdlib.h>
void df(void) {
    char *p = malloc(8);
    if (p == NULL) return;
    free(p);
    p = NULL;
}
"""

LEAK_BUGGY = """
#include <stdlib.h>
#include <string.h>
void leak(void) {
    char *s = malloc(100);
    if (s) {
        s[0] = 'x';
        s[1] = 0;
    }
}
"""

LEAK_FIXED = """
#include <stdlib.h>
void leak(void) {
    char *s = malloc(100);
    if (s == NULL) return;
    s[0] = 'x';
    free(s);
}
"""

INT_BUGGY = """
#include <stdlib.h>
void *grow(size_t n) {
    return malloc(n * sizeof(int));
}
"""

INT_FIXED = """
#include <stdlib.h>
#include <stdint.h>
void *grow(size_t n) {
    if (n > SIZE_MAX / sizeof(int)) {
        return NULL;
    }
    return malloc(n * sizeof(int));
}
"""

FMT_BUGGY = """
#include <stdio.h>
void log_msg(char *user_input) {
    printf(user_input);
}
"""

FMT_FIXED = """
#include <stdio.h>
void log_msg(char *user_input) {
    printf("%s", user_input);
}
"""

UNINIT_BUGGY = """
int uses_x(void) {
    int x;
    return x;
}
"""

UNINIT_FIXED = """
int uses_x(void) {
    int x = 0;
    return x;
}
"""

DIV_BUGGY = """
int divn(int n, int d) {
    return n / d;
}
"""

DIV_FIXED = """
int divn(int n, int d) {
    if (d == 0) {
        return 0;
    }
    return n / d;
}
"""

UNCHK_BUGGY = """
#include <stdlib.h>
void ignore(void) {
    malloc(32);
}
"""

UNCHK_FIXED = """
#include <stdlib.h>
void ignore(void) {
    void *p = malloc(32);
    if (p == NULL) {
        return;
    }
    free(p);
}
"""

DANG_BUGGY = """
#include <stdlib.h>
int parse(char *s) {
    return atoi(s);
}
"""

DANG_FIXED = """
#include <stdlib.h>
int parse(char *s) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s) {
        return 0;
    }
    return (int)v;
}
"""

LOOP_BUGGY = """
void hang(void) {
    while (1) {
        int x = 0;
        x = x + 1;
    }
}
"""

LOOP_FIXED = """
void hang(int running) {
    while (running) {
        if (!running) {
            break;
        }
        running = 0;
    }
}
"""

MISS_BUGGY = """
int answer(void) {
    int x = 1;
    x = x + 1;
}
"""

MISS_FIXED = """
int answer(void) {
    int x = 1;
    x = x + 1;
    return x;
}
"""

STACK_BUGGY = """
#include <alloca.h>
void big(int n) {
    char *p = alloca(n);
    p[0] = 1;
}
"""

STACK_FIXED = """
#include <stdlib.h>
void big(int n) {
    if (n <= 0 || n > 1024) {
        return;
    }
    char *p = malloc((size_t)n);
    if (p == NULL) {
        return;
    }
    p[0] = 1;
    free(p);
}
"""

ASSIGN_BUGGY = """
void cmp(int x, int y) {
    if (x = y) {
        x = 0;
    }
}
"""

ASSIGN_FIXED = """
void cmp(int x, int y) {
    if (x == y) {
        x = 0;
    }
}
"""


@pytest.mark.parametrize(
    "buggy,fixed,rule_id",
    [
        (BUF_BUGGY, BUF_FIXED, "BUF-OVERFLOW"),
        (NULL_BUGGY, NULL_FIXED, "NULL-DEREF"),
        (UAF_BUGGY, UAF_FIXED, "USE-AFTER-FREE"),
        (DF_BUGGY, DF_FIXED, "DOUBLE-FREE"),
        (LEAK_BUGGY, LEAK_FIXED, "MEMORY-LEAK"),
        (INT_BUGGY, INT_FIXED, "INT-OVERFLOW"),
        (FMT_BUGGY, FMT_FIXED, "FORMAT-STRING"),
        (UNINIT_BUGGY, UNINIT_FIXED, "UNINIT-VAR"),
        (DIV_BUGGY, DIV_FIXED, "DIV-ZERO"),
        (UNCHK_BUGGY, UNCHK_FIXED, "UNCHECKED-RETURN"),
        (DANG_BUGGY, DANG_FIXED, "DANGEROUS-FUNC"),
        (LOOP_BUGGY, LOOP_FIXED, "INFINITE-LOOP"),
        (MISS_BUGGY, MISS_FIXED, "MISSING-RETURN"),
        (STACK_BUGGY, STACK_FIXED, "STACK-ARRAY-LARGE"),
        (ASSIGN_BUGGY, ASSIGN_FIXED, "ASSIGN-IN-CONDITION"),
    ],
)
def test_rule_buggy_and_fixed(buggy: str, fixed: str, rule_id: str) -> None:
    assert_rule_fires(buggy, fixed, rule_id)


def test_acceptance_strcpy_null_leak() -> None:
    code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void greet(char *name) {
    char buf[16];
    strcpy(buf, name);
    printf("Hello %s\\n", buf);
}
int *make(int n) {
    int *p = malloc(n * sizeof(int));
    p[0] = 1;
    return p;
}
void leak(void) {
    char *s = malloc(100);
    if (s) strcpy(s, "x");
}
"""
    got = ids(code)
    assert "BUF-OVERFLOW" in got
    assert "NULL-DEREF" in got
    assert "MEMORY-LEAK" in got
    strcpy_hits = findings_for(code, "BUF-OVERFLOW")
    assert any("strcpy" in f.message.lower() for f in strcpy_hits)


def test_language_python_warning() -> None:
    info = detect_language("def foo():\n    print(1)\n")
    assert info.detected == "python"
    assert info.warning is not None
    analyze_code("def foo():\n    print(1)\n")  # must not crash


def test_empty_and_garbage_do_not_crash() -> None:
    assert analyze_code("") == []
    analyze_code("\x00\x01\xff")
    analyze_code("{{{{ not c")


def test_parser_extracts_functions() -> None:
    parsed = parse_source(BUF_BUGGY)
    names = {f.name for f in parsed.functions}
    assert "greet" in names


def test_off_by_one_loop() -> None:
    buggy = """
void fill(void) {
    char buf[8];
    for (int i = 0; i <= 8; i++) {
        buf[i] = 0;
    }
}
"""
    fixed = """
void fill(void) {
    char buf[8];
    for (int i = 0; i < 8; i++) {
        buf[i] = 0;
    }
}
"""
    assert_rule_fires(buggy, fixed, "BUF-OVERFLOW")
