"""API tests for CodeGuard FastAPI routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

ACCEPT_CODE = """
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

SAFE_CODE = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void greet(const char *name) {
    char buf[32];
    if (name == NULL) return;
    snprintf(buf, sizeof(buf), "%s", name);
    printf("Hello %s\\n", buf);
}
int *make(void) {
    int *p = malloc(8);
    if (p == NULL) return NULL;
    p[0] = 1;
    return p;
}
void done(int *p) {
    free(p);
}
"""


def test_health() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "model_ready" in body


def test_samples() -> None:
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 8


def test_analyze_acceptance_bugs() -> None:
    res = client.post("/api/analyze", json={"code": ACCEPT_CODE, "filename": "demo.c"})
    assert res.status_code == 200
    body = res.json()
    rules = {f["rule_id"] for f in body["findings"]}
    assert "BUF-OVERFLOW" in rules
    assert "NULL-DEREF" in rules
    assert "MEMORY-LEAK" in rules
    strcpy = [f for f in body["findings"] if f["rule_id"] == "BUF-OVERFLOW"]
    assert any(f["line"] >= 1 for f in strcpy)


def test_use_after_free() -> None:
    code = "void f(void) { char *p = malloc(8); free(p); p[0] = 1; }\n"
    res = client.post("/api/analyze", json={"code": code})
    assert res.status_code == 200
    assert any(f["cwe"] == "CWE-416" or f["rule_id"] == "USE-AFTER-FREE" for f in res.json()["findings"])


def test_format_string() -> None:
    code = "void f(char *user_input) { printf(user_input); }\n"
    res = client.post("/api/analyze", json={"code": code})
    assert res.status_code == 200
    assert any(f["cwe"] == "CWE-134" for f in res.json()["findings"])


def test_safe_function_not_high_critical() -> None:
    res = client.post("/api/analyze", json={"code": SAFE_CODE})
    assert res.status_code == 200
    body = res.json()
    bad = [f for f in body["findings"] if f["severity"] in {"critical", "high"}]
    assert not bad, bad
    assert body["verdict"] in {"Likely Clean", "Suspicious", "Suspicious (pattern-based)"}


def test_python_warning() -> None:
    res = client.post("/api/analyze", json={"code": "def foo():\n    print(1)\n"})
    assert res.status_code == 200
    body = res.json()
    assert body["language"]["warning"]
    assert "C" in body["language"]["warning"] or "unreliable" in body["language"]["warning"]


def test_too_large() -> None:
    huge = "int x;\n" + ("x+=1;\n" * 50000)
    res = client.post("/api/analyze", json={"code": huge})
    assert res.status_code == 422


def test_empty_ok() -> None:
    res = client.post("/api/analyze", json={"code": ""})
    assert res.status_code == 200
