"""Demo C snippets for the UI sample dropdown."""

from __future__ import annotations

SAMPLES: list[dict[str, str]] = [
    {
        "id": "strcpy",
        "title": "Buffer overflow (strcpy)",
        "code": (
            "#include <stdio.h>\n#include <string.h>\n"
            "void greet(char *name) {\n"
            "    char buf[16];\n"
            "    strcpy(buf, name);\n"
            "    printf(\"Hello %s\\n\", buf);\n"
            "}\n"
        ),
    },
    {
        "id": "null",
        "title": "NULL deref after malloc",
        "code": (
            "#include <stdlib.h>\n"
            "int *make(int n) {\n"
            "    int *p = malloc(n * sizeof(int));\n"
            "    p[0] = 1;\n"
            "    return p;\n"
            "}\n"
        ),
    },
    {
        "id": "uaf",
        "title": "Use after free",
        "code": (
            "#include <stdlib.h>\n"
            "void uaf(void) {\n"
            "    char *p = malloc(8);\n"
            "    if (!p) return;\n"
            "    free(p);\n"
            "    p[0] = 1;\n"
            "}\n"
        ),
    },
    {
        "id": "fmt",
        "title": "Format string",
        "code": (
            "#include <stdio.h>\n"
            "void log_msg(char *user_input) {\n"
            "    printf(user_input);\n"
            "}\n"
        ),
    },
    {
        "id": "df",
        "title": "Double free",
        "code": (
            "#include <stdlib.h>\n"
            "void df(void) {\n"
            "    char *p = malloc(8);\n"
            "    if (!p) return;\n"
            "    free(p);\n"
            "    free(p);\n"
            "}\n"
        ),
    },
    {
        "id": "leak",
        "title": "Memory leak",
        "code": (
            "#include <stdlib.h>\n"
            "void leak(void) {\n"
            "    char *s = malloc(100);\n"
            "    if (s) s[0] = 'x';\n"
            "}\n"
        ),
    },
    {
        "id": "div0",
        "title": "Division by zero",
        "code": "int divn(int n, int d) {\n    return n / d;\n}\n",
    },
    {
        "id": "assign",
        "title": "Assignment in condition",
        "code": (
            "void cmp(int x, int y) {\n"
            "    if (x = y) {\n"
            "        x = 0;\n"
            "    }\n"
            "}\n"
        ),
    },
    {
        "id": "clean_small",
        "title": "Clean: bounded copy",
        "code": (
            "#include <stdio.h>\n#include <string.h>\n"
            "void greet(const char *name) {\n"
            "    char buf[16];\n"
            "    if (name == NULL) return;\n"
            "    snprintf(buf, sizeof(buf), \"%s\", name);\n"
            "    printf(\"Hello %s\\n\", buf);\n"
            "}\n"
        ),
    },
    {
        "id": "clean_alloc",
        "title": "Clean: checked malloc/free",
        "code": (
            "#include <stdlib.h>\n"
            "int *make(size_t n) {\n"
            "    if (n == 0 || n > 1024) return NULL;\n"
            "    int *p = malloc(n * sizeof(int));\n"
            "    if (p == NULL) return NULL;\n"
            "    p[0] = 1;\n"
            "    return p;\n"
            "}\n"
            "void done(int *p) {\n"
            "    free(p);\n"
            "}\n"
        ),
    },
    {
        "id": "multi",
        "title": "Realistic multi-function file",
        "code": (
            "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n"
            "void greet(char *name) {\n"
            "    char buf[16];\n"
            "    strcpy(buf, name);\n"
            "    printf(\"Hello %s\\n\", buf);\n"
            "}\n\n"
            "int *make(int n) {\n"
            "    int *p = malloc(n * sizeof(int));\n"
            "    p[0] = 1;\n"
            "    return p;\n"
            "}\n\n"
            "void leak(void) {\n"
            "    char *s = malloc(100);\n"
            "    if (s) strcpy(s, \"x\");\n"
            "}\n\n"
            "static int safe_add(int a, int b) {\n"
            "    return a + b;\n"
            "}\n"
        ),
    },
]


def all_samples() -> list[dict[str, str]]:
    return SAMPLES
