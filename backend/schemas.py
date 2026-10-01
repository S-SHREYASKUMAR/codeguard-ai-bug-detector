"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    code: str = Field(..., min_length=0)
    filename: str | None = None

    @field_validator("code")
    @classmethod
    def limit_size(cls, value: str) -> str:
        encoded = value.encode("utf-8")
        if len(encoded) > 200 * 1024:
            raise ValueError("Input exceeds 200 KB limit")
        return value


class TokenHit(BaseModel):
    token: str
    weight: float
    positions: list[dict[str, Any]] = Field(default_factory=list)


class MLBlock(BaseModel):
    available: bool
    probability: float | None = None
    label: int | None = None
    top_tokens: list[TokenHit] = Field(default_factory=list)
    message: str | None = None


class LanguageBlock(BaseModel):
    detected: str
    warning: str | None = None


class FunctionBlock(BaseModel):
    name: str
    start_line: int
    end_line: int
    risk_score: float
    verdict: str
    ml_probability: float | None = None


class FindingOut(BaseModel):
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


class AnalyzeResponse(BaseModel):
    verdict: str
    risk_score: float
    ml: MLBlock
    language: LanguageBlock
    functions: list[FunctionBlock]
    findings: list[FindingOut]
    stats: dict[str, Any]
    summary: str
    notes: list[str] = Field(default_factory=list)
