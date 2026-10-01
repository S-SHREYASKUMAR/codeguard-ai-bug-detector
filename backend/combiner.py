"""Merge ML probabilities with static-analysis findings."""

from __future__ import annotations

from typing import Any

from backend.analyzer.language import detect_language
from backend.analyzer.parser import extract_functions
from backend.analyzer.rules import Finding, analyze_code
from backend.ml.predict import ModelNotTrainedError, artifacts_available, predict

SEVERITY_WEIGHT = {"critical": 35.0, "high": 25.0, "medium": 12.0, "low": 5.0}


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _verdict_from_risk(risk: float, *, forced_defective: bool, suspicious_ml: bool) -> str:
    if forced_defective or risk >= 55:
        return "Likely Defective"
    if suspicious_ml or risk >= 30:
        return "Suspicious"
    return "Likely Clean"


def _function_risk(
    ml_probability: float | None,
    findings: list[Finding],
) -> tuple[float, str, list[str]]:
    notes: list[str] = []
    ml_p = 0.0 if ml_probability is None else float(ml_probability)
    rule_sum = sum(SEVERITY_WEIGHT.get(f.severity, 5.0) * f.confidence for f in findings)
    rule_sum = min(rule_sum, 60.0)
    risk = _clamp(0.55 * ml_p * 100.0 + rule_sum)

    high_conf = [
        f
        for f in findings
        if f.severity in {"critical", "high"} and f.confidence >= 0.7
    ]
    ml_defective = ml_probability is not None and ml_p >= 0.5
    forced = bool(high_conf)
    if forced:
        risk = max(risk, 58.0)
        if any(f.severity == "critical" and f.confidence >= 0.7 for f in high_conf):
            risk = max(risk, 70.0)

    if ml_defective and not findings:
        notes.append(
            "The model found risky patterns but cannot name a specific bug."
        )
        return max(risk, 40.0), "Suspicious (pattern-based)", notes

    if findings and ml_probability is not None and not ml_defective:
        notes.append("Static rules fired while the ML model scored the code as clean.")

    if not findings and not ml_defective:
        notes.append("No tool can prove the absence of bugs.")
        verdict = "Likely Clean"
        return risk, verdict, notes

    verdict = _verdict_from_risk(risk, forced_defective=forced, suspicious_ml=False)
    if forced and verdict == "Likely Clean":
        verdict = "Suspicious"
    return risk, verdict, notes


def combine(code: str, filename: str | None = None) -> dict[str, Any]:
    """Run language detect, parser, rules, ML, and merge into the API payload."""
    lang = detect_language(code)
    functions = extract_functions(code)
    findings = analyze_code(code)

    ml_available = artifacts_available()
    ml_block: dict[str, Any] = {
        "available": ml_available,
        "probability": None,
        "label": None,
        "top_tokens": [],
        "message": None,
    }
    file_ml: float | None = None
    if ml_available:
        try:
            pred = predict(code)
            file_ml = float(pred["probability"])
            ml_block.update(
                {
                    "probability": file_ml,
                    "label": pred["label"],
                    "top_tokens": pred["top_tokens"],
                }
            )
        except ModelNotTrainedError as exc:
            ml_block["available"] = False
            ml_block["message"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            ml_block["available"] = False
            ml_block["message"] = f"ML inference failed: {exc}"
    else:
        ml_block["message"] = "Model not trained. Run python -m backend.ml.train"

    findings_by_fn: dict[str, list[Finding]] = {}
    for item in findings:
        findings_by_fn.setdefault(item.function, []).append(item)

    fn_payload: list[dict[str, Any]] = []
    all_notes: list[str] = []
    for fn in functions:
        fn_findings = findings_by_fn.get(fn.name, [])
        # also match findings that used <file>
        if fn.name == "<file>":
            fn_findings = findings
        ml_p: float | None = None
        if ml_block["available"]:
            try:
                ml_p = float(predict(fn.source)["probability"])
            except Exception:  # noqa: BLE001
                ml_p = file_ml
        risk, verdict, notes = _function_risk(ml_p, fn_findings)
        all_notes.extend(notes)
        fn_payload.append(
            {
                "name": fn.name,
                "start_line": fn.start_line,
                "end_line": fn.end_line,
                "risk_score": round(risk, 2),
                "verdict": verdict,
                "ml_probability": None if ml_p is None else round(ml_p, 4),
            }
        )

    if not fn_payload:
        risk, verdict, notes = _function_risk(file_ml, findings)
        all_notes.extend(notes)
        fn_payload.append(
            {
                "name": filename or "<file>",
                "start_line": 1,
                "end_line": (code or "").count("\n") + 1,
                "risk_score": round(risk, 2),
                "verdict": verdict,
                "ml_probability": None if file_ml is None else round(file_ml, 4),
            }
        )

    # File-level: worst function
    rank = {"Likely Defective": 3, "Suspicious": 2, "Suspicious (pattern-based)": 2, "Likely Clean": 1}
    worst = max(fn_payload, key=lambda f: (rank.get(f["verdict"], 0), f["risk_score"]))
    file_risk, file_verdict, file_notes = _function_risk(file_ml, findings)
    all_notes.extend(file_notes)
    if rank.get(worst["verdict"], 0) >= rank.get(file_verdict, 0):
        file_verdict = worst["verdict"]
        file_risk = max(file_risk, worst["risk_score"])
    if any(f.severity in {"critical", "high"} and f.confidence >= 0.7 for f in findings):
        if file_verdict == "Likely Clean":
            file_verdict = "Suspicious"
        if any(f.severity == "critical" and f.confidence >= 0.7 for f in findings):
            file_verdict = "Likely Defective"
            file_risk = max(file_risk, 70.0)

    unique_notes = list(dict.fromkeys(all_notes))
    summary = _summary(file_verdict, findings, file_ml)
    if lang.warning:
        unique_notes.append(lang.warning)

    return {
        "verdict": file_verdict,
        "risk_score": round(file_risk, 2),
        "ml": ml_block,
        "language": {"detected": lang.detected, "warning": lang.warning},
        "functions": fn_payload,
        "findings": [f.to_dict() for f in findings],
        "stats": {
            "lines": (code or "").count("\n") + (1 if code else 0),
            "functions": len(functions),
            "issues": len(findings),
        },
        "summary": summary,
        "notes": unique_notes,
        "filename": filename,
    }


def _summary(verdict: str, findings: list[Finding], ml_p: float | None) -> str:
    if findings:
        top = findings[0]
        return (
            f"{verdict}: {len(findings)} static finding(s); "
            f"top issue {top.bug_type} ({top.cwe}) on line {top.line}."
        )
    if verdict.startswith("Suspicious"):
        p = 0.0 if ml_p is None else ml_p
        return f"{verdict}: model probability {p:.2f} with no named static bug."
    return f"{verdict}: no high-confidence static bugs; treat this as a risk indicator, not a proof."
