"""Load trained artifacts and explain predictions."""

from __future__ import annotations

from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.ml.features import iter_token_spans

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
TRAIN_HINT = "Model not trained. Run python -m backend.ml.train"


class ModelNotTrainedError(RuntimeError):
    """Raised when artifacts/model.joblib is missing."""


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any] | None:
    path = ARTIFACTS / "model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def artifacts_available() -> bool:
    return (ARTIFACTS / "model.joblib").exists()


def load_metrics() -> dict[str, Any] | None:
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _codebert_signal(code: str) -> float | None:
    """Optional extra logit from a fine-tuned CodeBERT export."""
    folder = ARTIFACTS / "codebert"
    if not folder.exists():
        return None
    # Placeholder: a real export would include tokenizer + weights.
    # If only a probability sidecar exists, blend it; otherwise ignore.
    side = folder / "last_prob.json"
    if not side.exists():
        logger.info("artifacts/codebert present but unused (no inference weights).")
        return None
    try:
        payload = json.loads(side.read_text(encoding="utf-8"))
        return float(payload.get("probability"))
    except Exception:  # noqa: BLE001
        return None


def _probability(art: dict[str, Any], code: str) -> float:
    bundle = art["bundle"]
    x = bundle.transform([code])
    mode = art.get("mode", "lr")
    lr = art["lr"]
    if mode == "lr":
        cal = art["calibrated"]
        return float(cal.predict_proba(x)[0, 1])
    numeric_width = int(art.get("numeric_width", 13))
    svd = art["svd"]
    challenger = art["challenger"]
    wc = x[:, : x.shape[1] - numeric_width]
    num = x[:, -numeric_width:].toarray()
    ch_x = np.hstack([svd.transform(wc), num])
    if mode == "challenger":
        cal = art["calibrated"]
        return float(cal.predict_proba(ch_x)[0, 1])
    raw = 0.5 * lr.predict_proba(x)[0, 1] + 0.5 * challenger.predict_proba(ch_x)[0, 1]
    cal = art["calibrated"]
    if hasattr(cal, "predict"):
        return float(np.clip(cal.predict([raw])[0], 0.0, 1.0))
    return float(raw)


def _explain(art: dict[str, Any], code: str) -> list[dict[str, Any]]:
    bundle = art["bundle"]
    lr = art["lr"]
    word_x = bundle.word_transform([code])
    names = bundle.word.get_feature_names_out()
    coef = lr.coef_.ravel()[: len(names)]
    row = word_x.tocsr()
    contribs: list[tuple[str, float]] = []
    start, end = row.indptr[0], row.indptr[1]
    for idx, val in zip(row.indices[start:end], row.data[start:end]):
        if idx >= len(names):
            continue
        weight = float(coef[idx] * val)
        contribs.append((str(names[idx]), weight))
    contribs.sort(key=lambda t: t[1], reverse=True)
    risk = contribs[:10]
    protect = sorted(contribs, key=lambda t: t[1])[:5]
    chosen = risk + [(t, w) for t, w in protect if (t, w) not in risk]

    spans: dict[str, list[dict[str, int]]] = {}
    for token, line, start_col, end_col in iter_token_spans(code):
        spans.setdefault(token, []).append(
            {"line": line, "start_col": start_col, "end_col": end_col}
        )
        # also map unigrams inside n-grams later
    out: list[dict[str, Any]] = []
    for token, weight in chosen:
        positions: list[dict[str, int]] = []
        if token in spans:
            positions = spans[token][:8]
        else:
            parts = token.split()
            if len(parts) == 1 and parts[0] in spans:
                positions = spans[parts[0]][:8]
            else:
                for part in parts:
                    positions.extend(spans.get(part, [])[:3])
        out.append({"token": token, "weight": weight, "positions": positions[:8]})
    return out


def predict(code: str) -> dict[str, Any]:
    """Return defect probability, label, and contributing tokens."""
    art = load_artifacts()
    if art is None:
        raise ModelNotTrainedError(TRAIN_HINT)
    prob = _probability(art, code or "")
    extra = _codebert_signal(code or "")
    if extra is not None:
        prob = float(np.clip(0.7 * prob + 0.3 * extra, 0.0, 1.0))
    label = int(prob >= 0.5)
    return {
        "probability": prob,
        "label": label,
        "top_tokens": _explain(art, code or ""),
    }
