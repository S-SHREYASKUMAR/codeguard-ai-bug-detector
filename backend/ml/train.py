"""Download CodeXGLUE defect data, train CPU models, write artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from backend.ml.features import FeatureBundle, fit_features, preprocess

logger = logging.getLogger("backend.ml.train")

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MANUAL_DATA_HELP = """
Could not download google/code_x_glue_cc_defect_detection.

Manual fallback:
  1. Clone https://github.com/microsoft/CodeXGLUE
  2. Copy Code-Code/Defect-detection/dataset/{train,valid,test}.jsonl
     into a local folder (each line: {{"func": "...", "target": 0|1}})
  3. Re-run: python -m backend.ml.train --data-dir PATH_TO_JSONL_FOLDER
"""

PUBLISHED_BASELINES = [
    {"name": "CodeBERT", "accuracy": 0.621, "note": "published reference value"},
    {"name": "RoBERTa", "accuracy": 0.610, "note": "published reference value"},
    {"name": "BiLSTM", "accuracy": 0.594, "note": "published reference value"},
    {"name": "TextCNN", "accuracy": 0.607, "note": "published reference value"},
]


def artifacts_dir() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS


def _to_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return 1 if value.lower() in {"1", "true", "yes"} else 0
    return int(value)


def load_jsonl_split(path: Path) -> tuple[list[str], np.ndarray]:
    funcs: list[str] = []
    labels: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            funcs.append(row.get("func") or row.get("code") or "")
            labels.append(_to_label(row.get("target", row.get("label", 0))))
    return funcs, np.array(labels, dtype=np.int32)


def load_local_jsonl(data_dir: Path) -> dict[str, tuple[list[str], np.ndarray]]:
    mapping = {"train": "train.jsonl", "validation": "valid.jsonl", "test": "test.jsonl"}
    alt = {"validation": "validation.jsonl"}
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for split, name in mapping.items():
        path = data_dir / name
        if not path.exists() and split in alt:
            path = data_dir / alt[split]
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        out[split] = load_jsonl_split(path)
    return out


def load_codexglue() -> dict[str, tuple[list[str], np.ndarray]]:
    from datasets import load_dataset

    ds = load_dataset("google/code_x_glue_cc_defect_detection")
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        cols = ds[split]
        funcs = list(cols["func"])
        labels = np.array([_to_label(x) for x in cols["target"]], dtype=np.int32)
        out[split] = (funcs, labels)
    return out


def _norm_hash(code: str) -> str:
    return hashlib.sha1(preprocess(code).encode("utf-8")).hexdigest()


def maybe_load_diversevul(train_funcs: list[str], train_y: np.ndarray, test_funcs: list[str]) -> tuple[list[str], np.ndarray]:
    """Append DiverseVul C/C++ functions to TRAIN only; skip on failure."""
    test_hashes = {_norm_hash(c) for c in test_funcs}
    train_hashes = {_norm_hash(c) for c in train_funcs}
    candidates = [
        "claudios/DiverseVul",
        "DiverseVul/DiverseVul",
        "benjis/diversevul",
    ]
    last_err: Exception | None = None
    for dataset_id in candidates:
        try:
            from datasets import load_dataset

            print(f"Trying extra data {dataset_id} ...")
            ds = load_dataset(dataset_id, split="train")
            added_f: list[str] = []
            added_y: list[int] = []
            for row in ds:
                lang = str(row.get("lang") or row.get("language") or row.get("c_cpp") or "c").lower()
                if lang not in {"c", "c++", "cpp", "cc", "1", "true", ""}:
                    # DiverseVul is C/C++ focused; keep rows that look like C
                    func = str(row.get("func") or row.get("code") or "")
                    if not func or ("int " not in func and "void " not in func and "*" not in func):
                        continue
                func = str(row.get("func") or row.get("code") or "")
                if not func.strip():
                    continue
                h = _norm_hash(func)
                if h in test_hashes or h in train_hashes:
                    continue
                target = row.get("target", row.get("label", row.get("vulnerable")))
                if target is None:
                    continue
                train_hashes.add(h)
                added_f.append(func)
                added_y.append(_to_label(target))
            if not added_f:
                print(f"Warning: {dataset_id} loaded but no extra rows kept.")
                continue
            print(f"Appended {len(added_f)} DiverseVul functions to train only.")
            return train_funcs + added_f, np.concatenate([train_y, np.array(added_y, dtype=np.int32)])
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"Warning: could not load {dataset_id}: {exc}")
    print(f"Warning: DiverseVul skipped ({last_err}).")
    return train_funcs, train_y


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5
    cm = confusion_matrix(y_true, y_pred).tolist()
    fpr, tpr, _thr = roc_curve(y_true, y_prob)
    # downsample ROC for JSON
    step = max(1, len(fpr) // 80)
    roc = {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()}
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "roc": roc,
    }


def _top_tokens(bundle: FeatureBundle, lr: LogisticRegression, k: int = 25) -> dict[str, list[dict[str, Any]]]:
    names = bundle.word.get_feature_names_out()
    coef = lr.coef_.ravel()[: len(names)]
    order = np.argsort(coef)
    neg = [{"token": str(names[i]), "weight": float(coef[i])} for i in order[:k]]
    pos = [{"token": str(names[i]), "weight": float(coef[i])} for i in order[-k:][::-1]]
    return {"positive": pos, "negative": neg}


def _try_lightgbm(x: np.ndarray, y: np.ndarray):
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(x, y)
        return model
    except Exception as exc:  # noqa: BLE001
        print(f"LightGBM unavailable ({exc}); using HistGradientBoosting.")
        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=80,
            class_weight="balanced",
        )
        model.fit(x, y)
        return model


def train(
    *,
    quick: bool = False,
    extra_diversevul: bool = False,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    print("Loading dataset...")
    data = None
    if data_dir is not None:
        data = load_local_jsonl(data_dir)
    else:
        try:
            data = load_codexglue()
        except Exception as exc:  # noqa: BLE001
            print(MANUAL_DATA_HELP)
            raise SystemExit(f"Dataset download failed: {exc}") from exc

    train_f, train_y = data["train"]
    val_f, val_y = data["validation"]
    test_f, test_y = data["test"]
    if extra_diversevul:
        train_f, train_y = maybe_load_diversevul(train_f, train_y, test_f)

    if quick:
        rng = np.random.RandomState(0)
        tr_idx = rng.choice(len(train_f), size=min(5000, len(train_f)), replace=False)
        va_idx = rng.choice(len(val_f), size=min(800, len(val_f)), replace=False)
        te_idx = rng.choice(len(test_f), size=min(800, len(test_f)), replace=False)
        train_f = [train_f[i] for i in tr_idx]
        train_y = train_y[tr_idx]
        val_f = [val_f[i] for i in va_idx]
        val_y = val_y[va_idx]
        test_f = [test_f[i] for i in te_idx]
        test_y = test_y[te_idx]
        word_max, char_max = 20_000, 10_000
        print("Quick mode: 5k train subset, reduced TF-IDF features.")
    else:
        word_max, char_max = 100_000, 50_000

    stats = {
        "train": {"n": len(train_f), "positive": int(train_y.sum()), "negative": int((1 - train_y).sum())},
        "validation": {"n": len(val_f), "positive": int(val_y.sum()), "negative": int((1 - val_y).sum())},
        "test": {"n": len(test_f), "positive": int(test_y.sum()), "negative": int((1 - test_y).sum())},
        "source": "google/code_x_glue_cc_defect_detection",
        "quick": quick,
    }
    print(f"Splits train={stats['train']['n']} val={stats['validation']['n']} test={stats['test']['n']}")

    print("Fitting features...")
    bundle = fit_features(train_f, word_max_features=word_max, char_max_features=char_max)
    x_train = bundle.transform(train_f)
    x_val = bundle.transform(val_f)
    x_test = bundle.transform(test_f)

    print("Training logistic regression (tuning C on validation)...")
    best_lr = None
    best_lr_score = -1.0
    for c_value in ([1.0] if quick else [0.25, 1.0, 4.0]):
        lr = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=200,
            solver="liblinear",
        )
        lr.fit(x_train, train_y)
        val_prob = lr.predict_proba(x_val)[:, 1]
        score = f1_score(val_y, (val_prob >= 0.5).astype(int), zero_division=0)
        acc = accuracy_score(val_y, (val_prob >= 0.5).astype(int))
        print(f"  LR C={c_value}: val acc={acc:.4f} f1={score:.4f}")
        if score > best_lr_score:
            best_lr_score = score
            best_lr = lr
    assert best_lr is not None

    print("Training challenger on SVD-reduced features...")
    n_comp = 32 if quick else 64
    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    word_char_train = x_train[:, : x_train.shape[1] - 13]
    word_char_val = x_val[:, : x_val.shape[1] - 13]
    word_char_test = x_test[:, : x_test.shape[1] - 13]
    num_train = x_train[:, -13:].toarray()
    num_val = x_val[:, -13:].toarray()
    num_test = x_test[:, -13:].toarray()
    svd.fit(word_char_train)
    ch_train = np.hstack([svd.transform(word_char_train), num_train])
    ch_val = np.hstack([svd.transform(word_char_val), num_val])
    ch_test = np.hstack([svd.transform(word_char_test), num_test])
    challenger = _try_lightgbm(ch_train, train_y)
    ch_val_prob = challenger.predict_proba(ch_val)[:, 1]
    ch_f1 = f1_score(val_y, (ch_val_prob >= 0.5).astype(int), zero_division=0)
    ch_acc = accuracy_score(val_y, (ch_val_prob >= 0.5).astype(int))
    print(f"  Challenger val acc={ch_acc:.4f} f1={ch_f1:.4f}")

    lr_val_prob = best_lr.predict_proba(x_val)[:, 1]
    blend_val = 0.5 * lr_val_prob + 0.5 * ch_val_prob
    blend_f1 = f1_score(val_y, (blend_val >= 0.5).astype(int), zero_division=0)
    blend_acc = accuracy_score(val_y, (blend_val >= 0.5).astype(int))
    print(f"  Blend val acc={blend_acc:.4f} f1={blend_f1:.4f}")

    mode = "lr"
    best_name = "logistic_regression"
    if ch_f1 >= best_lr_score and ch_acc >= accuracy_score(val_y, (lr_val_prob >= 0.5).astype(int)):
        mode = "challenger"
        best_name = type(challenger).__name__
    if blend_f1 >= max(best_lr_score, ch_f1):
        mode = "blend"
        best_name = "lr_challenger_blend"
    print(f"Selected {best_name} ({mode})")

    print("Calibrating probabilities on validation...")
    if mode == "lr":
        calibrated = CalibratedClassifierCV(best_lr, method="isotonic", cv="prefit")
        calibrated.fit(x_val, val_y)
        test_prob = calibrated.predict_proba(x_test)[:, 1]
        val_prob_cal = calibrated.predict_proba(x_val)[:, 1]
    elif mode == "challenger":
        calibrated = CalibratedClassifierCV(challenger, method="isotonic", cv="prefit")
        calibrated.fit(ch_val, val_y)
        test_prob = calibrated.predict_proba(ch_test)[:, 1]
        val_prob_cal = calibrated.predict_proba(ch_val)[:, 1]
    else:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(blend_val, val_y)
        calibrated = iso
        test_blend = 0.5 * best_lr.predict_proba(x_test)[:, 1] + 0.5 * challenger.predict_proba(ch_test)[:, 1]
        test_prob = iso.predict(test_blend)
        val_prob_cal = iso.predict(blend_val)

    test_m = _metrics(test_y, test_prob)
    val_m = _metrics(val_y, val_prob_cal)
    elapsed = time.perf_counter() - t0
    print(
        f"TEST accuracy={test_m['accuracy']:.4f} f1={test_m['f1']:.4f} "
        f"roc_auc={test_m['roc_auc']:.4f} ({elapsed:.1f}s)"
    )

    payload = {
        "bundle": bundle,
        "lr": best_lr,
        "challenger": challenger,
        "svd": svd,
        "calibrated": calibrated,
        "mode": mode,
        "numeric_width": 13,
    }
    out_dir = artifacts_dir()
    joblib.dump(payload, out_dir / "model.joblib")
    metrics = {
        "test": test_m,
        "validation": val_m,
        "dataset": stats,
        "training_time_sec": elapsed,
        "selected_model": best_name,
        "mode": mode,
        "quick": quick,
        "top_tokens": _top_tokens(bundle, best_lr),
        "published_baselines": PUBLISHED_BASELINES,
        "notes": (
            "Published baseline accuracies are reference values from the CodeXGLUE "
            "defect-detection literature; verify against the current leaderboard."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'model.joblib'} and {out_dir / 'metrics.json'}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train CodeGuard defect classifier")
    parser.add_argument("--quick", action="store_true", help="5k subset smoke run")
    parser.add_argument("--extra-data", choices=["diversevul"], default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder with train/valid/test.jsonl")
    args = parser.parse_args(argv)
    train(
        quick=args.quick,
        extra_diversevul=args.extra_data == "diversevul",
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
