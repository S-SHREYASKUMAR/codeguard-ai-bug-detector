# CodeGuard AI — AI-Based Programming Bug Detection for C Code

CodeGuard AI is a lightweight, hybrid defect detection system and developer tool for C/C++ code. It runs entirely on CPU without requiring a GPU or complex build systems.

---

## 1. Quick Start

### Prerequisites
- Python 3.10+
- Prebuilt C compiler / tree-sitter C wheels

### Setup
```bash
cd codeguard

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training the ML Model
CodeGuard trains on the official Microsoft CodeXGLUE defect detection benchmark (Devign: ~21.8k training functions from FFmpeg and QEMU).

```bash
# Smoke / Quick run (~1-2 minutes, 5k training subset):
python -m backend.ml.train --quick

# Full dataset training (under 10 minutes on CPU):
python -m backend.ml.train

# Optional: augment training with DiverseVul dataset:
python -m backend.ml.train --extra-data diversevul

# Optional: use local JSONL files if offline:
python -m backend.ml.train --data-dir path/to/jsonl_folder
```

Artifacts will be written to `codeguard/artifacts/`:
- `model.joblib`: Trained Logistic Regression and challenger models, feature extractors, and calibrated probabilities.
- `metrics.json`: Detailed accuracy, F1, ROC-AUC, confusion matrix, ROC curve, and baseline comparison.

### Running the Web Application
```bash
uvicorn backend.main:app --reload --port 8000
```
Open your browser at `http://127.0.0.1:8000/`.

### Running Tests
```bash
# Run all tests (rules + API acceptance tests):
pytest
```

---

## 2. Hybrid Architecture: How It Works

A deep-learning or machine-learning classifier trained on CodeXGLUE outputs binary predictions (clean vs. defective) at the function level. It cannot pinpoint exact line numbers or identify bug types (such as buffer overflows or NULL dereferences).

To solve this, CodeGuard combines two complementary engines:

```
Source Code
    │
    ▼
tree-sitter C AST Parser ────► Language Detector (Heuristics)
    │
    ├─────────────────────────────┬─────────────────────────────┐
    ▼                                                           ▼
Engine A: Static Analyzer                               Engine B: ML Classifier
(Conservative AST/CFG rules)                            (CodeXGLUE Logistic Regression
• CWE-120/121: Buffer Overflow                          • Word/Char TF-IDF n-grams
• CWE-476: NULL Pointer Deref                           • 13 Handcrafted code metrics
• CWE-416: Use-After-Free                               • Calibrated defect probability
• CWE-415: Double Free                                  • Feature contribution tokens
• CWE-401: Memory Leak
• CWE-190: Integer Overflow
• CWE-134: Format String
• CWE-457: Uninitialized Variable
• CWE-369: Division by Zero
• CWE-252: Unchecked Return Value
• CWE-242: Dangerous Functions
• CWE-835: Infinite Loop / Missing Return
• CWE-770: Large Stack Array / VLA
• CWE-481: Assignment in Condition
    │                                                           │
    └─────────────────────────────┬─────────────────────────────┘
                                  ▼
                         Combiner Engine
             • Computes weighted per-function risk
             • Determines overall file risk & verdict
             • Resolves disagreements & flags patterns
                                  │
                                  ▼
                     Interactive UI & REST API
```

### Risk Combination Formula
For each function:
$$\text{risk} = \text{clamp}\left(0.55 \times P(\text{defective}) \times 100 + \sum (\text{severity\_weight} \times \text{confidence}),\ 0,\ 100\right)$$
- Severity weights: Critical = 35, High = 25, Medium = 12, Low = 5 (rule contribution capped at 60).
- High-confidence critical/high findings force a verdict of "Likely Defective".
- If ML indicates defectiveness without any triggered rule, the verdict is set to "Suspicious (pattern-based)".
- If neither rules fire nor ML flags the code, the verdict is "Likely Clean" (with an explicit note that no tool can prove absence of bugs).

---

## 3. How to Add a New Rule

1. Open `backend/analyzer/rules.py`.
2. Define a rule function taking `FunctionContext`:
   ```python
   def rule_my_custom_check(ctx: FunctionContext) -> list[Finding]:
       findings: list[Finding] = []
       for node in walk(ctx.fn.node):
           # inspect AST nodes using ctx.text(node), ctx.line(node), etc.
           if is_problematic(node):
               findings.append(
                   _finding(
                       ctx,
                       rule_id="MY-RULE-ID",
                       bug_type="Bug Type Name",
                       cwe="CWE-XXX",
                       severity="high",
                       node=node,
                       message="Short explanation of issue.",
                       why="Why this is dangerous.",
                       fix="Suggested fix description.",
                       snippet="Example safe code replacement.",
                       confidence=0.85,
                   )
               )
       return findings
   ```
3. Register the rule in `_analyze_function_ast()` inside `backend/analyzer/rules.py`.
4. Add a unit test with buggy and clean snippets in `tests/test_rules.py` verifying that:
   - The buggy snippet triggers the rule.
   - The fixed snippet produces no false positives.

---

## 4. Expected Performance & Metrics

| Model / Benchmark | Accuracy | Notes |
|:------------------|:--------:|:------|
| **CodeGuard LR (Quick)** | ~61.2% | CPU, ~1.5 min training |
| **CodeGuard LR (Full)**  | ~62–64% | CPU, ~5–8 min training |
| **CodeBERT (Reference)** | ~62.1% | Published reference value on CodeXGLUE |
| **RoBERTa (Reference)**  | ~61.0% | Published reference value on CodeXGLUE |
| **TextCNN (Reference)**  | ~60.7% | Published reference value on CodeXGLUE |
| **BiLSTM (Reference)**   | ~59.4% | Published reference value on CodeXGLUE |

*Note: Published baseline accuracies are reference values from CodeXGLUE defect-detection literature; consult the CodeXGLUE leaderboard for current updates.*

---

## 5. Limitations

- **Function-level ML labels**: CodeXGLUE labels entire functions as clean or vulnerable. The statistical model identifies risk-associated patterns and tokens, but cannot name the specific vulnerability.
- **Language Scope**: Engineered specifically for C/C++. Non-C inputs produce a heuristic language warning.
- **Static Analysis Conservatism**: Static rules use AST and intra-procedural data-flow heuristics designed to minimize false positives, rather than performing full symbolic execution or inter-procedural alias tracking.
- **Expected Benchmark Difficulty**: State-of-the-art accuracy on this benchmark is in the 60%–70% range. Scores should always be treated as risk indicators rather than absolute proofs.

---

## 6. Optional CodeBERT Fine-Tuning

A Google Colab notebook is provided in `notebooks/colab_codebert_finetune.ipynb` to fine-tune `microsoft/codebert-base` on a free T4 GPU. 

If an exported model or probability file is saved to `artifacts/codebert/last_prob.json`, `backend/ml/predict.py` will automatically blend its logit signal into the inference pipeline. If absent, the system degrades gracefully and relies on the scikit-learn models.

---

## 7. REST API Endpoints

- `POST /api/analyze`: Accepts `{ "code": "...", "filename": "optional.c" }` (max 200 KB). Returns combined verdict, risk score, ML probability & tokens, language warning, function breakdown, and AST rule findings.
- `GET /api/metrics`: Returns evaluation metrics and ROC data from `artifacts/metrics.json`.
- `GET /api/samples`: Returns pre-loaded demo snippets (buffer overflow, NULL dereference, use-after-free, memory leak, format string, safe code, etc.).
- `GET /api/health`: Health status and model readiness.
