# Email Spam Detection with Explainable AI & Quantitative Faithfulness Benchmarks

An audited, explainable machine-learning system comparing **Multinomial Naive Bayes**, **Complement Naive Bayes**, **Logistic Regression**, **Linear SVM**, and **Random Forest** on the SpamAssassin public mail corpus.

Features **exact closed-form token attributions**, **quantitative token deletion faithfulness tests**, **MIME HTML fallback ablation**, a **FastAPI backend**, and an interactive **Next.js 14 App Router dashboard**.

---

## Key Project Additions & Research Contributions

1. **Near-Duplicate Grouped Evaluation (Zero-Leakage Guarantee)**:
   - Automated character 4–5 gram TF-IDF similarity screen (&ge; 0.95) mapped into connected graph components.
   - Evaluated via `StratifiedGroupKFold(5, seed=42)` ensuring no spam campaign template leaks across training and test partitions.
2. **Exact Closed-Form Token Attributions**:
   - For Logistic Regression and Linear SVM: exact linear contributions ($w_j \cdot x_j = \text{coef}_j \times \text{tfidf}_j$).
   - For Naive Bayes / Complement NB: exact token log-odds evidence contributions.
   - Completeness mathematically verified to within $< 1.35 \times 10^{-13}$ numerical precision.
3. **Quantitative Faithfulness Evaluation (Deletion Tests)**:
   - Removes top-$k$ explained tokens ($k \in \{1, 3, 5, 10\}$) from actual raw text and re-evaluates the pipeline against 20 random baseline repetitions with 95% bootstrap confidence intervals ($n=87$ mutual holdout spam).
   - Diagnosed the upward TF-IDF $L_2$ norm rescaling effect upon token deletion ($1 / \sqrt{1 - x_1^2}$).
4. **Complement NB Threshold Effect Proof**:
   - Proved that in binary classification with unnormalized weights, Complement NB and Multinomial NB have **identical feature weights** ($w_j^{\text{CNB}} \equiv w_j^{\text{MNB}}$). CNB's holdout edge is strictly a decision threshold calibration advantage ($b = 0$), not better feature modeling.
5. **Corpus Era Artifact Discovery & Shortcut Learning**:
   - Global explanations exposed historical mailing list shortcuts (`rpm`, `wrote`, `users`, `spambayes`, `razor`, `2002`), demonstrating that models achieve 97% to 99.5% holdout accuracy partly by memorizing 2002 open-source mailing list jargon.
6. **Production FastAPI Backend & Next.js 14 Dashboard**:
   - Real-time classification and token highlighting (intensity = weight, color-blind accessible with underlines and `(+)`/`(-)` markers).
   - Strict privacy: 20k character limit, sliding-window rate limiting, and zero email text logging.

---

## Results Summary (Holdout Split: 563 Test Messages)

| Classifier | Holdout Accuracy | Spam Precision | Spam Recall | Holdout Spam F1 | CV Best Spam F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Complement Naive Bayes** | **99.47%** | **1.0000** | **97.20%** | **0.9858** | 0.9444 &plusmn; 0.0099 |
| **Linear SVM** | 99.29% | 1.0000 | 96.26% | 0.9810 | **0.9469 &plusmn; 0.0200** |
| **Multinomial Naive Bayes** | 99.11% | 1.0000 | 95.33% | 0.9761 | 0.9277 &plusmn; 0.0112 |
| **Logistic Regression** | 98.05% | 0.9898 | 90.65% | 0.9463 | 0.9452 &plusmn; 0.0180 |
| **Random Forest** | 97.16% | 0.9691 | 87.85% | 0.9216 | 0.9062 &plusmn; 0.0195 |

*Complete metrics, grid search tables, and confusion matrices are preserved in `reports/metrics.json`.*

---

## Quickstart & Running the Application

### Option A: VS Code (Recommended)
1. Open this folder in VS Code.
2. **Terminal 1 (Backend API)**:
   ```powershell
   uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
   ```
3. **Terminal 2 (Next.js Dashboard)**:
   ```powershell
   cd web
   npm run dev
   ```
4. Open **[http://localhost:3000](http://localhost:3000)** in your browser!

*(Alternatively, press `Ctrl+Shift+P` &rarr; `Tasks: Run Task` &rarr; select `1. Start Backend API` and `2. Start Next.js Dashboard`)*.

---

### Option B: Docker Compose (Full Stack)
Build and launch both the FastAPI backend and Next.js frontend in isolated containers:
```bash
docker-compose up --build
```
- Frontend: `http://localhost:3000`
- Backend API Docs: `http://localhost:8000/docs`

---

### Option C: Python Scripts & CLI Reproduction
- **Run the full end-to-end pipeline**:
  ```powershell
  python scripts/build_all.py
  ```
- **Run all automated tests (17 passed)**:
  ```powershell
  pytest
  ```
- **Run the original Streamlit demo**:
  ```powershell
  streamlit run app.py
  ```

---

## Dashboard Pages Overview

The Next.js dashboard (`web/`) contains 6 dedicated pages:
1. **Overview (`/`)**: Headline metrics, holdout comparison bar charts, and interactive confusion matrix viewer.
2. **Live Detector (`/detector`)**: Interactive email input with sample loaders, model/explainer selector, decision margin gauge, unigram & bigram token highlight intensity view, and feature contribution charts.
3. **Explainability (`/explainability`)**: Global top 25 terms per model, faithfulness deletion curves ($\Delta P$ and $\Delta z$ vs random baseline with 95% bootstrap CI bands), and method-agreement metrics.
4. **Error Analysis (`/errors`)**: Aggregated diagnostic patterns of False Positives and False Negatives, MIME source distribution, character length distributions, and misleading token frequencies (zero email bodies stored).
5. **Methodology (`/methodology`)**: Pipeline architecture flow diagram, grouped near-duplicate split rationale, CV grid results highlighting edge-of-grid parameters honestly, and HTML fallback ablation results.
6. **About / Novelty (`/about`)**: Full rendering of `NOVELTY.md` answering *"What is new here? Does it already exist?"* with academic citations.

---

## Project Structure

```
Minor_project/
├── .vscode/                     # VS Code tasks and debug configurations
├── api/                         # FastAPI backend service
│   ├── main.py                  # API endpoints, rate limiting, and timeout middleware
│   └── schemas.py               # Pydantic request & response schemas
├── data/                        # Raw and processed email data (gitignored)
├── models/                      # Saved scikit-learn pipelines (gitignored)
├── reports/                     # Numerical benchmarks and plots
│   ├── metrics.json             # Pinned benchmark metrics
│   ├── cv_grid_results.csv      # Cross-validation hyperparameter results
│   ├── html_ablation.csv        # HTML fallback text extraction ablation
│   └── explain/                 # Explainability & faithfulness artifacts
│       ├── faithfulness.json    # Deletion test results & method agreement
│       ├── faithfulness.csv     # Deletion statistics table
│       └── error_analysis_summary.json # Aggregated FP/FN patterns
├── scripts/
│   └── build_all.py             # CLI pipeline build script
├── src/
│   ├── explain/                 # Explainable AI & faithfulness modules
│   │   ├── local.py             # Exact linear/NB attributions and LIME
│   │   ├── global_terms.py      # Top spam & ham features per model
│   │   ├── faithfulness.py      # Token deletion test & method agreement
│   │   └── error_analysis.py    # Aggregated error pattern extraction
│   ├── prepare_data.py          # MIME parsing, HTML fallback, and grouping
│   └── train_evaluate.py        # Grouped split, GridSearchCV, and evaluation
├── tests/                       # Automated test suite
│   ├── test_api.py              # FastAPI endpoint integration tests
│   └── test_explain.py          # Completeness, faithfulness, and hash integrity tests
├── web/                         # Next.js 14 App Router frontend
│   ├── app/                     # App Router pages (/, /detector, /explainability, /errors, /methodology, /about)
│   ├── components/              # Shared components (Navbar, UI controls)
│   └── lib/                     # Typed API client and utilities
├── Dockerfile                   # Backend Dockerfile
├── docker-compose.yml           # Unified multi-container orchestration
├── Makefile                     # Common automation shortcuts
├── NOVELTY.md                   # Formal academic novelty statement
├── presentation.md              # 12-slide comprehensive presentation deck
├── viva_questions.md            # 35 viva questions with verified answers
└── requirements.txt             # Pinned Python dependencies
```

---

## Verification & Integrity Guarantees

- **Unmodified Benchmark Numbers**: `reports/metrics.json` is protected by SHA256 integrity checks verified by `tests/test_explain.py::test_metrics_json_integrity_unmodified`.
- **Completeness Guarantee**: Reconstructed model scores from exact token weights match actual model decision scores to within $< 1.35 \times 10^{-13}$.
- **Privacy Assurance**: No raw email text is stored in committed error reports or logged in API server outputs.
