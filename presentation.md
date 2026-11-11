# Presentation Deck: Explainable Email Spam Detection with Quantitative Faithfulness Benchmarks

A 12-slide comprehensive presentation covering the dataset audit, leakage-free modeling, exact XAI token attributions, empirical faithfulness deletion tests, and interactive dashboard.

---

### Slide 1: Title & Overview
- **Title**: Rigorous Email Spam Detection & Explainability: A Quantitative Faithfulness Benchmark
- **Context**: Apache SpamAssassin Public Corpus (2,955 emails; 2,473 ham, 482 spam).
- **Core Contribution**: Near-duplicate-aware grouped evaluation, exact closed-form linear & log-odds token attributions, quantitative token deletion tests, and real-time interactive XAI dashboard.

---

### Slide 2: Problem Statement & The Naive Split Trap
- **The Core Flaw in Literature**: Standard spam benchmarks apply naive random train/test splits. Spammers reuse templates (varying only recipient name or tracking link). Naive splits scatter near-identical templates into both train and test sets, artificially inflating accuracy (>99.9%).
- **Our Methodology**: Automated character 4–5 gram TF-IDF cosine similarity screen (&ge; 0.95) mapped into connected graph components. Entire campaign clusters are assigned exclusively to train OR test via `StratifiedGroupKFold(5, seed=42)`. Zero template leakage guaranteed.

---

### Slide 3: Feature Engineering & Model Architectures
- **TF-IDF Representation**: Word unigrams and bigrams (`ngram_range=(1, 2)`), sublinear term frequency scaling, `min_df=2`, `max_df=0.98`.
- **5 Diverse Model Families**:
  1. *Multinomial Naive Bayes* (Generative count-based prior + likelihood)
  2. *Complement Naive Bayes* (Tackles text class imbalance via complement classes)
  3. *Logistic Regression* (L2-regularized linear log-odds)
  4. *Linear SVM* (Maximum margin separation hyperplane)
  5. *Random Forest* (Non-linear decision tree ensemble)

---

### Slide 4: Fair Tuning & The Complement NB Threshold Proof
- **GridSearchCV on Training Set Only**: Tuning across &alpha; &in; [0.01, 1.0] and C &in; [0.1, 10.0] revealed that widening &alpha; to 0.01 raises MNB Spam F1 from 0.64 to 0.98, proving prior "weaknesses" of Naive Bayes were tuning artifacts.
- **Mathematical Derivation**: In binary classification with `norm=False`, Complement NB and Multinomial NB have **identical feature weights** ($w_j^{\text{CNB}} \equiv w_j^{\text{MNB}}$). They differ solely in the decision intercept ($b=0$ for CNB vs prior log-odds for MNB). CNB's holdout edge is strictly a decision threshold calibration advantage, not superior feature modeling.

---

### Slide 5: Holdout Performance & Benchmark Results
- **Held-Out Evaluation (563 test emails: 456 ham, 107 spam)**:
  - **Complement NB**: Accuracy 99.47%, Precision 100.0%, Recall 97.20%, **Holdout F1: 98.88%** (Top Model)
  - **Linear SVM**: Accuracy 99.29%, Precision 100.0%, Recall 96.26%, **Holdout F1: 98.10%** (Top CV Model)
  - **Multinomial NB**: Accuracy 99.11%, Precision 100.0%, Recall 95.33%, **Holdout F1: 97.61%**
  - **Logistic Regression**: Accuracy 98.05%, Precision 98.98%, Recall 90.65%, **Holdout F1: 94.63%**
  - **Random Forest**: Accuracy 97.16%, Precision 96.91%, Recall 87.85%, **Holdout F1: 92.16%**

---

### Slide 6: HTML-Fallback Text Extraction Ablation
- **Research Question**: Does parsing HTML when plain-text is missing help or harm classification?
- **Finding**: The hypothesis is **model-dependent**:
  - For Naive Bayes, HTML fallback raised recall dramatically from 78.5% to 95.3% (+16.8% recall boost) by recovering 138 otherwise empty emails.
  - For Logistic Regression, HTML fallback slightly lowered recall (91.6% to 90.7%) due to noisy HTML markup artifacts.

---

### Slide 7: Explainable AI: Exact Faithful Attributions
- **Exact Linear Decomposition**: For Logistic Regression and Linear SVM, token contribution is exactly $w_j \cdot x_j = \text{coef}_j \times \text{tfidf}_j$.
- **Exact Naive Bayes Decomposition**: For MNB and CNB, token contribution is $x_j \cdot (\log P(w_j|\text{spam}) - \log P(w_j|\text{ham}))$.
- **Completeness Guaranteed**: Automated test suite verifies that `intercept + sum(token_contributions) == model_score` to within machine precision ($< 1.35 \times 10^{-13}$).

---

### Slide 8: Quantitative Faithfulness Evaluation (Deletion Tests)
- **Beyond Qualitative Inspection**: We removed top-$k$ explained tokens ($k \in \{1, 3, 5, 10\}$) from actual raw text and re-evaluated the full vectorizer + classifier against 20 random token baseline repetitions on mutual test spam ($n=87$).
- **Results**:
  - Deleting top-1 token dropped spam score by an average of 1.48 to 2.45 points (96.6% to 100% win rate over random deletion).
  - Circulating Nuance: Acknowledged that deletion testing on linear models is partly circular due to closed-form weights.
  - L2 Rescaling Diagnosis: Diagnosed that deleting a high-TF-IDF token shrinks $\|x\|_2$, rescaling remaining spam tokens by $1/\sqrt{1 - x_1^2}$.

---

### Slide 9: Cross-Method Agreement: Exact vs LIME
- **Linear Models**: Strong rank correlation between Exact and LIME ($\rho \approx 0.880$, Top-5 Jaccard 0.589), confirming LIME recovers first-order linear decision boundaries.
- **Naive Bayes**: Discrepancy ($\rho \approx 0.165$, Top-5 Jaccard 0.142) explained by two hypotheses:
  1. Extreme probability saturation in log space creates a flat probability difference surface.
  2. Target mismatch: LIME fits in probability space while exact NB is additive in log-odds.
- **Random Forest**: Moderate agreement with linear models ($\rho \approx 0.45$), reflecting non-linear tree split interactions.

---

### Slide 10: Dataset Artifacts & Shortcut Learning Discovery
- **Global Explanations Expose Historical Era Artifacts**:
  - Top ham terms: `rpm`, `wrote`, `users`, `spambayes`, `razor`, `freshrpms`, `2002`.
  - Top spam terms: `insuranceiq`, `click`, `free`, `money`.
- **Takeaway**: Classifiers achieve >99% accuracy partly by memorizing 2002 open-source mailing list jargon rather than universal linguistic properties of legitimate communication. Global explanations make this shortcut learning visible.

---

### Slide 11: Real-Time Interactive Next.js Dashboard Architecture
- **Tech Stack**: Next.js 14 (App Router), TypeScript, Tailwind CSS, Lucide icons, Recharts, FastAPI backend, Docker.
- **Key Features**:
  1. *Overview*: Headline metrics, multi-metric comparison bar charts, interactive confusion matrix viewer.
  2. *Live Detector*: Real-time classification, decision margin gauge, unigram/bigram token highlighting (intensity = weight), color-blind accessible underlines and sign markers `(+)` / `(-)`.
  3. *Explainability*: Global top-25 terms, deletion curves with 95% bootstrap CI bands, method agreement tables.
  4. *Error Analysis & Methodology*: Aggregated FP/FN diagnostic patterns, CV grid tables, HTML ablation results.
  5. *About*: Transparent novelty disclosure rendering `NOVELTY.md`.

---

### Slide 12: Academic Limitations & Future Work
1. **Corpus Era**: SpamAssassin dates to 2002–2003; modern polymorphic/adversarial spam requires retraining on contemporary streams.
2. **Small Holdout Errors**: With only 1–4 errors per model, individual misclassified token weights are qualitative and illustrative.
3. **Future Directions**: Transformer-based attention rollouts (RoBERTa / DeBERTa) benchmarked against exact linear attributions.

---

### Presenter Tips for the Viva
- Emphasize the **StratifiedGroupKFold** leakage prevention when asked about preprocessing.
- If asked why Complement NB won holdout F1, explain the **threshold effect proof** ($w_j^{\text{CNB}} \equiv w_j^{\text{MNB}}$).
- If asked whether the deletion test is circular for linear models, affirm immediately: *"Yes, for linear models the exact weights are closed-form components of the score; the truly independent comparisons are LIME vs exact and Random Forest vs linear models."*
