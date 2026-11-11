# Novelty, Scientific Rigor, and Project Contributions

An honest, uninflated evaluation answering the academic question: **"What is new here? Does it already exist?"**

---

## 1. What Already Exists (Established Work)

To maintain scientific integrity, we explicitly acknowledge that several baseline components of this project are standard and well-established:
- **Corpus**: The Apache SpamAssassin Public Mail Corpus (collected around 2002–2003) has been widely used in benchmark literature for two decades.
- **Feature Pipeline**: Character/word n-gram TF-IDF representations paired with classical linear classifiers (Logistic Regression, Linear Support Vector Machines), probabilistic baselines (Multinomial and Complement Naive Bayes), and tree ensembles (Random Forest) are foundational, well-documented techniques.
- **Explainability Frameworks**: Local Interpretable Model-agnostic Explanations (LIME) and SHAP are existing, published attribution algorithms (Ribeiro et al., 2016; Lundberg & Lee, 2017).

---

## 2. What Is New Here (Project Contributions)

While the underlying algorithms exist, an initial naive implementation (such as in our own first run) often encounters common pitfalls: naive random train/test splits, default smoothing parameters, unablated preprocessing, and qualitative "visual inspections" of explanations without quantitative evaluation. 

This project contributes five specific axes of methodological and empirical rigor:

### A. Near-Duplicate-Aware Grouped Evaluation
- **Our Initial First-Run Pitfall**: In our first run, naive random splitting or standard cross-validation allowed slight textual variations of identical campaign templates to leak across train and test partitions, artificially inflating holdout scores.
- **Our Contribution**: We constructed an automated character-level TF-IDF similarity screen (4–5 character n-grams at cosine similarity $\ge 0.95$) that groups template variations into connected components. Evaluation uses `StratifiedGroupKFold`, mathematically guaranteeing that no near-duplicate cluster spans across train and holdout partitions or cross-validation folds.

### B. Fair Tuning and Elimination of Artificial Bias
- **Our Initial First-Run Pitfall**: In our own initial baseline run, Naive Bayes was evaluated with scikit-learn's default smoothing ($\alpha=1.0$), causing it to underperform compared to tuned linear models.
- **Our Contribution**: Controlled grid search on training data only revealed that widening the smoothing grid to $\alpha=0.01$ raises Multinomial Naive Bayes spam F1 from 0.64 to 0.98, demonstrating that the apparent weakness of Naive Bayes observed in our own initial run was an artifact of a restricted search range rather than algorithmic failure.
- **Complement NB Threshold Effect**: In binary classification without length normalization (`norm=False`), Complement NB and Multinomial NB learn **identical feature weights** ($w_j^{\text{CNB}} \equiv w_j^{\text{MNB}}$); they differ solely in the decision intercept (the prior log-odds: $b = 0$ for CNB vs $b = \log(N_{\text{spam}}/N_{\text{ham}}) \approx -1.68$ for MNB). Consequently, Complement NB's holdout advantage (0.9858 vs 0.9761 F1, catching 2 additional spam emails) is a **decision threshold effect**, not better feature modelling.

### C. Controlled HTML-Fallback Ablation
- **Our Initial First-Run Pitfall**: In our initial pipeline, email parsing either discarded HTML parts or stripped markup without measuring the differential impact across model families.
- **Our Contribution**: A paired ablation across identical splits evaluated models trained on `plain text only` versus `plain text + HTML fallback`. In this run, the effect differed by model; markup noise is a possible but untested explanation: fallback raised Naive Bayes spam recall dramatically (from 0.7850 to 0.9533) by recovering key vocabulary, but slightly lowered Logistic Regression recall (0.9159 to 0.9065).

### D. Quantitative Faithfulness Evaluation Across Model Families
- **Our Initial First-Run Pitfall**: In our initial exploration, sample attribution bar charts were inspected qualitatively without verifying whether the attributed tokens truly drive the classifier's decisions under quantitative perturbation.
- **Our Contribution**: We implemented a formal, quantitative **token deletion test** on holdout spam emails correctly classified by all five models ($n=87$). We note that restricting evaluation to mutual successes controls the comparison set across models, but introduces an easy-case benchmark bias. We measured the drop in both predicted spam probability ($\Delta P$) and log-odds / decision score ($\Delta z$) when removing top-$k$ tokens ($k \in \{1, 3, 5, 10\}$), compared against an empirical baseline of deleting $k$ random recognized tokens (20 independent repeats per email, with 95% bootstrap confidence intervals).
- **Exact Completeness**: All exact explainers satisfy completeness: `intercept + sum(tokens) == model_score` to within $10^{-13}$ numerical precision (verified by automated tests in `tests/test_explain.py`).
- **Correction of Deletion Matching**: Token deletion matching in raw text was corrected after the initial run to account for non-word punctuation separators (such as dots in URLs) and intermediate stop words within bigrams.
- **Critical Methodological Nuance**: For linear models (Logistic Regression, Linear SVM), the deletion test is partly circular because exact token weights $w_j \cdot x_j$ are closed-form components of the decision score. Therefore, the truly informative comparisons are **LIME vs exact closed-form linear attributions** and **Random Forest (nonlinear tree ensemble) vs the linear models**.
- **TF-IDF L2 Rescaling Diagnosis**: In raw text deletion, removing a token reduces the document vector norm $\|x\|_2$, which mathematically rescales all remaining active tokens upward by $1 / \sqrt{1 - x_{\text{deleted}}^2}$. For messages with dozens of remaining spam features, this upward rescaling is a likely diagnosis for why 3 of 87 emails at $k=1$ did not exceed random deletion, though this is framed as a diagnosis rather than a definitively proven causal result.

### E. Cross-Method Agreement and Objective Discrepancy
- **Our Initial First-Run Pitfall**: In our early design, we initially assumed different explainer methods would produce identical token rankings across model families.
- **Our Contribution**: We evaluated rank correlation (Spearman $\rho$) and Top-$k$ Jaccard overlap on shared active unigram tokens across $n=20–30$ emails (sample size selected due to computational simulation cost of repeated LIME perturbations):
  - For linear models, LIME demonstrates moderate-to-strong rank correlation ($\rho \approx 0.82–0.86$), confirming its utility as a local linear surrogate, though top-$k$ set overlap ($J \approx 0.22–0.28$) remains moderate. Hypothesis: this discrepancy arises because LIME optimizes an $L_2$-regularized objective predicting probability shifts under random masks, whereas exact attribution evaluates unregularized feature weights in logit space.
  - For Naive Bayes, rank correlation with LIME is markedly lower ($\rho \approx 0.43$). Hypotheses: (1) Naive Bayes is strictly additive in log-space ($\text{logit} = b + \sum x_j w_j$), but extreme probability saturation near 0 and 1 creates a flat, step-like probability surface that introduces severe noise into LIME's probability-difference surrogate; and (2) objective discrepancy between probability space and log-odds.
  - Hypothesis on Random Forest divergence: Random Forest relies on axis-aligned feature splits and decision paths that can capture nonlinear feature interactions. When explained via LIME, its feature attributions moderately correlate with linear models ($\rho \approx 0.46–0.50$) but diverge from Complement NB ($\rho \approx -0.04$).

---

## 3. Dataset Artifacts and Source-Specific Shortcuts

Global feature attribution revealed critical dataset artifacts:
- **Era and Mailing List Leakage**: The top ham-indicating terms across all five models include `rpm`, `wrote`, `users`, `spambayes`, `razor`, `freshrpms`, and `2002`. These terms reflect the specific technical mailing lists (Red Hat Linux packaging, SpamAssassin/Razor open-source discussions) from which the legitimate email was collected. Top spam terms include `insuranceiq` and `insuranceiq com`, pointing to a single prolific 2002 commercial sender.
- **Shortcut Learning**: Classifiers achieve $>0.98$ F1 partly by recognizing technical mailing-list jargon rather than universal linguistic properties of legitimate communication. Global explanations directly expose this historical era-coupling, providing a valuable cautionary lesson for deployed spam filtering.
- **Small Error Sample Caveat**: Because holdout accuracy is exceptionally high (Complement NB: 3 false negatives, 0 false positives; Linear SVM: 4 false negatives; Logistic Regression: 10 false negatives, 1 false positive), token patterns extracted from individual errors are anecdotal and illustrative rather than statistically generalizable findings.

---

## 4. Limitations

1. **Historical Corpus**: The SpamAssassin dataset dates from 2002–2003. Modern threats (polymorphic phishing, QR-code phishing, zero-day links, spear-phishing) are not represented.
2. **Easy Ham Benchmark**: The `easy_ham` subset contains clean, non-adversarial legitimate messages. Models were not evaluated against `hard_ham` (which contains technical discussion with high keyword overlap with spam).
3. **Small Holdout Spam Size**: The test set contains 107 spam messages (of which 87 are mutually correctly predicted by all five models). While adequate for statistical testing with bootstrap confidence intervals, larger contemporary corpora are needed for production certification.
4. **Edge-of-Grid Parameter Selections**: In cross-validation, the winning parameters for Naive Bayes ($\alpha=0.01$) and linear models ($C=10.0$) fell at the edge of the requested search grid.

---

## 5. Verified Related Work

1. **SpamAssassin Public Corpus**:
   - Apache Software Foundation. (2002–2006). *SpamAssassin Public Mail Corpus*. Available at: https://spamassassin.apache.org/old/publiccorpus/
2. **Naive Bayes in Spam Filtering**:
   - Sahami, M., Dumais, S., Heckerman, D., & Horvitz, E. (1998). "A Bayesian approach to filtering junk E-mail". In *Learning for Text Categorization: Papers from the 1998 Workshop* (Vol. 62, pp. 98–105).
   - Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003). "Tackling the poor assumptions of Naive Bayes text classifiers". In *Proceedings of the 20th International Conference on Machine Learning (ICML 2003)* (pp. 616–623). *(Introduced Complement Naive Bayes to handle text class imbalance).*
3. **Support Vector Machines for Text**:
   - Joachims, T. (1998). "Text categorization with Support Vector Machines: Learning with many relevant features". In *European Conference on Machine Learning (ECML 1998)* (pp. 137–142). Springer.
4. **Local Interpretable Model-agnostic Explanations (LIME)**:
   - Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). ""Why should I trust you?": Explaining the predictions of any classifier". In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1135–1144).
5. **Unified Feature Attribution (SHAP)**:
   - Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions". In *Advances in Neural Information Processing Systems (NeurIPS 2017)* (pp. 4765–4774).
6. **Perturbation-based evaluation of explanations (originally images)**:
   - Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K. R. (2017). "Evaluating the visualization of what a deep neural network has learned". *IEEE Transactions on Neural Networks and Learning Systems*, 28(11), 2660–2673.
7. **Faithfulness and Token Erasure in NLP**:
   - DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2020). "ERASER: A Benchmark to Evaluate Rationalized NLP Models". In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)* (pp. 4443–4458).
