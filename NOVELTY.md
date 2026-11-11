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

While the underlying algorithms exist, standard academic implementations and student projects often apply naive train/test splits, arbitrary hyperparameter defaults, unablated preprocessing, and qualitative "visual inspections" of explanations without quantitative evaluation. 

This project contributes five specific axes of methodological and empirical rigor:

### A. Near-Duplicate-Aware Grouped Evaluation
- **Standard Pitfall**: Real-world spam campaigns frequently send slight textual variations of identical templates. In random splits or naive $k$-fold cross-validation, near-identical templates leak into both train and test splits, artificially inflating holdout scores.
- **Our Contribution**: We constructed an automated character-level TF-IDF similarity screen (4–5 character n-grams at cosine similarity $\ge 0.95$) that groups template variations into connected components. Evaluation uses `StratifiedGroupKFold`, mathematically guaranteeing that no near-duplicate cluster spans across train and holdout partitions or cross-validation folds.

### B. Fair Tuning and Elimination of Artificial Bias
- **Standard Pitfall**: Academic comparisons frequently present Naive Bayes with default smoothing ($\alpha=1.0$), causing it to underperform compared to tuned linear models.
- **Our Contribution**: Controlled grid search on training data only revealed that widening the smoothing grid to $\alpha=0.01$ raises Multinomial Naive Bayes spam F1 from 0.64 to 0.98, demonstrating that previously reported "weaknesses" in Naive Bayes on this dataset were artifacts of restricted search ranges rather than algorithmic failure.

### C. Controlled HTML-Fallback Ablation
- **Standard Pitfall**: Email parsers either discard HTML parts (losing spam payload) or blindly strip tags without measuring the differential impact across model families.
- **Our Contribution**: A paired ablation across identical splits evaluated models trained on `plain text only` versus `plain text + HTML fallback`. We proved that the HTML fallback hypothesis is **model-dependent**: fallback raised Naive Bayes spam recall dramatically (from 0.7850 to 0.9533) by recovering key vocabulary, but slightly lowered Logistic Regression recall (0.9159 to 0.9065) due to noisy markup terms.

### D. Quantitative Faithfulness Evaluation Across Model Families
- **Standard Pitfall**: Most XAI studies show sample attribution bar charts and declare the model "explainable" without verifying whether the attributed tokens truly drive the classifier's decisions.
- **Our Contribution**: We implemented a formal, quantitative **token deletion test** on holdout spam emails correctly classified by all five models ($n=87$). We measured the drop in both predicted spam probability ($\Delta P$) and log-odds / decision score ($\Delta z$) when removing top-$k$ tokens ($k \in \{1, 3, 5, 10\}$), compared against an empirical baseline of deleting $k$ random recognized tokens (20 independent repeats per email, with 95% bootstrap confidence intervals).
- **Critical Methodological Nuance**: For linear models (Logistic Regression, Linear SVM), the deletion test is partly circular because exact token weights $w_j \cdot x_j$ are closed-form components of the decision score. Therefore, the truly informative comparisons are **LIME vs exact closed-form linear attributions** and **Random Forest (nonlinear tree ensemble) vs the linear models**.

### E. Cross-Method Agreement and Objective Discrepancy
- **Standard Pitfall**: Assuming all explainer methods should produce identical rankings.
- **Our Contribution**: We evaluated rank correlation (Spearman $\rho$) and Top-$k$ Jaccard overlap on shared active unigram tokens:
  - For linear models, LIME demonstrates moderate-to-strong rank correlation ($\rho \approx 0.82–0.86$), confirming its utility as a local linear surrogate, though top-$k$ set overlap ($J \approx 0.22–0.28$) remains moderate due to LIME's $L_2$ regularization.
  - For Naive Bayes, rank correlation with LIME is markedly lower ($\rho \approx 0.43$). We formulated the hypothesis: Naive Bayes is strictly additive in log-space ($\text{logit} = b + \sum x_j w_j$), but extreme probability saturation near 0 and 1 creates a flat, step-like probability surface that introduces severe noise into LIME's probability-difference surrogate.

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
6. **Faithfulness and Token Perturbation Evaluation in NLP**:
   - DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2020). "ERASER: A Benchmark to Evaluate Rationalized NLP Models". In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)* (pp. 4443–4458).
   - Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K. R. (2016). "Evaluating the visualization of what a deep neural network has learned". *IEEE Transactions on Neural Networks and Learning Systems*, 28(11), 2660–2673.
