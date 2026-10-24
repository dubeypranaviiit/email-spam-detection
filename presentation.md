# Suggested 9-slide presentation

1. **Title** — PS: 6.6, Email Spam Detection Using Machine Learning: A Comparative Study. Student, guide, department.
2. **Problem and objective** — Explain spam vs ham; compare Multinomial Naive Bayes, Complement Naive Bayes, Logistic Regression, Linear SVM and Random Forest on email.
3. **Dataset** — Apache SpamAssassin Public Mail Corpus, easy_ham and spam groups; 2,955 usable messages (2,473 ham, 482 spam); 210 parsed messages used HTML fallback. Note age and imbalance.
4. **Preparation and leakage prevention** — MIME subject/body parsing with HTML fallback, malformed/empty handling, exact deduplication, folder labels, and similarity-grouped stratified split before vectorizer fitting.
5. **TF-IDF features** — Show raw text → numeric TF-IDF unigrams/bigrams → model. Fit vectorizer on training messages only.
6. **Algorithms and tuning** — NB estimates class evidence; Logistic Regression and SVM use linear boundaries; RF uses trees. Five-fold grouped CV tunes on training data; the C and alpha grids are shown.
7. **Observed results** — 2,392 train / 563 test. F1: Multinomial NB 0.98, Complement NB 0.99, Logistic Regression 0.95, Linear SVM 0.98, Random Forest 0.92. Linear SVM had top CV spam F1 0.9469 ± 0.0200. Include `reports/model_comparison.png`.
8. **HTML ablation and demo** — NB recall was 0.95 with HTML fallback vs 0.79 plain-only; LR was 0.91 vs 0.92. The hypothesis is model-dependent. Launch Streamlit and classify examples.
9. **Conclusion and limitations** — Complement NB had top holdout F1 (0.99). Multinomial NB had better accuracy (0.99 vs 0.98), recall (0.95 vs 0.91), and F1 (0.98 vs 0.95) than Logistic Regression; NB precision was 1.00 vs LR 0.99. Top CV and holdout models differed. Zero near-duplicate crossing is guaranteed by grouping.

## Presenter notes

Say “spam” means the corpus folder label, not an independently verified modern safety judgement. Explain that group-aware 5-fold CV tunes models on training data only, then a separate holdout is evaluated. Multinomial NB had accuracy 0.99, precision 1.00, recall 0.95 and F1 0.98; Logistic Regression had 0.98, 0.99, 0.91 and 0.95. Complement NB had the highest holdout F1 (0.99), while SVM had top CV mean F1 (0.9469 ± 0.0200). HTML fallback improved NB recall but did not improve LR recall. Predictions are not guarantees.
