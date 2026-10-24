# Suggested 9-slide presentation

1. **Title** — PS: 6.6, Email Spam Detection Using Machine Learning: A Comparative Study. Student, guide, department.
2. **Problem and objective** — Explain spam vs ham; compare Multinomial Naive Bayes, Logistic Regression, Linear SVM and Random Forest on email.
3. **Dataset** — Apache SpamAssassin Public Mail Corpus, easy_ham and spam groups; 2,955 usable messages (2,473 ham, 482 spam); 210 parsed messages used HTML fallback. Note age and imbalance.
4. **Preparation and leakage prevention** — MIME subject/body parsing with HTML fallback, malformed/empty handling, exact deduplication, folder labels, and similarity-grouped stratified split before vectorizer fitting.
5. **TF-IDF features** — Show raw text → numeric TF-IDF unigrams/bigrams → model. Fit vectorizer on training messages only.
6. **Algorithms and tuning** — NB word evidence; LR and SVM linear boundaries; RF trees. Five-fold grouped CV tunes on training data; balanced class weights are compared for LR/SVM/RF.
7. **Observed results** — 2,392 train / 563 test. NB F1 0.64; LR F1 0.94; Linear SVM F1 0.94; Random Forest F1 0.92. Include `reports/model_comparison.png` and confusion matrices.
8. **HTML ablation and demo** — NB recall was 0.47 with HTML fallback vs 0.22 plain-only; LR was 0.90 vs 0.92. Explain the hypothesis is partly supported. Launch Streamlit and classify examples.
9. **Conclusion and limitations** — LR beat NB on all four metrics, while SVM tied LR on the holdout. Similarity groups did not cross holdout/CV folds; still, the corpus is historic and limited. Future work: independent current email data and time-based evaluation.

## Presenter notes

Say “spam” means the corpus folder label, not an independently verified modern safety judgement. Explain that group-aware 5-fold CV tunes the models on training data only, then a separate holdout is evaluated. Logistic Regression's F1 was 0.94 vs Naive Bayes at 0.64; SVM tied LR on the test split. Explain that HTML fallback improved NB recall but did not improve LR recall. Predictions are not guarantees.
