# Suggested 9-slide presentation

1. **Title** — PS: 6.6, Email Spam Detection Using Machine Learning: A Comparative Study. Student, guide, department.
2. **Problem and objective** — Explain spam vs ham; objective is a controlled comparison of Multinomial Naive Bayes and Logistic Regression on email.
3. **Dataset** — Apache SpamAssassin Public Mail Corpus, easy_ham and spam groups; 2,955 usable messages after exact deduplication (2,473 ham, 482 spam); note age and imbalance.
4. **Preparation and leakage prevention** — MIME subject/body extraction, malformed/empty handling, exact duplicate removal, folder labels, stratified 80/20 split before vectorizer fitting.
5. **TF-IDF features** — Show raw text → numeric TF-IDF unigrams/bigrams → model. Fit vectorizer on training messages only.
6. **Algorithms** — NB estimates class evidence from words (alpha=1); logistic regression learns linear feature weights (C=1, max_iter=1000). Same split/features.
7. **Observed results** — 2,364 train / 591 test. NB accuracy/precision/recall/F1 = 0.9239/1.0000/0.5313/0.6939; Logistic Regression = 0.9475/1.0000/0.6771/0.8075. Include generated charts.
8. **Error analysis and demo** — One anonymized false positive and false negative from generated CSV; launch Streamlit and paste a clearly labeled sample email.
9. **Conclusion, limitations, questions** — Logistic Regression led on this split; historical/easy corpus, imbalance, 541 near-duplicate candidates cross splits (9 differing labels), domain shift, uncalibrated labels; future work includes group/time split.

## Presenter notes

Say “spam” means the corpus folder label, not an independently verified modern safety judgement. In this split, both models had no false positives; NB missed 45 spam and Logistic Regression missed 31. Explain why these error types matter. Mention that exact duplicates are removed and the TF-IDF vocabulary sees training text only. Demonstrate with two examples but do not promise every message will be classified correctly.
