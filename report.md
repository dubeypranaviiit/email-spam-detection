# Project Report: Email Spam Detection Using Machine Learning

## Abstract

This project builds a supervised spam classifier and compares Multinomial Naive Bayes, Logistic Regression, Linear SVM and Random Forest. It uses the Apache SpamAssassin Public Mail Corpus, MIME-aware subject/body extraction, TF-IDF, five-fold grouped cross-validation and a separate group-aware holdout. The final run contains 2,955 unique messages. Logistic Regression and Linear SVM each reached spam F1 0.94 on the holdout, compared with 0.64 for Naive Bayes. HTML fallback improved Naive Bayes recall in a controlled ablation but did not improve Logistic Regression recall. Results are specific to this historic corpus and split.

## 1. Problem and objectives

Spam wastes attention and may carry unwanted content. This project classifies email messages as Spam or Ham. The objectives are to prepare labeled email data reproducibly, avoid duplicate leakage, compare classical classifiers fairly, evaluate precision/recall/F1 and error types, and demonstrate a trained model through Streamlit.

## 2. Dataset and preparation

Source: Apache [SpamAssassin Public Mail Corpus](https://spamassassin.apache.org/old/publiccorpus/readme.html), using `20030228_easy_ham.tar.bz2` and `20030228_spam.tar.bz2`. These are email messages, not SMS. Labels come from archive folders. Python's email parser reads MIME parts. The preparation script prefers text/plain and falls back to visible text from text/html using BeautifulSoup when plain text is unavailable; it skips attachments and script/style content. Subject and extracted body are combined. Empty or malformed records are skipped. Exact normalized-content duplicates are removed; ambiguous cross-label exact duplicates would be excluded.

| Class | Meaning | Usable count |
|---|---|---:|
| Ham | Legitimate email | 2,473 |
| Spam | Spam email | 482 |
| Total | After exact deduplication | 2,955 |

Body-source audit: 2,788 records used text/plain, 210 used the HTML fallback, and 4 had no text body (a subject could still be present). The source corpus is old, curated and imbalanced; `easy_ham` represents easier legitimate email and does not reflect current inboxes.

## 3. Method

Subject and body become one text sample. The preparation script computes exact hashes and a character-TF-IDF similarity screen (4–5 character n-grams; cosine at least 0.95). Similar pairs are connected into groups. A reproducible five-fold `StratifiedGroupKFold` creates the final approximately 20% holdout and the training validation folds. A similarity group never crosses the holdout or a CV fold. The final split contains 2,392 training and 563 test messages (456 ham and 107 spam). The computed exact-hash overlap and near-duplicate pair crossing the holdout are both zero.

Every classifier is inside a scikit-learn Pipeline with the same word TF-IDF setup: lowercase, accent stripping, English stop-word filtering, unigrams and bigrams, `min_df=2`, `max_df=0.98`, and sublinear TF. The vectorizer is refit within each training fold. GridSearchCV uses five grouped, stratified folds on the training partition only and chooses parameters by spam F1. The test partition is not used to tune models.

### Algorithms in simple terms

**TF-IDF:** converts words into numeric weights. A word gets more influence when it is useful to a message but less common across the training collection. Classifiers need numbers rather than raw strings.

**Multinomial Naive Bayes:** combines word evidence for each class with a simplifying conditional-independence assumption. CV searches `alpha` smoothing values.

**Logistic Regression:** learns weights for word features and combines them into a linear decision score. CV searches `C` regularization and compares no class weighting with `class_weight="balanced"`.

**Linear SVM:** finds a margin separating spam from ham in the TF-IDF feature space. CV searches `C` and class weighting.

**Random Forest:** combines decision trees to make a nonlinear classifier. It is included as a tree-based comparison; sparse text is often better suited to linear models.

Class weighting raises the training penalty for errors on the minority spam class. It was compared inside CV and never selected using the test set.

## 4. Results and analysis

The table reports observed held-out results. Precision, recall and F1 refer to the spam class.

| Model | Accuracy | Precision | Recall | F1 | Mean CV spam F1 |
|---|---:|---:|---:|---:|---:|
| Multinomial Naive Bayes | 0.8988 | 1.0000 | 0.4673 | 0.6369 | 0.6053 |
| Logistic Regression | 0.9787 | 0.9897 | 0.8972 | 0.9412 | 0.9399 |
| Linear SVM | 0.9787 | 0.9897 | 0.8972 | 0.9412 | 0.9422 |
| Random Forest | 0.9716 | 0.9691 | 0.8785 | 0.9216 | 0.9062 |

Selected CV parameters: Naive Bayes `alpha=0.5`; Logistic Regression `C=0.5`, balanced class weights; Linear SVM `C=0.5`, balanced class weights; Random Forest balanced class weights. The full grid is in `reports/cv_grid_results.csv`.

Logistic Regression beat Naive Bayes on all four held-out metrics: accuracy 0.98 vs 0.90, spam precision 0.99 vs 1.00, spam recall 0.90 vs 0.47, and spam F1 0.94 vs 0.64. SVM tied Logistic Regression on this holdout; SVM's mean CV F1 was slightly higher, so one split does not establish a dependable winner between the two. Random Forest also performed strongly. Confusion matrix rows are actual `[ham, spam]` and columns predicted `[ham, spam]`:

- Naive Bayes: `[[456, 0], [57, 50]]`
- Logistic Regression: `[[455, 1], [11, 96]]`
- Linear SVM: `[[455, 1], [11, 96]]`
- Random Forest: `[[453, 3], [13, 94]]`

### HTML extraction ablation

To test whether HTML-only messages explain low recall, both text representations were tuned using the same train-only grouped CV grid and evaluated on the same grouped holdout:

| Model | Body representation | Spam recall | Spam F1 |
|---|---|---:|---:|
| Naive Bayes | Plain text plus HTML fallback | 0.4673 | 0.6369 |
| Naive Bayes | Plain text only | 0.2243 | 0.3664 |
| Logistic Regression | Plain text plus HTML fallback | 0.8972 | 0.9412 |
| Logistic Regression | Plain text only | 0.9159 | 0.9561 |

The hypothesis is **partly supported**: HTML fallback more than doubled Naive Bayes recall, but Logistic Regression recall was slightly lower with HTML fallback on this split. HTML-only messages are an important preprocessing case, but HTML extraction alone does not explain low recall for every classifier. Results are available in `reports/html_ablation.csv`.

The near-duplicate screen found 1,436 candidate pairs at the chosen threshold. Connected groups were kept together across holdout and CV folds; no candidate pair crossed the holdout. This grouping heuristic may miss semantically related messages or over-group boilerplate, so independent time-based/current-corpus validation remains necessary.

## 5. Application

`app.py` loads the tuned Logistic Regression pipeline saved with joblib and accepts a subject and body. It rejects blank input, handles missing or unreadable models, and displays a class label only. No calibrated probability is shown; a prediction is not a guarantee of email safety.

## 6. Reproduction and artifacts

On Windows PowerShell, create a virtual environment, install `requirements.txt`, run `python src\prepare_data.py`, then `python src\train_evaluate.py`, followed by `streamlit run app.py`. For offline acquisition, put the official archives in `data/raw` and run preparation with `--local`.

Aggregate outputs include `reports/metrics.json`, `reports/model_comparison.csv`, `reports/cv_grid_results.csv`, `reports/html_ablation.csv`, and the comparison/confusion-matrix PNGs. Misclassified message CSVs are generated locally for manual inspection but excluded from Git because they contain email text. Trained joblib models and raw/prepared datasets are also excluded from Git.

## 7. Conclusion and future work

On this group-aware holdout, Logistic Regression beat Naive Bayes on accuracy, spam precision, recall and F1; its spam F1 was 0.94 compared with Naive Bayes at 0.64. Linear SVM matched Logistic Regression on the test metrics, while having a slightly higher mean CV F1. Balanced class weights and grouped validation address the earlier recall and duplicate-leakage concerns. HTML fallback materially improved Naive Bayes recall but did not improve Logistic Regression recall in this run. The historical corpus and single holdout limit generalization. Next steps are testing on an independent newer email dataset, time-based validation, threshold/calibration analysis on validation data, and stronger MIME/header features. A large transformer model is outside the scope of this small corpus and beginner-friendly classical comparison.
