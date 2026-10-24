# PS: 6.6 — Email Spam Detection Using Machine Learning

This beginner project compares **Multinomial Naive Bayes** and **Logistic Regression** for the supervised classification of real email messages into **Spam (1)** and **Ham (0)**. Both models use the same train/test split and TF-IDF features. The project does not claim a new algorithm; the educational contribution is a transparent, reproducible comparison and a small demonstration app.

## Dataset

We use the Apache [SpamAssassin Public Mail Corpus](https://spamassassin.apache.org/old/publiccorpus/readme.html), specifically its `easy_ham` (ham) and `spam` collections. Apache describes roughly 2,500 easy ham and 500 spam messages. The preparation script records the actual usable counts and labels after parsing and exact duplicate removal. This is an old corpus with substantial class imbalance and an intentionally easy ham subset, so scores should not be generalized to current inboxes. Dataset acquisition instructions are in [data/README.md](data/README.md).

## What each part does

| Component | Plain-English purpose | Why it is here / likely viva question |
|---|---|---|
| MIME email parsing | Extracts subject and readable plain-text body from message files. | Email files have headers, encodings and attachments. *Q: Why not read every file as plain text?* Because multipart and encoded messages would be mishandled. |
| Cleaning and labels | Skips empty/broken messages, joins subject/body, removes exact duplicate content, and assigns labels from corpus folders. | Prevents duplicate examples and label guessing. *Q: Could a duplicate affect scores?* Yes, if copies land in both train and test. |
| Stratified split | Reserves 20% of messages for a final evaluation, maintaining class proportions. | Estimates performance on unseen examples. *Q: Why stratify?* The dataset is imbalanced, so both sets should retain similar spam/ham proportions. |
| TF-IDF | Turns words and word pairs into numeric weights; frequent-in-one-email but less-common-in-corpus terms receive more weight. | Classifiers need numbers, not raw text. Fit occurs inside each pipeline using training messages only. *Q: What prevents vectorizer leakage?* `Pipeline.fit` sees only X_train. |
| Multinomial Naive Bayes | Scores each class using word evidence and a simplifying conditional-independence assumption. | Fast, strong text baseline. *Q: Why smoothing?* `alpha=1` avoids zero likelihood for unseen terms. |
| Logistic Regression | Learns a linear decision boundary over TF-IDF features. | Common discriminative text classifier. *Q: Why max_iter=1000?* To allow optimization to converge on sparse text features. |
| Evaluation | Computes accuracy, spam precision/recall/F1, reports, confusion matrices and error samples. | A single accuracy number can hide missed spam and false alarms. *Q: What is a false positive?* A ham message incorrectly marked spam. |
| Streamlit | Provides a local browser interface to classify pasted messages. | Makes the trained model easy to demonstrate without changing its training data. |

## Windows setup

Open PowerShell in this folder. Python 3.10–3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\prepare_data.py
python src\train_evaluate.py
streamlit run app.py
```

If PowerShell blocks environment activation, use `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\streamlit.exe` directly. If downloading is unavailable, manually download the two archives named in [data/README.md](data/README.md) into `data/raw`, then run `python src\prepare_data.py --local`.

## Results

**Observed results from the completed local run:** 2,955 unique usable messages (2,473 ham; 482 spam); 2,364 train and 591 test (495 ham; 96 spam in test). Multinomial Naive Bayes: accuracy 0.9239, spam precision 1.0000, spam recall 0.5313, spam F1 0.6939. Logistic Regression: accuracy 0.9475, spam precision 1.0000, spam recall 0.6771, spam F1 0.8075. Both matrices have zero false positives on this test split; NB had 45 false negatives and Logistic Regression had 31. Logistic Regression performed better on all four listed metrics in this one split. These are observed results, not a guarantee of future performance. See `reports/metrics.json` and `reports/model_comparison.csv`. The run uses seed 42, a stratified 80/20 split, unigram/bigram TF-IDF, English stop-word filtering, `min_df=2`, `max_df=0.98`, and `sublinear_tf=True`. NB uses `alpha=1.0`; logistic regression uses `C=1.0`, `liblinear`, and `max_iter=1000`. The test set is not used for tuning.

**Near-duplicate caveat:** the preparation audit found 1,440 character-TF-IDF similarity pairs at threshold 0.95; 541 cross the train/test boundary, including 9 with different labels. The heuristic is a screening device (shared template boilerplate can look similar), and these pairs were not removed. Treat this split's metrics as preliminary; use grouped/campaign-aware or time-based validation for a stronger generalization claim.

## Files

- `src/prepare_data.py`: download and parse data, remove exact duplicates, report class counts.
- `src/train_evaluate.py`: split, train, evaluate, save pipelines and artifacts.
- `app.py`: paste-email classification UI.
- `models/`: local joblib pipelines (generated, gitignored).
- `reports/`: local metrics, plots and error cases (generated, gitignored).
- `report.md`, `presentation.md`, `viva_questions.md`: project deliverables.

## Limitations and next steps

The source mail is old; labels and group choices are inherited from the corpus; easy ham is not representative of difficult modern cases; language and topic shift can break the learned vocabulary; exact deduplication cannot detect every campaign/template variant; TF-IDF ignores much email structure; scores depend on this one split; and the displayed class is not a calibrated probability or a guarantee of safety. Future work could add hard ham, newer email corpora, time-based validation, sender/header features, threshold analysis, robust near-duplicate clustering, calibration, and privacy/security review before real deployment.
