# PS: 6.6 — Email Spam Detection Using Machine Learning

This beginner project compares **Multinomial Naive Bayes**, **Logistic Regression**, **Linear SVM**, and **Random Forest** for classifying real email as **Spam (1)** or **Ham (0)**. They use the same TF-IDF features and group-aware holdout. Five-fold grouped CV tunes models on training data only. The project does not claim a new algorithm; its contribution is a transparent comparison and demonstration app.

## Dataset

We use the Apache [SpamAssassin Public Mail Corpus](https://spamassassin.apache.org/old/publiccorpus/readme.html), specifically its `easy_ham` (ham) and `spam` collections. Apache describes roughly 2,500 easy ham and 500 spam messages. The parser prefers `text/plain` and extracts visible text from `text/html` when no plain-text part exists. The run kept 2,955 unique messages: 2,473 ham and 482 spam. This is an old corpus with class imbalance and an intentionally easy ham subset, so scores should not be generalized to current inboxes. Dataset instructions are in [data/README.md](data/README.md).

## What each part does

| Component | Plain-English purpose | Why it is here / likely viva question |
|---|---|---|
| MIME email parsing | Extracts subject and readable body from message files, preferring plain text and converting HTML to visible text when plain text is absent. | Email files have headers, encodings and attachments. *Q: Why not read every file as plain text?* Because multipart and encoded messages would be mishandled. |
| Cleaning and labels | Skips empty/broken messages, joins subject/body, removes exact duplicate content, and assigns labels from corpus folders. | Prevents duplicate examples and label guessing. *Q: Could a duplicate affect scores?* Yes, if copies land in both train and test. |
| Grouped stratified split | Reserves about 20% for a final evaluation while keeping near-duplicate groups together. | Estimates unseen performance without placing similar templates in both sets. *Q: Why group?* Near copies can inflate scores if split across train and test. |
| TF-IDF | Turns words and word pairs into numeric weights; frequent-in-one-email but less-common-in-corpus terms receive more weight. | Classifiers need numbers, not raw text. Fit occurs inside each pipeline using training messages only. *Q: What prevents vectorizer leakage?* `Pipeline.fit` sees only X_train. |
| Multinomial Naive Bayes | Scores each class using word evidence and a simplifying conditional-independence assumption. | Fast, strong text baseline. *Q: Why smoothing?* The selected `alpha=0.5` avoids zero likelihood for unseen terms. |
| Logistic Regression and Linear SVM | Learn linear boundaries over TF-IDF features; CV compares regularization and class weighting to help minority spam recall. | Strong text classifiers. *Q: What does `class_weight="balanced"` do?* It gives more weight to errors on the smaller spam class. |
| Random Forest | Combines decision trees. | Adds a nonlinear tree-based comparison. |
| Evaluation | Uses grouped, stratified five-fold CV for training-only grid search, then reports accuracy, spam precision/recall/F1, confusion matrices and local error samples. | A single accuracy number can hide missed spam and false alarms. *Q: What is a false positive?* A ham message incorrectly marked spam. |
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

Observed run: 2,392 train and 563 test messages. Naive Bayes accuracy/precision/recall/F1 = 0.8988/1.0000/0.4673/0.6369; Logistic Regression = 0.9787/0.9897/0.8972/0.9412; Linear SVM = 0.9787/0.9897/0.8972/0.9412; Random Forest = 0.9716/0.9691/0.8785/0.9216. Logistic Regression beat Naive Bayes on all four measures; SVM tied LR on this holdout. The complete metrics and confusion matrices are in `reports/metrics.json`.

Five-fold StratifiedGroupKFold CV tunes all hyperparameters on the training partition only; spam F1 is the selection metric. Near-duplicate components stay within one holdout/CV partition. The computed exact-duplicate and near-duplicate overlap across the holdout was zero. Grid results are in `reports/cv_grid_results.csv`.

**HTML extraction ablation:** Naive Bayes recall was 0.47 with HTML fallback versus 0.22 without it. Logistic Regression was 0.90 with HTML fallback and 0.92 without it. The hypothesis is partly supported: HTML helps NB on this split but does not explain LR recall. Both representations use the same holdout and are tuned independently by training-only CV. Details are in `reports/html_ablation.csv`.

## Files

- `src/prepare_data.py`: download and parse data, remove exact duplicates, report class counts.
- `src/train_evaluate.py`: grouped split, five-fold CV grid search, train, evaluate, save pipelines and artifacts.
- `app.py`: paste-email classification UI.
- `models/`: local joblib pipelines (generated, gitignored).
- `reports/`: aggregate metrics, CV results, plots, HTML ablation and local error cases. Error examples are gitignored because they contain message text.
- `report.md`, `presentation.md`, `viva_questions.md`: project deliverables.

## Limitations and next steps

The source mail is old; easy ham is not representative of difficult modern cases; language and topic shift can break the learned vocabulary; TF-IDF ignores much email structure; and results depend on this one group-aware holdout. A large LSTM/transformer model was not added because the corpus is small and the comparison focuses on classic models. Future work could add hard ham, a newer independent email corpus, time-based validation, sender/header features, calibration and threshold analysis.
