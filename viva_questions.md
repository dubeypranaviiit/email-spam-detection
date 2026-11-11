# Viva questions and simple answers

1. **What is the project doing?** It learns from labeled email text and predicts whether a new message resembles spam or ham.
2. **Why is this supervised learning?** Each training message has a known class label, so the model learns a mapping from examples to labels.
3. **What is ham?** Legitimate, non-spam email in the corpus.
4. **Which dataset did you use?** Apache SpamAssassin Public Mail Corpus; this implementation uses the `easy_ham` and `spam` archives, not an SMS dataset.
5. **How are labels assigned?** From the corpus directory containing each message, rather than words or headers guessed by our code.
6. **Why parse subject and body?** Email is structured and often MIME encoded; the parser extracts subject and text/plain, then converts visible text/html when plain text is absent.
7. **What is TF-IDF?** A way to convert words into numbers that emphasizes terms important to a message but less common across the training collection.
8. **Why do you need a vectorizer?** Machine-learning classifiers need numeric feature vectors rather than raw strings.
9. **What is data leakage?** Evaluation data improperly influencing model fitting. Grouped splitting keeps similar emails together, and the pipeline fits TF-IDF only on each training fold.
10. **Why use the same split and feature method?** So differences in metrics reflect the classifiers and selected settings, not different examples or text representations.
11. **What is stratification?** Keeping roughly similar spam/ham proportions across folds. Grouping additionally keeps near-duplicate messages in one fold.
12. **Explain the Naive Bayes models.** Multinomial NB combines word evidence for each class; Complement NB estimates each class using evidence from the other classes, which can help with imbalanced text. Both use smoothing, and CV selected `alpha=0.01` in this run.
13. **Explain Logistic Regression.** It learns feature weights for the two classes and predicts from their combined linear score.
14. **What is precision?** Of messages predicted spam, the fraction that really are spam.
15. **What is recall?** Of actual spam messages, the fraction the model catches.
16. **What is F1?** A single harmonic-mean score balancing precision and recall.
17. **What is a false positive?** Ham predicted as spam; it may block legitimate mail.
18. **What is a false negative?** Spam predicted as ham; unwanted mail gets through.
19. **Why remove duplicates?** Copies shared between train and test can make evaluation look better than generalization really is.
20. **How did you handle near-duplicates?** A character-TF-IDF similarity screen connects highly similar messages into groups. Group-aware holdout and CV keep each group together; zero crossing is guaranteed by that split design, not an independent finding.
21. **Which model was better?** Complement Naive Bayes had the highest holdout spam F1 (0.99). Multinomial NB's accuracy was 0.99 vs LR's 0.98, recall 0.95 vs 0.91, and F1 0.98 vs 0.95; NB precision was also higher at 1.00 vs 0.99. Linear SVM had the highest mean CV F1 (0.9469 ± 0.0200), so the top CV and holdout models differed.
22. **Are predictions guaranteed or calibrated probabilities?** No. The app displays a classification label only; it is not a calibrated safety probability.
23. **What are the dataset limitations?** It is old, curated, imbalanced and uses easy ham, so results may not transfer to present-day mail.
24. **How do you run the application?** `streamlit run app.py` for the original Streamlit demo; or `uvicorn api.main:app` and `npm run dev` in `web/` for the modern XAI dashboard.
25. **What would you improve?** Test on newer independent email, add hard ham, evaluate time-based splits, inspect email headers, and study thresholds/calibration using validation data.

---

### New XAI, Faithfulness & Dashboard Viva Questions

26. **What is Explainable AI (XAI) in this project, and how is it implemented?**
    XAI decomposes a classifier's prediction into signed token-level contributions ($+$ for spam, $-$ for ham). For Logistic Regression and Linear SVM, token contribution is the exact closed-form product $w_j \cdot x_j$. For Naive Bayes, it is the exact token log-odds difference. For Random Forest, it uses local surrogate perturbations via LIME.

27. **Why is the token contribution exact for linear models and Naive Bayes, but approximate for Random Forest?**
    Linear models and Naive Bayes have strictly additive decision boundaries in logit/log space ($z = b + \sum w_j x_j$), allowing exact closed-form decomposition. Random Forest uses ensembles of non-linear decision trees with axis-aligned splits, which cannot be simplified into fixed independent token weights; hence LIME fits a local linear surrogate to approximate its local neighborhood.

28. **What is completeness in XAI, and did your explainers satisfy it?**
    Completeness requires that the baseline intercept plus the sum of all individual token contributions equals the model's exact decision score ($b + \sum c_j \equiv \text{score}$). All our exact linear and Naive Bayes explainers satisfy completeness to within machine precision ($< 1.35 \times 10^{-13}$), as verified in `tests/test_explain.py`.

29. **What is the token deletion test, and why is it considered a measure of faithfulness?**
    Faithfulness measures whether an explanation accurately reflects the model's true reasoning. In the deletion test, we remove the top-$k$ explained tokens ($k \in \{1, 3, 5, 10\}$) from the raw email text, re-vectorize and re-evaluate through the model, and compare the drop in predicted score against deleting $k$ random tokens (20 repeats per email). If the explanation is faithful, deleting top-$k$ tokens will cause a significantly larger score drop than random deletion.

30. **Is the token deletion test circular for linear models?**
    Yes, partially. For linear models, the decision score is mathematically defined by the exact token weights ($w_j x_j$), so removing the highest-weight tokens naturally reduces the score by definition. The truly independent and informative comparisons are LIME vs exact attribution, and Random Forest (non-linear) vs linear models.

31. **Why didn't deleting the top-1 token drop the spam score in 100% of Naive Bayes emails? (The L2 norm rescaling diagnosis)**
    In 3 of 87 test spam emails, deleting the top spam token did not drop the score more than random deletion (a 96.6% win rate). Our diagnosis is TF-IDF L2 norm rescaling: when a high-TF-IDF word is deleted from a message, the document vector norm $\|x\|_2$ shrinks, causing all remaining spam features to be mathematically rescaled upward by $1 / \sqrt{1 - x_{\text{deleted}}^2}$, which can occasionally compensate for the deleted token.

32. **Why does LIME agree strongly with exact linear models but diverge significantly on Naive Bayes?**
    On Logistic Regression and Linear SVM, LIME achieves high rank correlation ($\rho \approx 0.880$) because its local linear surrogate accurately captures the linear gradient. On Naive Bayes, agreement drops to $\rho \approx 0.165$. We hypothesize two causes: (1) extreme probability saturation in NB creates flat probability gradients near 0 and 1, introducing noise into LIME's perturbation objective; and (2) an objective mismatch (LIME optimizes probability differences, whereas exact NB explains log-odds).

33. **What did global feature explanations reveal about dataset artifacts and "shortcut learning"?**
    Global explanations revealed that the top ham indicators across all models are terms like `rpm`, `wrote`, `users`, `spambayes`, `razor`, and `2002`. These terms reflect the specific technical mailing lists from which the corpus was collected. The models achieve >99% holdout accuracy partly by learning these historical era shortcuts rather than universal properties of legitimate communication.

34. **Why are error analysis findings considered anecdotal in this project?**
    Because holdout accuracy is exceptionally high (>99%), models produce only 0 to 3 false positives and 3 to 13 false negatives across 563 test messages. With such small sample sizes, token patterns found in errors are illustrative and qualitative rather than statistically generalizable findings.

35. **How is the modern Next.js / FastAPI dashboard architected to protect user privacy?**
    The backend API enforces a 20,000-character ceiling, sliding-window rate limiting, and request timeouts. Critically, the API never logs email bodies or persists input text to disk. In the frontend, token highlights and decision margins are calculated on the fly without storing email content in git or browser storage.
