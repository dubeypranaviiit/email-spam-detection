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
24. **How do you run the application?** `streamlit run app.py` after preparing data and training the model.
25. **What would you improve?** Test on newer independent email, add hard ham, evaluate time-based splits, inspect email headers, and study thresholds/calibration using validation data.
