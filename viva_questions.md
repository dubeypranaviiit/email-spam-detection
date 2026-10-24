# Viva questions and simple answers

1. **What is the project doing?** It learns from labeled email text and predicts whether a new message resembles spam or ham.
2. **Why is this supervised learning?** Each training message has a known class label, so the model learns a mapping from examples to labels.
3. **What is ham?** Legitimate, non-spam email in the corpus.
4. **Which dataset did you use?** Apache SpamAssassin Public Mail Corpus; this implementation uses the `easy_ham` and `spam` archives, not an SMS dataset.
5. **How are labels assigned?** From the corpus directory containing each message, rather than words or headers guessed by our code.
6. **Why parse subject and body?** Email is structured and often MIME encoded; parsing extracts the subject and readable plain-text content.
7. **What is TF-IDF?** A way to convert words into numbers that emphasizes terms important to a message but less common across the training collection.
8. **Why do you need a vectorizer?** Machine-learning classifiers need numeric feature vectors rather than raw strings.
9. **What is data leakage?** Information from the evaluation data improperly influencing model fitting. We split first and let each pipeline fit TF-IDF only on training text.
10. **Why use the same split for both?** So comparison is fair: differences are due to classifiers rather than different examples.
11. **What is stratification?** Keeping roughly the same class proportions in training and test sets.
12. **Explain Naive Bayes.** It combines word evidence for each class while assuming words are conditionally independent; `alpha=1` smooths unseen word counts.
13. **Explain Logistic Regression.** It learns feature weights for the two classes and predicts from their combined linear score.
14. **What is precision?** Of messages predicted spam, the fraction that really are spam.
15. **What is recall?** Of actual spam messages, the fraction the model catches.
16. **What is F1?** A single harmonic-mean score balancing precision and recall.
17. **What is a false positive?** Ham predicted as spam; it may block legitimate mail.
18. **What is a false negative?** Spam predicted as ham; unwanted mail gets through.
19. **Why remove duplicates?** Copies shared between train and test can make evaluation look better than generalization really is.
20. **Did you remove near-duplicates?** Exact normalized-text duplicates are removed. Sophisticated near-duplicate/template matching is not implemented and remains a limitation.
21. **Which model was better?** Read the actual metric table generated for this run and state the metric and tradeoff. Do not claim a winner before running it.
22. **Are predictions guaranteed or calibrated probabilities?** No. The app displays a classification label only; it is not a calibrated safety probability.
23. **What are the dataset limitations?** It is old, curated, imbalanced and uses easy ham, so results may not transfer to present-day mail.
24. **How do you run the application?** `streamlit run app.py` after preparing data and training the model.
25. **What would you improve?** Add hard ham and newer independent email data, stronger near-duplicate detection, validation-based threshold selection, calibration checks, and external testing.
