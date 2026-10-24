"""Train comparable TF-IDF + NB / Logistic Regression email classifiers."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/processed/emails.csv"; MODELS=ROOT/"models"; REPORTS=ROOT/"reports"
SEED=42; TEST_SIZE=.20

def main():
    if not DATA.exists(): raise FileNotFoundError(f"{DATA} missing. Run python src/prepare_data.py first.")
    df=pd.read_csv(DATA).fillna("")
    df=df.drop_duplicates("content_hash")
    X=df.text.astype(str); y=df.label.astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=TEST_SIZE,random_state=SEED,stratify=y)
    # Both models use identical vectorizer settings, split, and feature space.
    configs={"Multinomial Naive Bayes":MultinomialNB(alpha=1.0),
             "Logistic Regression":LogisticRegression(C=1.0,max_iter=1000,solver="liblinear",random_state=SEED)}
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    rows=[]; details={"seed":SEED,"test_size":TEST_SIZE,"train_count":len(Xtr),"test_count":len(Xte),
                      "class_distribution":df.label_name.value_counts().to_dict(),"models":{}}
    for name,clf in configs.items():
        pipe=Pipeline([("tfidf",TfidfVectorizer(lowercase=True,strip_accents="unicode",stop_words="english",
                                                ngram_range=(1,2),min_df=2,max_df=0.98,sublinear_tf=True)),("classifier",clf)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        values={"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),
                "recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0)}
        rows.append({"model":name,**values})
        tag="naive_bayes" if name.startswith("Multinomial") else "logistic_regression"
        joblib.dump(pipe,MODELS/f"{tag}.joblib")
        report=classification_report(yte,pred,target_names=["ham","spam"],output_dict=True,zero_division=0)
        details["models"][name]={**values,"classification_report":report,
            "confusion_matrix":confusion_matrix(yte,pred,labels=[0,1]).tolist()}
        errors=pd.DataFrame({"text":Xte.values,"true_label":yte.values,"predicted_label":pred})
        errors=errors[errors.true_label!=errors.predicted_label]
        errors.to_csv(REPORTS/f"{tag}_errors.csv",index=False,encoding="utf-8")
        cm=confusion_matrix(yte,pred,labels=[0,1])
        plt.figure(figsize=(5,4)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Ham","Spam"],yticklabels=["Ham","Spam"])
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{name} confusion matrix"); plt.tight_layout()
        plt.savefig(REPORTS/f"{tag}_confusion_matrix.png",dpi=160); plt.close()
    comp=pd.DataFrame(rows); comp.to_csv(REPORTS/"model_comparison.csv",index=False)
    details["duplicate_audit"]={"exact_duplicates_removed_during_preparation":int(pd.read_csv(DATA).duplicated("content_hash").sum()),
                                "cross_split_exact_hash_overlap":0}
    prep_file=DATA.parent/"preparation.json"
    if prep_file.exists():
        details["duplicate_audit"]["exact_duplicates_removed_during_preparation"]=json.loads(prep_file.read_text(encoding="utf-8"))["exact_duplicates_removed"]
    near_file=DATA.parent/"near_duplicate_audit.csv"
    if near_file.exists():
        near=pd.read_csv(near_file)
        train_ids=set(Xtr.index); test_ids=set(Xte.index)
        cross=near[((near.index_a.isin(train_ids)) & (near.index_b.isin(test_ids))) |
                   ((near.index_a.isin(test_ids)) & (near.index_b.isin(train_ids)))]
        details["duplicate_audit"].update({
            "near_duplicate_heuristic": "character TF-IDF cosine similarity >= 0.95 using 4-5 character ngrams",
            "near_duplicate_candidate_pairs": int(len(near)),
            "near_duplicate_pairs_crossing_train_test": int(len(cross)),
            "near_duplicate_pairs_crossing_train_test_with_different_labels": int((cross.label_a != cross.label_b).sum())
        })
    (REPORTS/"metrics.json").write_text(json.dumps(details,indent=2),encoding="utf-8")
    comp.set_index("model").plot(kind="bar",ylim=(0,1),figsize=(9,5),rot=10)
    plt.ylabel("Score"); plt.title("SpamAssassin email classification comparison"); plt.tight_layout()
    plt.savefig(REPORTS/"model_comparison.png",dpi=160); plt.close()
    print(f"Train={len(Xtr)}, test={len(Xte)}; class counts={df.label_name.value_counts().to_dict()}")
    print(comp.to_string(index=False,float_format=lambda n:f"{n:.4f}"))
    print(f"Artifacts saved in {REPORTS}")

if __name__=="__main__": main()
