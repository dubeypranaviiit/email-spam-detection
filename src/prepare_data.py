"""Download and parse the Apache SpamAssassin public email corpus."""
from __future__ import annotations
import argparse
import hashlib
import json
import io
import tarfile
import urllib.request
from email import policy
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "emails.csv"
BASE = "https://spamassassin.apache.org/old/publiccorpus/"
ARCHIVES = {"20030228_easy_ham.tar.bz2": 0, "20030228_spam.tar.bz2": 1}

def text_body(msg):
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart" or part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/plain":
                try: parts.append(part.get_content())
                except Exception: pass
    else:
        try: parts.append(msg.get_content())
        except Exception: pass
    return "\n".join(p if isinstance(p, str) else str(p) for p in parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="use archives already placed in data/raw")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    records = []
    for archive, label in ARCHIVES.items():
        path = RAW / archive
        if not args.local and not path.exists():
            print(f"Downloading {archive}...")
            urllib.request.urlretrieve(BASE + archive, path)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Download from {BASE} or rerun without --local.")
        with tarfile.open(path, "r:bz2") as tar:
            for member in tar.getmembers():
                if not member.isfile(): continue
                f = tar.extractfile(member)
                if f is None: continue
                try:
                    msg = BytesParser(policy=policy.default).parsebytes(f.read())
                    subject = str(msg.get("subject", "") or "").strip()
                    body = text_body(msg).strip()
                    content = (subject + "\n" + body).strip()
                    if not content: continue
                    norm = " ".join(content.casefold().split())
                    records.append({"subject": subject, "body": body, "text": content, "label": label,
                                    "label_name": "spam" if label else "ham", "source_file": member.name,
                                    "content_hash": hashlib.sha256(norm.encode("utf-8", "ignore")).hexdigest()})
                except Exception as exc:
                    print(f"Skipping malformed record {member.name}: {exc}")
    import pandas as pd
    df = pd.DataFrame(records)
    if df.empty: raise RuntimeError("No messages parsed.")
    before = len(df)
    conflict_hashes = df.groupby("content_hash").label.nunique()
    conflict_hashes = set(conflict_hashes[conflict_hashes > 1].index)
    conflicting_records = int(df.content_hash.isin(conflict_hashes).sum())
    if conflict_hashes:
        df = df[~df.content_hash.isin(conflict_hashes)].copy()
    before_dedup = len(df)
    df = df.drop_duplicates("content_hash", keep="first").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    # Similarity audit only: report highly similar messages; retain them because
    # campaign-template messages are still distinct corpus examples.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    char_vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), max_features=30000).fit_transform(df.text)
    similarities = cosine_similarity(char_vectors, dense_output=True)
    pairs = []
    for i in range(len(df)):
        for j in np.flatnonzero(similarities[i, i+1:] >= 0.95) + i + 1:
            pairs.append({"index_a": int(i), "index_b": int(j), "similarity": float(similarities[i, j]),
                          "label_a": df.iloc[i].label_name, "label_b": df.iloc[j].label_name,
                          "source_a": df.iloc[i].source_file, "source_b": df.iloc[j].source_file})
    pd.DataFrame(pairs, columns=["index_a", "index_b", "similarity", "label_a", "label_b", "source_a", "source_b"]).to_csv(
        OUT.parent / "near_duplicate_audit.csv", index=False, encoding="utf-8")
    (OUT.parent / "preparation.json").write_text(json.dumps({
        "parsed_records_before_deduplication": before,
        "exact_duplicates_removed": before_dedup-len(df),
        "ambiguous_cross_label_duplicate_records_excluded": conflicting_records,
        "usable_records_after_deduplication": len(df),
        "class_counts": {"ham": int((df.label == 0).sum()), "spam": int((df.label == 1).sum())},
        "near_duplicate_pairs_at_char_tfidf_cosine_0_95": len(pairs)
    }, indent=2), encoding="utf-8")
    print(f"Parsed {before}; excluded {conflicting_records} records in cross-label exact-duplicate groups; removed {before_dedup-len(df)} duplicate copies; saved {len(df)} to {OUT}")
    print("Class counts:")
    print(df.label_name.value_counts().to_string())
    print(f"Near-duplicate candidates (character TF-IDF cosine >= 0.95): {len(pairs)}; see data/processed/near_duplicate_audit.csv")

if __name__ == "__main__": main()
