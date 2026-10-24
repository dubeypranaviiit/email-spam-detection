"""Download and parse the Apache SpamAssassin public email corpus."""
from __future__ import annotations
import argparse
import hashlib
import json
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
    """Return readable body text and the selected MIME representation.

    Prefer text/plain when present to avoid duplicate text from
    multipart/alternative messages. Fall back to HTML text when needed.
    Attachments and non-text MIME parts are ignored.
    """
    from bs4 import BeautifulSoup

    plain_parts = []
    html_parts = []
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        if part.get_content_maintype() == "multipart" or part.get_content_disposition() == "attachment":
            continue
        if part.get_content_type() not in {"text/plain", "text/html"}:
            continue
        try:
            content = part.get_content()
        except Exception:
            continue
        if not isinstance(content, str):
            content = str(content)
        if part.get_content_type() == "text/plain":
            plain_parts.append(content)
        else:
            html_parts.append(content)

    if plain_parts:
        return "\n".join(plain_parts), "text/plain"
    if html_parts:
        soup = BeautifulSoup("\n".join(html_parts), "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        return soup.get_text(" ", strip=True), "text/html"
    return "", "no_text_body"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="use archives already placed in data/raw")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    records = []
    body_source_counts = {"text/plain": 0, "text/html": 0, "no_text_body": 0}
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
                    body, body_source = text_body(msg)
                    body = body.strip()
                    body_source_counts[body_source] += 1
                    content = (subject + "\n" + body).strip()
                    if not content: continue
                    plain_only_content = (subject + "\n" + (body if body_source == "text/plain" else "")).strip()
                    norm = " ".join(content.casefold().split())
                    records.append({"subject": subject, "body": body, "text": content,
                                    "text_without_html_fallback": plain_only_content, "label": label,
                                    "label_name": "spam" if label else "ham", "source_file": member.name,
                                    "body_source": body_source,
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
        "parsed_body_source_counts": body_source_counts,
        "near_duplicate_pairs_at_char_tfidf_cosine_0_95": len(pairs)
    }, indent=2), encoding="utf-8")
    print(f"Parsed {before}; excluded {conflicting_records} records in cross-label exact-duplicate groups; removed {before_dedup-len(df)} duplicate copies; saved {len(df)} to {OUT}")
    print("Class counts:")
    print(df.label_name.value_counts().to_string())
    print(f"Near-duplicate candidates (character TF-IDF cosine >= 0.95): {len(pairs)}; see data/processed/near_duplicate_audit.csv")

if __name__ == "__main__": main()
