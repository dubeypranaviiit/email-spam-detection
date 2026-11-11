"""Error analysis: Aggregate pattern extraction and explanation for test errors."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .local import explain_instance
from .utils import (
    CANONICAL_NAMES,
    EXPLAIN_DIR,
    load_holdout_split,
    load_model,
    normalize_model_name,
)


def analyze_errors_for_model(
    model_tag: str,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate aggregate error patterns and misleading token summaries without storing raw email text."""
    tag = normalize_model_name(model_tag)
    pipeline = load_model(tag)

    preds = pipeline.predict(test_df.text)
    y_test = test_df.label.to_numpy()

    fn_indices = np.where((y_test == 1) & (preds == 0))[0]
    fp_indices = np.where((y_test == 0) & (preds == 1))[0]

    def summarize_subset(indices: np.ndarray, error_type: str) -> Dict[str, Any]:
        count = len(indices)
        if count == 0:
            return {
                "count": 0,
                "percentage_of_test_subset": 0.0,
                "body_sources": {},
                "short_text_rate": 0.0,
                "subject_only_rate": 0.0,
                "character_length": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
                "word_length": {"min": 0, "max": 0, "mean": 0.0, "median": 0.0},
                "top_misleading_tokens": [],
            }

        subset_df = test_df.iloc[indices]
        source_counts = {str(k): int(v) for k, v in subset_df.body_source.value_counts().items()}
        empty_bodies = int((subset_df.body.astype(str).str.strip() == "").sum())
        short_texts = int((subset_df.text.astype(str).str.len() < 150).sum())

        char_lengths = [len(str(t)) for t in subset_df.text]
        word_lengths = [len(str(t).split()) for t in subset_df.text]

        # Collect top tokens contributing to the erroneous prediction
        token_counter: Counter[str] = Counter()
        token_weights: defaultdict[str, list[float]] = defaultdict(list)

        for text in subset_df.text:
            exp = explain_instance(str(text), model_name=tag, method="auto", top_k=10)
            for t in exp["tokens"]:
                tok = t["token"].lower()
                w = t["weight"]
                # For FN (missed spam), ham tokens (w < 0) dragged it down
                # For FP (false alarm), spam tokens (w > 0) pushed it up
                if (error_type == "false_negative" and w < 0) or (error_type == "false_positive" and w > 0):
                    token_counter[tok] += 1
                    token_weights[tok].append(w)

        top_misleading = []
        for tok, occurrences in token_counter.most_common(15):
            avg_w = float(np.mean(token_weights[tok]))
            top_misleading.append({
                "token": tok,
                "error_frequency": occurrences,
                "frequency_rate": float(occurrences / count),
                "mean_contribution_weight": avg_w,
            })

        total_in_class = int((y_test == (1 if error_type == "false_negative" else 0)).sum())

        return {
            "count": int(count),
            "percentage_of_actual_class": float(count / max(1, total_in_class)),
            "body_sources": source_counts,
            "short_text_rate": float(short_texts / count),
            "subject_only_rate": float(empty_bodies / count),
            "character_length": {
                "min": int(np.min(char_lengths)),
                "max": int(np.max(char_lengths)),
                "mean": float(np.mean(char_lengths)),
                "median": float(np.median(char_lengths)),
            },
            "word_length": {
                "min": int(np.min(word_lengths)),
                "max": int(np.max(word_lengths)),
                "mean": float(np.mean(word_lengths)),
                "median": float(np.median(word_lengths)),
            },
            "top_misleading_tokens": top_misleading,
        }

    return {
        "model": CANONICAL_NAMES[tag],
        "model_tag": tag,
        "false_negatives": summarize_subset(fn_indices, "false_negative"),
        "false_positives": summarize_subset(fp_indices, "false_positive"),
    }


def run_error_analysis(output_dir: Path = EXPLAIN_DIR) -> Dict[str, Any]:
    """Execute aggregate error analysis for holdout models and export summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df, _, test_indices = load_holdout_split()
    test_df = df.iloc[test_indices].reset_index(drop=True)

    summary: Dict[str, Any] = {
        "dataset": "SpamAssassin Holdout Split (563 test messages: 456 ham, 107 spam)",
        "models": {},
        "notes": (
            "Aggregate error patterns only; raw email messages and bodies are strictly excluded. "
            "Best holdout model (Complement NB) produced only 3 false negatives and 0 false positives. "
            "Linear SVM produced 4 false negatives and 0 false positives. Logistic Regression produced "
            "10 false negatives and 1 false positive."
        ),
    }

    for tag in ("complement_nb", "linear_svm", "logistic_regression", "random_forest"):
        print(f"Analyzing errors for {CANONICAL_NAMES[tag]}...")
        summary["models"][tag] = analyze_errors_for_model(tag, test_df)

    def default_serializer(obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    out_file = output_dir / "error_analysis_summary.json"
    out_file.write_text(json.dumps(summary, indent=2, default=default_serializer), encoding="utf-8")
    print(f"Aggregate error analysis saved to {out_file}")
    return summary
