/**
 * Frontend API client communicating with FastAPI backend.
 * Uses process.env.NEXT_PUBLIC_API_URL or defaults to http://localhost:8000.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ModelMetric {
  model: string;
  best_params: Record<string, any>;
  train_fit_time_seconds: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
  confusion_matrix: [[number, number], [number, number]];
}

export interface ModelsResponse {
  models: ModelMetric[];
}

export interface MetricsSummary {
  holdout_test_size: number;
  cv_folds: number;
  best_overall_holdout_model: string;
  best_overall_holdout_f1: number;
  per_model: Record<string, ModelMetric>;
}

export interface MetricsBundleResponse {
  summary: MetricsSummary;
  model_comparison: Array<Record<string, any>>;
  cv_grid_results: Array<Record<string, any>>;
  html_ablation: Array<Record<string, any>>;
}

export interface PredictRequest {
  text: string;
  model?: string;
}

export interface PredictResponse {
  model: string;
  label: number;
  label_name: "ham" | "spam";
  spam_probability: number;
  confidence: number;
  char_count: number;
  token_count: number;
}

export interface TokenContribution {
  token: string;
  weight: number;
  abs_weight: number;
  sign: "+" | "-";
  indication: "spam" | "ham";
}

export interface ExplainRequest {
  text: string;
  model?: string;
  method?: string;
  top_k?: number;
}

export interface ExplainResponse {
  model: string;
  method: string;
  label: number;
  label_name: "ham" | "spam";
  spam_probability: number;
  base_value: number;
  score: number;
  tokens: TokenContribution[];
}

export interface GlobalTerm {
  term: string;
  weight: number;
}

export interface GlobalTermsResponse {
  model: string;
  method: string;
  top_spam_terms: GlobalTerm[];
  top_ham_terms: GlobalTerm[];
}

export interface DeletionTestResult {
  model: string;
  k: number;
  topk_prob_drop_mean: number;
  topk_prob_drop_ci_low: number;
  topk_prob_drop_ci_high: number;
  random_prob_drop_mean: number;
  random_prob_drop_ci_low: number;
  random_prob_drop_ci_high: number;
  prob_drop_diff: number;
  prob_win_rate_pct: number;
  topk_score_drop_mean: number;
  topk_score_drop_ci_low: number;
  topk_score_drop_ci_high: number;
  random_score_drop_mean: number;
  random_score_drop_ci_low: number;
  random_score_drop_ci_high: number;
  score_drop_diff: number;
  score_win_rate_pct: number;
  n_evaluated: number;
  sample_description: string;
}

export interface MethodAgreementResult {
  comparison: string;
  n_emails: number;
  feature_set: string;
  spearman_rho_mean: number;
  top5_jaccard_mean: number;
  top10_jaccard_mean: number;
  interpretation: string;
}

export interface FaithfulnessResponse {
  deletion_tests: DeletionTestResult[];
  method_agreement: MethodAgreementResult[];
}

export interface ErrorAnalysisSummary {
  models: Record<
    string,
    {
      total_test: number;
      test_spam: number;
      test_ham: number;
      false_positives_count: number;
      false_negatives_count: number;
      false_positive_rate_pct: number;
      false_negative_rate_pct: number;
      patterns: Record<string, { fp: number; fn: number }>;
      source_breakdown: Record<string, { fp: number; fn: number }>;
      length_summary: Record<string, any>;
      top_misleading_spam_tokens_in_fp: Array<{ term: string; count: number }>;
      top_misleading_ham_tokens_in_fn: Array<{ term: string; count: number }>;
    }
  >;
}

async function fetchFromApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers || {}),
      },
    });
    if (!res.ok) {
      let errorMsg = `API error ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) errorMsg = errorJson.detail;
      } catch {
        // ignore
      }
      throw new Error(errorMsg);
    }
    return (await res.json()) as T;
  } catch (err: any) {
    if (err.message && err.message.includes("Failed to fetch")) {
      throw new Error(
        `Unable to reach backend API at ${API_BASE}. Make sure the FastAPI server is running.`
      );
    }
    throw err;
  }
}

export const api = {
  getHealth: () => fetchFromApi<{ status: string; models_loaded: string[] }>("/health"),
  getModels: () => fetchFromApi<ModelsResponse>("/models"),
  getMetrics: () => fetchFromApi<MetricsBundleResponse>("/metrics"),
  predict: (req: PredictRequest) =>
    fetchFromApi<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  explain: (req: ExplainRequest) =>
    fetchFromApi<ExplainResponse>("/explain", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  getGlobalTerms: (model: string, top_k: number = 25) =>
    fetchFromApi<GlobalTermsResponse>(
      `/explain/global?model=${encodeURIComponent(model)}&top_k=${top_k}`
    ),
  getFaithfulness: () => fetchFromApi<FaithfulnessResponse>("/explain/faithfulness"),
  getErrors: () => fetchFromApi<ErrorAnalysisSummary>("/explain/errors"),
};
