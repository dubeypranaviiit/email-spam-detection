"use client";

import { useEffect, useState } from "react";
import {
  api,
  FaithfulnessResponse,
  GlobalTermsResponse,
  DeletionTestResult,
  MethodAgreementResult,
} from "@/lib/api";
import { formatPercent, formatDec } from "@/lib/utils";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  Sparkles,
  Award,
  AlertCircle,
  HelpCircle,
  BookOpen,
  Filter,
  CheckCircle2,
} from "lucide-react";

export default function ExplainabilityPage() {
  const [selectedModel, setSelectedModel] = useState<string>("logistic_regression");
  const [globalTerms, setGlobalTerms] = useState<GlobalTermsResponse | null>(null);
  const [faithfulness, setFaithfulness] = useState<FaithfulnessResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Load faithfulness benchmarks once
    api
      .getFaithfulness()
      .then((res) => {
        setFaithfulness(res);
      })
      .catch((err) => {
        setError(err.message || "Failed to load faithfulness data");
      });
  }, []);

  useEffect(() => {
    // Load global terms when selectedModel changes
    setLoading(true);
    api
      .getGlobalTerms(selectedModel, 25)
      .then((res) => {
        setGlobalTerms(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || "Failed to load global terms");
        setLoading(false);
      });
  }, [selectedModel]);

  // Safely normalize deletion tests for the active model (handles both dict and array formats)
  const activeDeletionData: DeletionTestResult[] = (() => {
    if (!faithfulness?.deletion_tests) return [];

    // Case A: If deletion_tests is already a flat array (e.g. from tabular CSV)
    if (Array.isArray(faithfulness.deletion_tests)) {
      return (faithfulness.deletion_tests as any[]).filter(
        (d) =>
          d.model?.toLowerCase().replace(/\s+/g, "_") === selectedModel.toLowerCase() ||
          d.model_tag?.toLowerCase() === selectedModel.toLowerCase() ||
          (selectedModel === "logistic_regression" && d.model?.includes("Logistic")) ||
          (selectedModel === "linear_svm" && d.model?.includes("Linear SVM")) ||
          (selectedModel === "multinomial_nb" && (d.model?.includes("Multinomial") || d.model_tag === "naive_bayes")) ||
          (selectedModel === "complement_nb" && d.model?.includes("Complement")) ||
          (selectedModel === "random_forest" && d.model?.includes("Random Forest"))
      );
    }

    // Case B: If deletion_tests is a dictionary keyed by model_tag (from faithfulness.json)
    const testsObj = faithfulness.deletion_tests as Record<string, any>;
    const tag = selectedModel === "multinomial_nb" ? "naive_bayes" : selectedModel;
    const modelData = testsObj[tag] || testsObj[selectedModel] || Object.values(testsObj)[0];

    if (!modelData || !modelData.deletion_results) return [];

    const results = modelData.deletion_results;
    return [1, 3, 5, 10].map((k) => {
      const res = results[String(k)] || {};
      const topkDrop = res.mean_prob_drop_top_k ?? 0;
      const randDrop = res.mean_prob_drop_random ?? 0;
      const topkScoreDrop = res.mean_score_drop_top_k ?? 0;
      const randScoreDrop = res.mean_score_drop_random ?? 0;

      return {
        model: modelData.model || selectedModel,
        k,
        topk_prob_drop_mean: topkDrop,
        topk_prob_drop_ci_low: res.ci_prob_drop_top_k_95?.[0] ?? 0,
        topk_prob_drop_ci_high: res.ci_prob_drop_top_k_95?.[1] ?? 0,
        random_prob_drop_mean: randDrop,
        random_prob_drop_ci_low: res.ci_prob_drop_random_95?.[0] ?? 0,
        random_prob_drop_ci_high: res.ci_prob_drop_random_95?.[1] ?? 0,
        prob_drop_diff: topkDrop - randDrop,
        prob_win_rate_pct: (res.win_rate_prob_vs_random ?? 1.0) * 100,
        topk_score_drop_mean: topkScoreDrop,
        topk_score_drop_ci_low: res.ci_score_drop_top_k_95?.[0] ?? 0,
        topk_score_drop_ci_high: res.ci_score_drop_top_k_95?.[1] ?? 0,
        random_score_drop_mean: randScoreDrop,
        random_score_drop_ci_low: res.ci_score_drop_random_95?.[0] ?? 0,
        random_score_drop_ci_high: res.ci_score_drop_random_95?.[1] ?? 0,
        score_drop_diff: topkScoreDrop - randScoreDrop,
        score_win_rate_pct: (res.win_rate_score_vs_random ?? 1.0) * 100,
        n_evaluated: modelData.n_evaluated ?? 87,
        sample_description: "Mutual holdout spam",
      };
    });
  })();

  // Comparison name formatting
  const formatComparisonName = (key: string): string => {
    const map: Record<string, string> = {
      logistic_regression_exact_vs_lime: "Logistic Regression: Exact vs LIME",
      linear_svm_exact_vs_lime: "Linear SVM: Exact vs LIME",
      naive_bayes_exact_vs_lime: "Multinomial NB: Exact vs LIME",
      complement_nb_exact_vs_lime: "Complement NB: Exact vs LIME",
      random_forest_lime_vs_logistic_regression_exact: "Random Forest (LIME) vs Logistic Regression (Exact)",
      random_forest_lime_vs_linear_svm_exact: "Random Forest (LIME) vs Linear SVM (Exact)",
      random_forest_lime_vs_complement_nb_exact: "Random Forest (LIME) vs Complement NB (Exact)",
      logistic_regression_exact_vs_linear_svm_exact: "Logistic Regression (Exact) vs Linear SVM (Exact)",
    };
    return map[key] || key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  };

  const getInterpretation = (key: string): string => {
    if (key.includes("naive_bayes") || key.includes("complement_nb_exact_vs_lime")) {
      return "Low rank agreement: severe NB probability saturation (step-like surface) & log-odds mismatch.";
    }
    if (key.includes("logistic_regression_exact_vs_linear_svm")) {
      return "Very high agreement (ρ > 0.94): both linear models identify nearly identical decision surfaces.";
    }
    if (key.includes("random_forest")) {
      return "Moderate agreement (ρ ≈ 0.46–0.50): tree splits capture non-linear interactions while sharing core linear unigrams.";
    }
    return "High rank correlation (ρ > 0.86): LIME local perturbation accurately recovers linear hyperplanes.";
  };

  // Safely normalize method agreement entries (handles both dict and array formats)
  const agreementData: MethodAgreementResult[] = (() => {
    if (!faithfulness?.method_agreement) return [];

    if (Array.isArray(faithfulness.method_agreement)) {
      return (faithfulness.method_agreement as any[]).map((item) => ({
        comparison: item.comparison || "Comparison",
        n_emails: item.n_emails ?? 30,
        spearman_rho_mean: item.spearman_rho_mean ?? 0,
        top5_jaccard_mean: item.top5_jaccard_mean ?? 0,
        top10_jaccard_mean: item.top10_jaccard_mean ?? 0,
        interpretation: item.interpretation || getInterpretation(item.comparison || ""),
      }));
    }

    const agreeObj = faithfulness.method_agreement as Record<string, any>;
    return Object.entries(agreeObj).map(([key, item]) => ({
      comparison: formatComparisonName(key),
      n_emails: item.n_evaluated ?? 30,
      spearman_rho_mean: item.mean_spearman_rho ?? 0,
      top5_jaccard_mean: item.mean_top_5_jaccard ?? 0,
      top10_jaccard_mean: item.mean_top_10_jaccard ?? 0,
      interpretation: getInterpretation(key),
    }));
  })();

  // Chart data for probability drop
  const probChartData = [1, 3, 5, 10].map((k) => {
    const row = activeDeletionData.find((d) => d.k === k);
    return {
      k: `k=${k}`,
      TopK_Drop: row ? parseFloat((row.topk_prob_drop_mean * 100).toFixed(2)) : 0,
      Random_Drop: row ? parseFloat((row.random_prob_drop_mean * 100).toFixed(2)) : 0,
    };
  });

  // Chart data for score drop
  const scoreChartData = [1, 3, 5, 10].map((k) => {
    const row = activeDeletionData.find((d) => d.k === k);
    return {
      k: `k=${k}`,
      TopK_Score_Drop: row ? parseFloat(row.topk_score_drop_mean.toFixed(3)) : 0,
      Random_Score_Drop: row ? parseFloat(row.random_score_drop_mean.toFixed(3)) : 0,
    };
  });

  return (
    <div className="space-y-10">
      {/* Title */}
      <div className="border-b border-border pb-5">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
          Explainable AI &amp; Faithfulness Evaluation
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Global term contributions, rigorous deletion perturbation benchmarks, and cross-method agreement metrics.
        </p>
      </div>

      {/* Model Selector Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-lg bg-card border border-border shadow-sm">
        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-primary" />
          <span className="text-sm font-semibold text-foreground">Select Classifier:</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {[
            { id: "logistic_regression", label: "Logistic Regression" },
            { id: "linear_svm", label: "Linear SVM" },
            { id: "complement_nb", label: "Complement NB" },
            { id: "multinomial_nb", label: "Multinomial NB" },
            { id: "random_forest", label: "Random Forest" },
          ].map((m) => (
            <button
              key={m.id}
              onClick={() => setSelectedModel(m.id)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                selectedModel === m.id
                  ? "bg-primary text-primary-foreground font-semibold shadow-sm"
                  : "bg-secondary text-muted-foreground hover:text-foreground hover:bg-secondary/70"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* SECTION 1: GLOBAL TOP TERMS */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
          <div>
            <h2 className="text-lg font-bold text-foreground">
              Global Top 25 Predictive Terms ({globalTerms?.model || selectedModel})
            </h2>
            <p className="text-xs text-muted-foreground">
              Highest-magnitude features pushing classification toward Spam vs Ham.
            </p>
          </div>
          <div className="text-xs font-mono text-muted-foreground bg-secondary px-2.5 py-1 rounded">
            Method: {globalTerms?.method || "Exact Coefficients"}
          </div>
        </div>

        {selectedModel === "random_forest" && (
          <div className="p-3 text-xs bg-accent/40 border border-accent text-accent-foreground rounded-md">
            <strong>Methodology Note:</strong> Random Forest feature importances (Gini impurities) are unsigned. Signed terms are derived by weighting each feature&apos;s Gini importance by its class conditional mean difference: <code className="font-mono text-[11px]">sign(μ_spam - μ_ham) × importance</code>.
          </div>
        )}

        {(selectedModel === "complement_nb" || selectedModel === "multinomial_nb") && (
          <div className="p-3 text-xs bg-accent/40 border border-accent text-accent-foreground rounded-md space-y-1">
            <strong>Theoretical Note (Rennie et al., 2003):</strong> In the binary classification case, Complement NB and Multinomial NB have <em>identical feature weights</em> (<code className="font-mono text-[11px]">w_j = log θ_j - log θ&#773;_j</code>). They differ only in the intercept threshold (<code className="font-mono text-[11px]">b_CNB = log(N_ham / N_spam)</code> vs <code className="font-mono text-[11px]">b_MNB = log(N_spam / N_ham)</code>). CNB&apos;s holdout advantage is therefore an empirical threshold shift effect on imbalanced classes, not superior feature modelling.
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Top Spam Terms */}
          <div className="p-5 rounded-lg bg-card border border-border shadow-sm space-y-3">
            <div className="flex items-center space-x-2 text-rose-600 dark:text-rose-400 font-semibold text-sm border-b border-border pb-2">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" />
              <span>Top Spam Indicators (+)</span>
            </div>

            <div className="space-y-1.5 max-h-96 overflow-y-auto pr-1">
              {globalTerms?.top_spam_terms.map((t, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between text-xs py-1 px-2 rounded hover:bg-secondary/50"
                >
                  <span className="font-mono text-foreground">
                    <span className="text-muted-foreground mr-2 font-sans">{idx + 1}.</span>
                    {t.term}
                  </span>
                  <span className="font-mono font-semibold text-rose-600 dark:text-rose-400">
                    +{formatDec(t.weight, 4)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Top Ham Terms */}
          <div className="p-5 rounded-lg bg-card border border-border shadow-sm space-y-3">
            <div className="flex items-center space-x-2 text-emerald-600 dark:text-emerald-400 font-semibold text-sm border-b border-border pb-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
              <span>Top Ham Indicators (-)</span>
            </div>

            <div className="space-y-1.5 max-h-96 overflow-y-auto pr-1">
              {globalTerms?.top_ham_terms.map((t, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between text-xs py-1 px-2 rounded hover:bg-secondary/50"
                >
                  <span className="font-mono text-foreground">
                    <span className="text-muted-foreground mr-2 font-sans">{idx + 1}.</span>
                    {t.term}
                  </span>
                  <span className="font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                    {formatDec(t.weight, 4)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 2: FAITHFULNESS DELETION BENCHMARK */}
      <div className="space-y-6 pt-4 border-t border-border">
        <div>
          <h2 className="text-lg font-bold text-foreground">
            Faithfulness Evaluation: Token Deletion Tests (k = 1, 3, 5, 10)
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Evaluated on all n=87 mutual test spam emails correctly classified by all five models. Top-k explained tokens are deleted from the raw text and re-evaluated through the full vectorizer and model pipeline against 20 random token baseline repetitions.
          </p>
        </div>

        {/* Charts: Prob Drop and Score Drop */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-5 rounded-lg bg-card border border-border shadow-sm space-y-3">
            <h3 className="text-sm font-semibold text-foreground">
              Spam Probability Drop (ΔP) vs Random Baseline
            </h3>
            <div className="h-60 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={probChartData} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="k" tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
                  <YAxis unit="%" tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--card)",
                      borderColor: "var(--border)",
                      borderRadius: "0.5rem",
                      fontSize: "12px",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <Line
                    type="monotone"
                    dataKey="TopK_Drop"
                    name="Top-k Explained Tokens"
                    stroke="#2563eb"
                    strokeWidth={2.5}
                    dot={{ r: 4 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="Random_Drop"
                    name="Random Baseline (20 repeats)"
                    stroke="#94a3b8"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="p-5 rounded-lg bg-card border border-border shadow-sm space-y-3">
            <h3 className="text-sm font-semibold text-foreground">
              Decision Score Drop (Δz / Log-odds) vs Random Baseline
            </h3>
            <div className="h-60 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={scoreChartData} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="k" tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
                  <YAxis tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--card)",
                      borderColor: "var(--border)",
                      borderRadius: "0.5rem",
                      fontSize: "12px",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <Line
                    type="monotone"
                    dataKey="TopK_Score_Drop"
                    name="Top-k Explained Score Drop"
                    stroke="#10b981"
                    strokeWidth={2.5}
                    dot={{ r: 4 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="Random_Score_Drop"
                    name="Random Baseline"
                    stroke="#94a3b8"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Deletion Metrics Table */}
        <div className="p-5 rounded-lg bg-card border border-border shadow-sm space-y-3">
          <h3 className="text-sm font-semibold text-foreground">
            Detailed Deletion Perturbation Statistics (n=87 Mutual Test Spam)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-border text-muted-foreground uppercase text-[10px] tracking-wider font-semibold">
                  <th className="py-2.5 px-3">k</th>
                  <th className="py-2.5 px-3">Top-k ΔP [95% CI]</th>
                  <th className="py-2.5 px-3">Random ΔP</th>
                  <th className="py-2.5 px-3">ΔP Advantage</th>
                  <th className="py-2.5 px-3">Top-k Δz Score</th>
                  <th className="py-2.5 px-3">Random Δz</th>
                  <th className="py-2.5 px-3">Score Win Rate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border font-mono">
                {activeDeletionData.map((d) => (
                  <tr key={d.k} className="hover:bg-secondary/40">
                    <td className="py-2 px-3 font-semibold text-foreground">k = {d.k}</td>
                    <td className="py-2 px-3 text-foreground">
                      {formatPercent(d.topk_prob_drop_mean)}{" "}
                      <span className="text-[10px] text-muted-foreground font-sans">
                        [{formatPercent(d.topk_prob_drop_ci_low)}, {formatPercent(d.topk_prob_drop_ci_high)}]
                      </span>
                    </td>
                    <td className="py-2 px-3 text-muted-foreground">
                      {formatPercent(d.random_prob_drop_mean)}
                    </td>
                    <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400 font-semibold">
                      +{formatPercent(d.prob_drop_diff)}
                    </td>
                    <td className="py-2 px-3 text-foreground">
                      {formatDec(d.topk_score_drop_mean, 3)}
                    </td>
                    <td className="py-2 px-3 text-muted-foreground">
                      {formatDec(d.random_score_drop_mean, 3)}
                    </td>
                    <td className="py-2 px-3 font-semibold text-blue-600 dark:text-blue-400">
                      {formatPercent(d.score_win_rate_pct / 100)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="text-[11px] text-muted-foreground pt-2 space-y-1">
            <div>
              &bull; <strong>Circularity Note:</strong> For linear models, the deletion test is partly circular because the exact weights determine the linear prediction. The informative benchmarks are LIME vs exact and Random Forest vs the others.
            </div>
            <div>
              &bull; <strong>L2-renormalization Diagnosis:</strong> In Naive Bayes at k=1, the win rate is 96.6% (84/87). In the 3 non-winning instances, deleting a high-TF-IDF spam token shrinks the email vector&apos;s L2 norm, rescaling remaining spam tokens by <code className="font-mono text-[10px]">1 / √(1 - x₁²)</code>. Note: Deletion matching was corrected after the initial run; this L2-renormalization explanation is a diagnosis, not a mathematically proven result.
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 3: METHOD AGREEMENT */}
      <div className="space-y-4 pt-4 border-t border-border">
        <div>
          <h2 className="text-lg font-bold text-foreground">
            Cross-Method &amp; Cross-Model Explanation Agreement
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Rank correlation (Spearman ρ) and token overlap (Top-5 and Top-10 Jaccard similarity) evaluated over active email unigrams on a fixed random sample (n=20-30 emails, bounded by LIME perturbation overhead).
          </p>
        </div>

        <div className="overflow-x-auto rounded-lg bg-card border border-border shadow-sm">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-border text-muted-foreground uppercase text-[10px] tracking-wider font-semibold bg-secondary/50">
                <th className="py-3 px-4">Comparison Pair</th>
                <th className="py-3 px-3">Sample n</th>
                <th className="py-3 px-3">Spearman ρ</th>
                <th className="py-3 px-3">Top-5 Jaccard</th>
                <th className="py-3 px-3">Top-10 Jaccard</th>
                <th className="py-3 px-4">Key Takeaway</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {agreementData.map((m, idx) => (
                <tr key={idx} className="hover:bg-secondary/40">
                  <td className="py-2.5 px-4 font-semibold text-foreground">{m.comparison}</td>
                  <td className="py-2.5 px-3 font-mono text-muted-foreground">{m.n_emails}</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-foreground">
                    {formatDec(m.spearman_rho_mean, 3)}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-foreground">
                    {formatPercent(m.top5_jaccard_mean)}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-foreground">
                    {formatPercent(m.top10_jaccard_mean)}
                  </td>
                  <td className="py-2.5 px-4 text-xs text-muted-foreground font-sans max-w-xs">
                    {m.interpretation}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Scientific Interpretation */}
        <div className="p-5 rounded-lg bg-card border border-border shadow-sm space-y-2">
          <div className="flex items-center space-x-2 text-foreground font-semibold text-sm">
            <BookOpen className="w-4 h-4 text-primary" />
            <span>Scientific Interpretation of Agreement Results</span>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            The high rank correlation on Logistic Regression (ρ = 0.880) and Linear SVM (ρ = 0.865) confirms that LIME effectively recovers the linear model&apos;s first-order decision surface under local perturbations. For Naive Bayes, low agreement (ρ = 0.165) is hypothesized to result from two factors: (1) extreme probability saturation in NB yielding flat probability gradients across perturbations, and (2) a target mismatch, where LIME fits perturbations in probability space while exact NB is additive in log-odds. The moderate agreement between Random Forest and linear models (ρ ≈ 0.45, Top-5 Jaccard ≈ 29%) highlights that tree ensembles capture non-linear token interactions while still sharing core spam signals.
          </p>
        </div>
      </div>
    </div>
  );
}
