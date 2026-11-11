"use client";

import { useEffect, useState } from "react";
import { api, MetricsBundleResponse, ModelMetric } from "@/lib/api";
import { formatPercent, formatDec } from "@/lib/utils";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  ShieldAlert,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  TrendingUp,
  Cpu,
  Layers,
  ArrowRight,
} from "lucide-react";
import Link from "next/link";

export default function OverviewPage() {
  const [data, setData] = useState<MetricsBundleResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("Complement NB");

  useEffect(() => {
    api
      .getMetrics()
      .then((res) => {
        setData(res);
        if (res.summary?.best_overall_holdout_model) {
          setSelectedModel(res.summary.best_overall_holdout_model);
        }
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || "Failed to load metrics data");
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-8 bg-muted rounded w-1/4" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-28 bg-muted rounded-lg" />
          ))}
        </div>
        <div className="h-80 bg-muted rounded-lg" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-6 bg-card border border-destructive/20 rounded-lg">
        <div className="flex items-center space-x-3 text-destructive mb-2">
          <AlertCircle className="w-5 h-5" />
          <h2 className="text-lg font-semibold">Backend Connection Unavailable</h2>
        </div>
        <p className="text-sm text-muted-foreground mb-4">
          {error || "Could not retrieve metrics. Ensure the FastAPI backend is running on port 8000."}
        </p>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-md hover:bg-primary/90 transition-colors"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  const { summary, model_comparison } = data;
  const bestModelMetric = summary.per_model[summary.best_overall_holdout_model] || Object.values(summary.per_model)[0];
  const activeMetric: ModelMetric = summary.per_model[selectedModel] || bestModelMetric;

  // Chart data formatting
  const chartData = model_comparison.map((m) => ({
    name: m.model,
    Precision: parseFloat((m.precision * 100).toFixed(1)),
    Recall: parseFloat((m.recall * 100).toFixed(1)),
    F1: parseFloat((m.f1 * 100).toFixed(1)),
  }));

  const [[tn, fp], [fn, tp]] = activeMetric.confusion_matrix;
  const total = tn + fp + fn + tp;

  return (
    <div className="space-y-8">
      {/* Title & Introduction */}
      <div className="border-b border-border pb-5">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
              Corpus Performance &amp; Model Benchmark
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Rigorous 5-fold Stratified Grouped Cross-Validation and held-out evaluation on the SpamAssassin public corpus.
            </p>
          </div>
          <Link
            href="/detector"
            className="inline-flex items-center justify-center space-x-2 px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 shadow-sm transition-colors"
          >
            <span>Launch Live Detector</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </div>

      {/* Headline Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-lg bg-card border border-border shadow-sm">
          <div className="flex items-center justify-between text-muted-foreground mb-1">
            <span className="text-xs uppercase tracking-wider font-semibold">Best Classifier</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="text-xl font-bold text-foreground truncate">{summary.best_overall_holdout_model}</div>
          <div className="text-xs text-muted-foreground mt-1">
            Holdout F1: <span className="font-semibold text-emerald-600 dark:text-emerald-400">{formatPercent(summary.best_overall_holdout_f1)}</span>
          </div>
        </div>

        <div className="p-4 rounded-lg bg-card border border-border shadow-sm">
          <div className="flex items-center justify-between text-muted-foreground mb-1">
            <span className="text-xs uppercase tracking-wider font-semibold">Holdout Accuracy</span>
            <TrendingUp className="w-4 h-4 text-blue-500" />
          </div>
          <div className="text-xl font-bold text-foreground">{formatPercent(bestModelMetric.accuracy)}</div>
          <div className="text-xs text-muted-foreground mt-1">
            {tn + tp} / {total} emails correct
          </div>
        </div>

        <div className="p-4 rounded-lg bg-card border border-border shadow-sm">
          <div className="flex items-center justify-between text-muted-foreground mb-1">
            <span className="text-xs uppercase tracking-wider font-semibold">Spam Precision</span>
            <ShieldAlert className="w-4 h-4 text-purple-500" />
          </div>
          <div className="text-xl font-bold text-foreground">{formatPercent(bestModelMetric.precision)}</div>
          <div className="text-xs text-muted-foreground mt-1">
            FP Rate: <span className="font-semibold text-foreground">{formatPercent(fp / (tn + fp))}</span> ({fp} false alarms)
          </div>
        </div>

        <div className="p-4 rounded-lg bg-card border border-border shadow-sm">
          <div className="flex items-center justify-between text-muted-foreground mb-1">
            <span className="text-xs uppercase tracking-wider font-semibold">Spam Recall</span>
            <ShieldCheck className="w-4 h-4 text-amber-500" />
          </div>
          <div className="text-xl font-bold text-foreground">{formatPercent(bestModelMetric.recall)}</div>
          <div className="text-xs text-muted-foreground mt-1">
            FN Rate: <span className="font-semibold text-foreground">{formatPercent(fn / (fn + tp))}</span> ({fn} missed spam)
          </div>
        </div>
      </div>

      {/* Model Comparison Bar Chart */}
      <div className="p-6 rounded-lg bg-card border border-border shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold text-foreground">Holdout Comparison (Precision, Recall, F1)</h2>
            <p className="text-xs text-muted-foreground">
              Evaluated on n={total} holdout emails isolated before hyperparameter grid searching.
            </p>
          </div>
          <div className="text-xs font-mono text-muted-foreground bg-secondary px-2.5 py-1 rounded">
            Metric: Spam F1 Score
          </div>
        </div>

        <div className="h-72 w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
              <YAxis domain={[85, 100]} tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} unit="%" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--card)",
                  borderColor: "var(--border)",
                  borderRadius: "0.5rem",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ paddingTop: "10px", fontSize: "12px" }} />
              <Bar dataKey="Precision" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Recall" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="F1" fill="#2563eb" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrix and Model Details */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Confusion Matrix Viewer */}
        <div className="lg:col-span-1 p-6 rounded-lg bg-card border border-border shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-foreground">Confusion Matrix</h3>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="text-xs bg-secondary border border-border rounded px-2 py-1 text-foreground focus:ring-1 focus:ring-primary"
            >
              {Object.keys(summary.per_model).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-2 text-center pt-2">
            <div className="p-3 rounded-md bg-secondary/60 border border-border">
              <div className="text-xs text-muted-foreground uppercase font-semibold">True Negative (Ham)</div>
              <div className="text-2xl font-bold text-foreground mt-1">{tn}</div>
              <div className="text-[11px] text-muted-foreground">{formatPercent(tn / (tn + fp))} specificity</div>
            </div>
            <div className="p-3 rounded-md bg-rose-50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/50">
              <div className="text-xs text-rose-600 dark:text-rose-400 uppercase font-semibold">False Positive</div>
              <div className="text-2xl font-bold text-rose-600 dark:text-rose-400 mt-1">{fp}</div>
              <div className="text-[11px] text-muted-foreground">ham classified as spam</div>
            </div>
            <div className="p-3 rounded-md bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/50">
              <div className="text-xs text-amber-600 dark:text-amber-400 uppercase font-semibold">False Negative</div>
              <div className="text-2xl font-bold text-amber-600 dark:text-amber-400 mt-1">{fn}</div>
              <div className="text-[11px] text-muted-foreground">spam missed</div>
            </div>
            <div className="p-3 rounded-md bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900/50">
              <div className="text-xs text-emerald-600 dark:text-emerald-400 uppercase font-semibold">True Positive (Spam)</div>
              <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 mt-1">{tp}</div>
              <div className="text-[11px] text-muted-foreground">{formatPercent(tp / (fn + tp))} sensitivity</div>
            </div>
          </div>

          <div className="text-xs text-muted-foreground border-t border-border pt-3 space-y-1">
            <div><span className="font-semibold text-foreground">Selected Params:</span> <code className="font-mono text-[11px]">{JSON.stringify(activeMetric.best_params)}</code></div>
            <div><span className="font-semibold text-foreground">Training Time:</span> {formatDec(activeMetric.train_fit_time_seconds, 2)}s</div>
          </div>
        </div>

        {/* Plain-English Summary of Findings */}
        <div className="lg:col-span-2 p-6 rounded-lg bg-card border border-border shadow-sm space-y-4">
          <div className="flex items-center space-x-2 text-foreground">
            <Layers className="w-5 h-5 text-primary" />
            <h3 className="text-base font-semibold">Summary of Empirical Findings</h3>
          </div>

          <div className="text-sm text-muted-foreground space-y-3 leading-relaxed">
            <p>
              <strong className="text-foreground">Near-zero error rates on held-out data:</strong> All five tuned pipelines achieve holdout spam F1 scores between 97.8% and 98.9%. Complement Naive Bayes attained the top score (F1 = 98.88%, with only 1 false positive and 1 false negative).
            </p>
            <p>
              <strong className="text-foreground">Complement NB vs Multinomial NB threshold effect:</strong> Our mathematical derivation proves that in binary classification with unnormalized weights, Complement NB and Multinomial NB yield identical feature weights (<code className="font-mono text-xs">w_j^(CNB) ≡ w_j^(MNB)</code>). Their sole difference is the decision threshold intercept (<code className="font-mono text-xs">b = 0</code> vs <code className="font-mono text-xs">b = log(N_spam / N_ham)</code>). CNB&apos;s holdout edge is strictly a decision threshold calibration advantage, not superior feature modeling.
            </p>
            <p>
              <strong className="text-foreground">Corpus artifacts and generalization limitation:</strong> Global explanation inspection reveals that models heavily exploit corpus-era artifacts such as <code className="font-mono text-xs">rpm</code>, <code className="font-mono text-xs">spambayes</code>, <code className="font-mono text-xs">razor</code>, and <code className="font-mono text-xs">2002</code>. While achieving near-perfect metrics on the SpamAssassin benchmark, real-world deployment on contemporary email requires ongoing retraining or domain adaptation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
