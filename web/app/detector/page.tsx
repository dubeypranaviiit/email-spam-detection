"use client";

import { useState } from "react";
import { api, ExplainResponse, PredictResponse, TokenContribution } from "@/lib/api";
import { formatPercent, formatDec } from "@/lib/utils";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import {
  ShieldAlert,
  ShieldCheck,
  Zap,
  Info,
  RefreshCw,
  Sparkles,
  AlertTriangle,
  FileText,
} from "lucide-react";

const SAMPLE_SPAM = `Subject: Urgent Refinance Notice: Lower Your Rate to 3.25% Today!
Dear Homeowner,

You have been pre-approved for an exclusive mortgage refinancing package.
Act now to consolidate your high-interest credit card debt into one low monthly payment.
This limited-time rate of 3.25% APR is guaranteed for 30 years. No upfront closing costs!

Click here immediately to claim your instant approval voucher:
http://mortgage-refinance-deals-now.com/apply

Sincerely,
National Lending Department`;

const SAMPLE_HAM = `Subject: [PATCH] Fix buffer overflow in socket initialization routine
Hi team,

Attached is a patch addressing the memory leak reported in issue #4821.
During socket reconnection on high-latency links, the buffer pointer was not being reset prior to reading the next incoming frame.

I have tested this against the regression suite on both Linux and FreeBSD kernels without any regressions. Please review and let me know if any adjustments are needed before merging.

Thanks,
Alex`;

export default function DetectorPage() {
  const [text, setText] = useState<string>(SAMPLE_SPAM);
  const [model, setModel] = useState<string>("logistic_regression");
  const [method, setMethod] = useState<string>("exact");
  const [topK, setTopK] = useState<number>(10);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please enter email text to analyze.");
      return;
    }
    setLoading(true);
    setError(null);

    try {
      // 1. Run prediction
      const pred = await api.predict({ text, model });
      setPrediction(pred);

      // 2. Run explanation
      const expl = await api.explain({ text, model, method, top_k: topK });
      setExplanation(expl);
    } catch (err: any) {
      setError(err.message || "Failed to analyze email.");
    } finally {
      setLoading(false);
    }
  };

  // Build highlighted email text segments
  const renderHighlightedText = () => {
    if (!explanation || explanation.tokens.length === 0) {
      return <div className="whitespace-pre-wrap font-sans text-sm">{text}</div>;
    }

    // Map token weights for fast case-insensitive lookup
    const tokenMap = new Map<string, TokenContribution>();
    for (const t of explanation.tokens) {
      tokenMap.set(t.token.toLowerCase(), t);
    }

    // Tokenize text keeping spaces and punctuation intact
    const wordsAndSpaces = text.split(/(\s+|[^\w\s]+)/);

    return (
      <div className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-foreground bg-card p-4 rounded-md border border-border">
        {wordsAndSpaces.map((part, idx) => {
          const lower = part.toLowerCase();
          const match = tokenMap.get(lower);

          if (!match) {
            return <span key={idx}>{part}</span>;
          }

          const isSpam = match.sign === "+";
          return (
            <span
              key={idx}
              className={isSpam ? "token-spam" : "token-ham"}
              title={`Token: "${match.token}" | Weight: ${match.weight > 0 ? "+" : ""}${match.weight.toFixed(4)} (${match.indication})`}
            >
              {part}
              <span className="text-[10px] ml-0.5 opacity-80 font-mono font-bold">
                ({match.sign})
              </span>
            </span>
          );
        })}
      </div>
    );
  };

  // Format bar chart data
  const barChartData = explanation?.tokens.map((t) => ({
    name: t.token,
    weight: parseFloat(t.weight.toFixed(4)),
    absWeight: parseFloat(t.abs_weight.toFixed(4)),
    indication: t.indication,
    sign: t.sign,
  })) || [];

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="border-b border-border pb-5">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
          Live Detector &amp; Exact Token Attribution
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Classify arbitrary email text in real time with transparent, mathematically faithful feature contributions.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Input and Controls (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="p-5 rounded-lg bg-card border border-border shadow-sm space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <label htmlFor="email-text" className="text-sm font-semibold text-foreground flex items-center gap-1.5">
                <FileText className="w-4 h-4 text-primary" />
                <span>Email Content</span>
              </label>
              <div className="flex items-center space-x-2">
                <span className="text-xs text-muted-foreground">Load Sample:</span>
                <button
                  type="button"
                  onClick={() => setText(SAMPLE_SPAM)}
                  className="px-2.5 py-1 text-xs rounded bg-rose-50 dark:bg-rose-950/30 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-900/40 hover:bg-rose-100 transition-colors"
                >
                  Spam Sample
                </button>
                <button
                  type="button"
                  onClick={() => setText(SAMPLE_HAM)}
                  className="px-2.5 py-1 text-xs rounded bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-900/40 hover:bg-emerald-100 transition-colors"
                >
                  Ham Sample
                </button>
              </div>
            </div>

            <textarea
              id="email-text"
              rows={11}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste email headers and body text here..."
              maxLength={20000}
              className="w-full p-3 rounded-md border border-input bg-background text-foreground text-sm font-mono placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary leading-relaxed"
            />

            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Limit: 20,000 characters &bull; Email bodies are never logged or stored</span>
              <span>{text.length} / 20,000 chars</span>
            </div>

            {/* Model & Explainer Controls */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 border-t border-border">
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Classifier
                </label>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="w-full text-xs bg-secondary border border-border rounded px-2.5 py-2 text-foreground focus:ring-1 focus:ring-primary"
                >
                  <option value="logistic_regression">Logistic Regression</option>
                  <option value="linear_svm">Linear SVM</option>
                  <option value="complement_nb">Complement NB (Best)</option>
                  <option value="multinomial_nb">Multinomial NB</option>
                  <option value="random_forest">Random Forest</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  XAI Method
                </label>
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  className="w-full text-xs bg-secondary border border-border rounded px-2.5 py-2 text-foreground focus:ring-1 focus:ring-primary"
                >
                  <option value="exact">Exact Contribution (Faithful)</option>
                  <option value="lime">LIME Explainer (Perturbation)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Top Tokens
                </label>
                <select
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                  className="w-full text-xs bg-secondary border border-border rounded px-2.5 py-2 text-foreground focus:ring-1 focus:ring-primary"
                >
                  <option value={5}>Top 5 tokens</option>
                  <option value={10}>Top 10 tokens</option>
                  <option value={15}>Top 15 tokens</option>
                  <option value={20}>Top 20 tokens</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full mt-2 py-2.5 px-4 bg-primary text-primary-foreground text-sm font-semibold rounded-md shadow-sm hover:bg-primary/90 disabled:opacity-50 flex items-center justify-center space-x-2 transition-colors"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing Attributions...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>Analyze Email &amp; Explain Verdict</span>
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="p-4 rounded-md bg-destructive/10 border border-destructive/20 text-destructive text-sm flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right Column: Prediction & Explanations (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Verdict Banner */}
          {prediction ? (
            <div
              className={`p-5 rounded-lg border shadow-sm space-y-3 ${
                prediction.label_name === "spam"
                  ? "bg-rose-50/70 dark:bg-rose-950/20 border-rose-200 dark:border-rose-900/60"
                  : "bg-emerald-50/70 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-900/60"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {prediction.label_name === "spam" ? (
                    <ShieldAlert className="w-6 h-6 text-rose-600 dark:text-rose-400" />
                  ) : (
                    <ShieldCheck className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
                  )}
                  <span className="text-lg font-bold uppercase tracking-wide text-foreground">
                    Verdict: {prediction.label_name}
                  </span>
                </div>
                <span className="text-xs font-mono font-medium text-muted-foreground">
                  {prediction.model}
                </span>
              </div>

              {/* Probability Meter */}
              <div>
                <div className="flex justify-between text-xs font-medium text-muted-foreground mb-1.5">
                  <span>Spam Probability: <strong className="text-foreground">{formatPercent(prediction.spam_probability)}</strong></span>
                  <span>Confidence: <strong className="text-foreground">{formatPercent(prediction.confidence)}</strong></span>
                </div>
                <div className="w-full bg-secondary rounded-full h-3 overflow-hidden border border-border">
                  <div
                    className={`h-full transition-all duration-500 rounded-full ${
                      prediction.label_name === "spam" ? "bg-rose-500" : "bg-emerald-500"
                    }`}
                    style={{ width: `${Math.max(prediction.spam_probability * 100, 3)}%` }}
                  />
                </div>
              </div>

              {explanation && (
                <div className="text-xs text-muted-foreground pt-1 flex justify-between">
                  <span>Decision Score / Logit: <code className="font-mono text-foreground font-semibold">{formatDec(explanation.score, 3)}</code></span>
                  <span>Base Intercept: <code className="font-mono text-foreground">{formatDec(explanation.base_value, 3)}</code></span>
                </div>
              )}
            </div>
          ) : (
            <div className="p-8 rounded-lg bg-card border border-border border-dashed text-center text-muted-foreground space-y-2">
              <Sparkles className="w-8 h-8 mx-auto text-muted-foreground/60" />
              <p className="text-sm font-medium">No email analyzed yet</p>
              <p className="text-xs">
                Select a sample email or enter your own text, then click &ldquo;Analyze Email&rdquo; to see instant classification and token attributions.
              </p>
            </div>
          )}

          {/* Top Contributing Tokens Chart */}
          {explanation && barChartData.length > 0 && (
            <div className="p-5 rounded-lg bg-card border border-border shadow-sm space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-foreground">Top Feature Contributions</h3>
                <span className="text-xs text-muted-foreground font-mono">
                  {explanation.method}
                </span>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    layout="vertical"
                    data={barChartData}
                    margin={{ top: 5, right: 20, left: 35, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                    <XAxis type="number" tick={{ fill: "var(--muted-foreground)", fontSize: 11 }} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      tick={{ fill: "var(--foreground)", fontSize: 11 }}
                      width={65}
                    />
                    <Tooltip
                      formatter={(val: any) => [val, "Weight"]}
                      contentStyle={{
                        backgroundColor: "var(--card)",
                        borderColor: "var(--border)",
                        borderRadius: "0.5rem",
                        fontSize: "12px",
                      }}
                    />
                    <ReferenceLine x={0} stroke="var(--border)" />
                    <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
                      {barChartData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.weight > 0 ? "#ef4444" : "#10b981"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="flex items-center justify-center space-x-6 text-xs text-muted-foreground pt-1 border-t border-border">
                <div className="flex items-center space-x-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-rose-500 inline-block" />
                  <span>(+) Pushes toward Spam</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block" />
                  <span>(-) Pushes toward Ham</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Full Highlighted Email View */}
      {explanation && (
        <div className="p-6 rounded-lg bg-card border border-border shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-border pb-3">
            <div>
              <h3 className="text-base font-semibold text-foreground">In-Context Token Highlights</h3>
              <p className="text-xs text-muted-foreground">
                Words highlighted in red with <strong className="text-rose-600 dark:text-rose-400">(+)</strong> signal spam; words in green with <strong className="text-emerald-600 dark:text-emerald-400">(-)</strong> signal legitimate ham. Underlines assist color-blind readers.
              </p>
            </div>
            <div className="text-xs text-muted-foreground font-mono bg-secondary px-2.5 py-1 rounded">
              Method: {explanation.method}
            </div>
          </div>

          {renderHighlightedText()}
        </div>
      )}
    </div>
  );
}
