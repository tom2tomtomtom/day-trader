"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Cpu,
  AlertTriangle,
  Activity,
  Target,
  BarChart3,
  Clock,
  Layers,
  CheckCircle,
  XCircle,
  TrendingUp,
} from "lucide-react";
import { TimeAgo } from "@/components/TimeAgo";
import { useRealtimeSubscription } from "@/hooks/useRealtimeSubscription";

interface MLModel {
  version: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  training_samples: number;
  feature_importance: Record<string, number> | null;
  trained_at: string;
}

interface ModelHistoryEntry {
  version: number;
  accuracy: number;
  f1: number;
  training_samples: number;
  trained_at: string;
}

interface Prediction {
  symbol: string;
  actual_pnl_pct: number;
  profitable: boolean;
  exit_reason: string;
}

interface MLData {
  model: MLModel | null;
  history: ModelHistoryEntry[];
  recent_predictions: Prediction[];
  error?: string;
}

function MetricCard({
  label,
  value,
  suffix,
  positive,
}: {
  label: string;
  value: string;
  suffix?: string;
  positive?: boolean | null;
}) {
  return (
    <div className="bg-black-deep rounded-lg p-3">
      <div className="text-xs text-white-dim mb-1">{label}</div>
      <div
        className={`text-lg font-bold ${
          positive === true
            ? "text-orange-accent"
            : positive === false
              ? "text-red-hot"
              : "text-white-full"
        }`}
      >
        {value}
        {suffix && <span className="text-xs text-white-dim ml-1">{suffix}</span>}
      </div>
    </div>
  );
}

export default function MLPage() {
  const [data, setData] = useState<MLData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);

  const fetchML = useCallback(async () => {
    try {
      const res = await fetch("/api/ml");
      if (res.ok) {
        const json = await res.json();
        setData(json);
        setFetchedAt(new Date().toISOString());
      } else {
        const err = await res.json();
        setError(err.error || "Failed to fetch ML data");
      }
    } catch {
      setError("Failed to connect to ML API");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchML();
  }, [fetchML]);

  // Real-time: refetch when ML models or trades change
  useRealtimeSubscription([
    { table: "ml_models", onchange: fetchML },
    { table: "trades", onchange: fetchML },
  ]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-hot"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <PageHeader fetchedAt={fetchedAt} />
        <EmptyState message={error} />
      </div>
    );
  }

  if (!data?.model) {
    return (
      <div className="space-y-6">
        <PageHeader fetchedAt={fetchedAt} />
        <EmptyState message="No ML models trained yet" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader fetchedAt={fetchedAt} />
      <ActiveModelSection model={data.model} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FeatureImportanceChart features={data.model.feature_importance} />
        <SignalQualityDistribution predictions={data.recent_predictions} />
      </div>
      <PredictionAccuracy predictions={data.recent_predictions} />
      <ModelHistoryTable history={data.history} />
    </div>
  );
}

function PageHeader({ fetchedAt }: { fetchedAt: string | null }) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Cpu className="w-7 h-7 text-orange-accent" />
          ML Performance
        </h1>
        <p className="text-white-muted text-sm">
          Machine learning model metrics, feature importance, and prediction
          accuracy
        </p>
      </div>
      <TimeAgo timestamp={fetchedAt} staleAfterMs={3600000} />
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="bg-black-card rounded-xl p-8 border border-border-subtle text-center">
      <Cpu className="w-12 h-12 text-orange-accent/50 mx-auto mb-4" />
      <h2 className="text-xl font-bold mb-2">ML Models Training</h2>
      <p className="text-white-muted mb-4 max-w-md mx-auto">{message}</p>
      <p className="text-white-dim text-sm mb-4">Models auto-train nightly once you have 30+ trades. You can also trigger training manually:</p>
      <code className="bg-black-deep px-4 py-2 rounded text-sm">
        python3 -m core.orchestrator --train-ml
      </code>
    </div>
  );
}

function ActiveModelSection({ model }: { model: MLModel }) {
  const trainedDate = new Date(model.trained_at).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="space-y-4">
      {/* Model header */}
      <div className="bg-gradient-to-r from-red-hot/20 to-black-card rounded-xl p-6 border border-red-hot/30">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold flex items-center gap-2">
              <span className="w-2 h-2 bg-red-hot rounded-full animate-pulse"></span>
              Signal Quality Model v{model.version}
            </h2>
            <p className="text-sm text-white-muted flex items-center gap-1 mt-1">
              <Clock className="w-3 h-3" />
              Trained: {trainedDate}
            </p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-orange-accent">
              {(model.accuracy * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-white-muted">Accuracy</div>
          </div>
        </div>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard
          label="Accuracy"
          value={`${(model.accuracy * 100).toFixed(1)}%`}
          positive={model.accuracy > 0.6 ? true : model.accuracy > 0.5 ? null : false}
        />
        <MetricCard
          label="F1 Score"
          value={model.f1.toFixed(3)}
          positive={model.f1 > 0.6 ? true : model.f1 > 0.5 ? null : false}
        />
        <MetricCard
          label="Precision"
          value={`${(model.precision * 100).toFixed(1)}%`}
          positive={model.precision > 0.6 ? true : model.precision > 0.5 ? null : false}
        />
        <MetricCard
          label="Recall"
          value={`${(model.recall * 100).toFixed(1)}%`}
          positive={model.recall > 0.6 ? true : model.recall > 0.5 ? null : false}
        />
        <MetricCard
          label="Training Samples"
          value={model.training_samples.toLocaleString()}
        />
      </div>
    </div>
  );
}

function FeatureImportanceChart({
  features,
}: {
  features: Record<string, number> | null;
}) {
  if (!features || Object.keys(features).length === 0) {
    return (
      <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <Layers className="w-4 h-4 text-orange-accent" />
          Feature Importance
        </h3>
        <p className="text-white-dim text-center py-8 text-sm">
          Feature importance is calculated after model training. It shows which indicators drive predictions most.
        </p>
      </div>
    );
  }

  // Sort and take top 15
  const sorted = Object.entries(features)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 15);

  const maxValue = sorted[0]?.[1] || 1;

  return (
    <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
      <h3 className="font-semibold mb-4 flex items-center gap-2">
        <Layers className="w-4 h-4 text-orange-accent" />
        Feature Importance (Top 15)
      </h3>
      <div className="space-y-2">
        {sorted.map(([name, value]) => {
          const pct = (value / maxValue) * 100;
          return (
            <div key={name} className="flex items-center gap-3">
              <div className="w-36 text-xs text-white-muted truncate text-right flex-shrink-0" title={name}>
                {formatFeatureName(name)}
              </div>
              <div className="flex-1 bg-black-deep rounded-full h-5 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-red-hot to-orange-accent rounded-full transition-all duration-500"
                  style={{ width: `${Math.max(2, pct)}%` }}
                />
              </div>
              <div className="w-14 text-xs text-white-dim text-right flex-shrink-0">
                {(value * 100).toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatFeatureName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/Rsi/g, "RSI")
    .replace(/Macd/g, "MACD")
    .replace(/Sma/g, "SMA")
    .replace(/Ema/g, "EMA")
    .replace(/Atr/g, "ATR")
    .replace(/Vix/g, "VIX")
    .replace(/Pct/g, "%");
}

function SignalQualityDistribution({
  predictions,
}: {
  predictions: Prediction[];
}) {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-orange-accent" />
          Signal Quality Distribution
        </h3>
        <p className="text-white-dim text-center py-8 text-sm">
          Signal quality distribution appears after the ML model has made predictions on live signals.
        </p>
      </div>
    );
  }

  const profitable = predictions.filter((p) => p.profitable).length;
  const unprofitable = predictions.length - profitable;
  const total = predictions.length;
  const winRate = total > 0 ? (profitable / total) * 100 : 0;

  // Group by exit reason
  const byReason: Record<string, { count: number; wins: number }> = {};
  for (const p of predictions) {
    const reason = p.exit_reason || "unknown";
    if (!byReason[reason]) {
      byReason[reason] = { count: 0, wins: 0 };
    }
    byReason[reason].count++;
    if (p.profitable) byReason[reason].wins++;
  }

  const sortedReasons = Object.entries(byReason).sort(
    ([, a], [, b]) => b.count - a.count
  );

  return (
    <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
      <h3 className="font-semibold mb-4 flex items-center gap-2">
        <BarChart3 className="w-4 h-4 text-orange-accent" />
        Signal Quality Distribution
      </h3>

      {/* Win/Loss bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between text-sm mb-2">
          <span className="text-orange-accent flex items-center gap-1">
            <CheckCircle className="w-3 h-3" /> {profitable} Profitable
          </span>
          <span className="text-red-hot flex items-center gap-1">
            {unprofitable} Unprofitable <XCircle className="w-3 h-3" />
          </span>
        </div>
        <div className="flex h-4 rounded-full overflow-hidden bg-black-deep">
          {profitable > 0 && (
            <div
              className="bg-red-hot transition-all duration-500"
              style={{ width: `${winRate}%` }}
            />
          )}
          {unprofitable > 0 && (
            <div
              className="bg-red-hot transition-all duration-500"
              style={{ width: `${100 - winRate}%` }}
            />
          )}
        </div>
        <div className="text-center text-sm text-white-muted mt-2">
          {winRate.toFixed(1)}% win rate across {total} recent trades
        </div>
      </div>

      {/* By exit reason */}
      <div className="space-y-3">
        <div className="text-xs text-white-dim font-medium uppercase tracking-wider">
          By Exit Reason
        </div>
        {sortedReasons.map(([reason, stats]) => (
          <div key={reason} className="flex items-center justify-between text-sm">
            <span className="text-white-muted">
              <span
                className={`inline-block text-xs px-2 py-0.5 rounded mr-2 ${
                  reason === "take_profit"
                    ? "bg-red-hot/10 text-orange-accent"
                    : reason === "stop_loss"
                      ? "bg-red-hot/10 text-red-hot"
                      : "bg-black-card text-white-muted"
                }`}
              >
                {reason.replace(/_/g, " ")}
              </span>
            </span>
            <span className="text-white-muted">
              {stats.wins}/{stats.count}{" "}
              <span className="text-white-dim">
                ({((stats.wins / stats.count) * 100).toFixed(0)}%)
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function PredictionAccuracy({
  predictions,
}: {
  predictions: Prediction[];
}) {
  if (!predictions || predictions.length < 2) {
    return null;
  }

  // Show a rolling accuracy tracker across the recent predictions
  // Display as a streak of win/loss indicators
  const recent = predictions.slice(0, 30);

  // Compute rolling accuracy (window of 5)
  const windowSize = Math.min(5, recent.length);
  const rollingAccuracies: number[] = [];
  for (let i = 0; i <= recent.length - windowSize; i++) {
    const window = recent.slice(i, i + windowSize);
    const wins = window.filter((p) => p.profitable).length;
    rollingAccuracies.push(wins / windowSize);
  }

  // Avg PnL for profitable vs unprofitable
  const profitableTrades = recent.filter((p) => p.profitable);
  const unprofitableTrades = recent.filter((p) => !p.profitable);
  const avgWinPnl =
    profitableTrades.length > 0
      ? profitableTrades.reduce((s, p) => s + p.actual_pnl_pct, 0) /
        profitableTrades.length
      : 0;
  const avgLossPnl =
    unprofitableTrades.length > 0
      ? unprofitableTrades.reduce((s, p) => s + p.actual_pnl_pct, 0) /
        unprofitableTrades.length
      : 0;

  return (
    <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
      <h3 className="font-semibold mb-4 flex items-center gap-2">
        <Target className="w-4 h-4 text-orange-accent" />
        Prediction Accuracy (Recent {recent.length} Trades)
      </h3>

      {/* Win/Loss streak visualization */}
      <div className="mb-6">
        <div className="text-xs text-white-dim mb-2">Trade Outcomes (newest first)</div>
        <div className="flex gap-1 flex-wrap">
          {recent.map((p, i) => (
            <div
              key={i}
              className={`w-7 h-7 rounded flex items-center justify-center text-xs font-bold ${
                p.profitable
                  ? "bg-red-hot/20 text-orange-accent border border-red-hot/30"
                  : "bg-red-hot/20 text-red-hot border border-red-hot/30"
              }`}
              title={`${p.symbol}: ${p.actual_pnl_pct >= 0 ? "+" : ""}${p.actual_pnl_pct.toFixed(2)}% (${p.exit_reason})`}
            >
              {p.profitable ? "W" : "L"}
            </div>
          ))}
        </div>
      </div>

      {/* Rolling accuracy sparkline */}
      {rollingAccuracies.length > 1 && (
        <div className="mb-6">
          <div className="text-xs text-white-dim mb-2">
            Rolling Accuracy (window of {windowSize})
          </div>
          <div className="h-16 flex items-end gap-px">
            {rollingAccuracies.map((acc, i) => {
              const height = acc * 100;
              return (
                <div
                  key={i}
                  className={`flex-1 rounded-t transition-all ${
                    acc >= 0.6
                      ? "bg-red-hot/50"
                      : acc >= 0.4
                        ? "bg-yellow-500/50"
                        : "bg-red-hot/50"
                  }`}
                  style={{ height: `${Math.max(4, height)}%` }}
                  title={`${(acc * 100).toFixed(0)}% accuracy`}
                />
              );
            })}
          </div>
          <div className="flex justify-between text-xs text-white-dim mt-1">
            <span>Oldest</span>
            <span>Newest</span>
          </div>
        </div>
      )}

      {/* Avg returns */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Avg Win"
          value={`+${avgWinPnl.toFixed(2)}%`}
          positive={true}
        />
        <MetricCard
          label="Avg Loss"
          value={`${avgLossPnl.toFixed(2)}%`}
          positive={false}
        />
      </div>
    </div>
  );
}

function ModelHistoryTable({
  history,
}: {
  history: ModelHistoryEntry[];
}) {
  if (!history || history.length === 0) {
    return null;
  }

  return (
    <div className="bg-black-card rounded-xl border border-border-subtle overflow-hidden">
      <div className="p-4 border-b border-border-subtle">
        <h3 className="font-semibold flex items-center gap-2">
          <Activity className="w-4 h-4 text-orange-accent" />
          Model History ({history.length} versions)
        </h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-black-deep/50">
            <tr className="text-white-muted">
              <th className="text-left p-3">Version</th>
              <th className="text-right p-3">Accuracy</th>
              <th className="text-right p-3">F1 Score</th>
              <th className="text-right p-3">Training Samples</th>
              <th className="text-left p-3">Trained At</th>
              <th className="text-right p-3">Trend</th>
            </tr>
          </thead>
          <tbody>
            {history.map((h, i) => {
              const prevAccuracy = history[i + 1]?.accuracy;
              const accDiff =
                prevAccuracy !== undefined ? h.accuracy - prevAccuracy : null;

              return (
                <tr
                  key={h.version}
                  className={`border-b border-border-subtle/30 hover:bg-black-deep/20 ${
                    i === 0 ? "bg-red-hot/5" : ""
                  }`}
                >
                  <td className="p-3">
                    <span className="font-bold">v{h.version}</span>
                    {i === 0 && (
                      <span className="ml-2 text-xs px-2 py-0.5 rounded bg-red-hot/20 text-orange-accent">
                        active
                      </span>
                    )}
                  </td>
                  <td
                    className={`p-3 text-right font-medium ${
                      h.accuracy > 0.6
                        ? "text-orange-accent"
                        : h.accuracy > 0.5
                          ? "text-white-full"
                          : "text-red-hot"
                    }`}
                  >
                    {(h.accuracy * 100).toFixed(1)}%
                  </td>
                  <td className="p-3 text-right">{h.f1.toFixed(3)}</td>
                  <td className="p-3 text-right text-white-muted">
                    {h.training_samples.toLocaleString()}
                  </td>
                  <td className="p-3 text-white-muted">
                    {new Date(h.trained_at).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    })}
                  </td>
                  <td className="p-3 text-right">
                    {accDiff !== null ? (
                      <span
                        className={`flex items-center justify-end gap-1 text-xs ${
                          accDiff > 0
                            ? "text-orange-accent"
                            : accDiff < 0
                              ? "text-red-hot"
                              : "text-white-dim"
                        }`}
                      >
                        {accDiff > 0 ? (
                          <TrendingUp className="w-3 h-3" />
                        ) : accDiff < 0 ? (
                          <TrendingUp className="w-3 h-3 rotate-180" />
                        ) : null}
                        {accDiff > 0 ? "+" : ""}
                        {(accDiff * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-xs text-white-dim">--</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
