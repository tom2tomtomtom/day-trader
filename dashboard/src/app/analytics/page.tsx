"use client";

import { useEffect, useState, useCallback } from "react";
import {
  BarChart3,
  AlertTriangle,
  TrendingDown,
  Layers,
  Target,
  Activity,
  Grid3X3,
} from "lucide-react";
import { TimeAgo } from "@/components/TimeAgo";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  Cell,
} from "recharts";

// --- Types ---

interface WinRateByRegime {
  regime: string;
  win_rate: number;
  total_trades: number;
  wins: number;
  losses: number;
}

interface PnlByExitReason {
  reason: string;
  total_pnl: number;
  count: number;
  avg_pnl: number;
}

interface FeatureImportanceEntry {
  feature: string;
  importance: number;
}

interface SignalAccuracyPoint {
  date: string;
  accuracy: number;
  total_signals: number;
  correct_signals: number;
}

interface DrawdownPoint {
  date: string;
  cumulative_pnl: number;
  drawdown: number;
  drawdown_pct: number;
}

interface PositionHeatMapEntry {
  symbol: string;
  trade_count: number;
  total_pnl: number;
  avg_pnl: number;
  win_rate: number;
}

interface AnalyticsData {
  win_rate_by_regime: WinRateByRegime[];
  pnl_by_exit_reason: PnlByExitReason[];
  feature_importance: FeatureImportanceEntry[];
  signal_accuracy_over_time: SignalAccuracyPoint[];
  drawdown_data: DrawdownPoint[];
  position_heat_map: PositionHeatMapEntry[];
  error?: string;
}

// --- Custom Tooltip Components ---

function RegimeTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: WinRateByRegime }>;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-black-deep border border-border-subtle rounded-lg p-3 text-sm">
      <div className="font-bold text-white-full mb-1">{d.regime}</div>
      <div className="text-orange-accent">Win Rate: {d.win_rate.toFixed(1)}%</div>
      <div className="text-white-muted">
        {d.wins}W / {d.losses}L ({d.total_trades} trades)
      </div>
    </div>
  );
}

function PnlTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: PnlByExitReason }>;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-black-deep border border-border-subtle rounded-lg p-3 text-sm">
      <div className="font-bold text-white-full mb-1">
        {formatExitReason(d.reason)}
      </div>
      <div className={d.total_pnl >= 0 ? "text-orange-accent" : "text-red-hot"}>
        Total P&L: ${d.total_pnl.toLocaleString()}
      </div>
      <div className="text-white-muted">
        {d.count} trades | Avg: ${d.avg_pnl.toFixed(2)}
      </div>
    </div>
  );
}

function AccuracyTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: SignalAccuracyPoint }>;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-black-deep border border-border-subtle rounded-lg p-3 text-sm">
      <div className="font-bold text-white-full mb-1">Week of {d.date}</div>
      <div className="text-orange-accent">Accuracy: {d.accuracy}%</div>
      <div className="text-white-muted">
        {d.correct_signals}/{d.total_signals} correct
      </div>
    </div>
  );
}

function FeatureTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: ReadonlyArray<{ payload: { displayName: string; importance: number } }>;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-black-deep border border-border-subtle rounded-lg p-3 text-sm">
      <div className="font-bold text-white-full mb-1">{d.displayName}</div>
      <div className="text-orange-accent">Importance: {d.importance.toFixed(2)}%</div>
    </div>
  );
}

function DrawdownTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: DrawdownPoint }>;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-black-deep border border-border-subtle rounded-lg p-3 text-sm">
      <div className="font-bold text-white-full mb-1">{d.date}</div>
      <div className="text-red-hot">Drawdown: {d.drawdown_pct.toFixed(2)}%</div>
      <div className="text-white-muted">
        ${d.drawdown.toLocaleString()} from peak
      </div>
      <div className="text-white-muted">
        Cumulative P&L: ${d.cumulative_pnl.toLocaleString()}
      </div>
    </div>
  );
}

// --- Helpers ---

function formatExitReason(reason: string): string {
  return reason
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
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

// --- Main Page ---

export default function AnalyticsPage() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);

  const fetchAnalytics = useCallback(async () => {
    try {
      const res = await fetch("/api/analytics");
      if (res.ok) {
        const json = await res.json();
        if (json.error && !json.win_rate_by_regime) {
          setError(json.error);
        } else {
          setData(json);
          setFetchedAt(new Date().toISOString());
        }
      } else {
        const err = await res.json();
        setError(err.error || "Failed to fetch analytics data");
      }
    } catch {
      setError("Failed to connect to analytics API");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-hot"></div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="space-y-6">
        <PageHeader fetchedAt={fetchedAt} />
        <EmptyState message={error} />
      </div>
    );
  }

  const hasAnyData =
    data &&
    (data.win_rate_by_regime.length > 0 ||
      data.pnl_by_exit_reason.length > 0 ||
      data.feature_importance.length > 0 ||
      data.signal_accuracy_over_time.length > 0 ||
      data.drawdown_data.length > 0 ||
      data.position_heat_map.length > 0);

  if (!hasAnyData) {
    return (
      <div className="space-y-6">
        <PageHeader fetchedAt={fetchedAt} />
        <EmptyState message="No analytics data available yet. Run some trades to populate the dashboard." />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader fetchedAt={fetchedAt} />

      {/* Row 1: Win Rate by Regime + P&L by Exit Reason */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <WinRateByRegimeChart data={data!.win_rate_by_regime} />
        <PnlByExitReasonChart data={data!.pnl_by_exit_reason} />
      </div>

      {/* Row 2: Feature Importance + Signal Accuracy */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FeatureImportanceChart data={data!.feature_importance} />
        <SignalAccuracyChart data={data!.signal_accuracy_over_time} />
      </div>

      {/* Row 3: Drawdown */}
      <DrawdownChart data={data!.drawdown_data} />

      {/* Row 4: Position Heat Map */}
      <PositionHeatMap data={data!.position_heat_map} />
    </div>
  );
}

// --- Sub-components ---

function PageHeader({ fetchedAt }: { fetchedAt: string | null }) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <BarChart3 className="w-7 h-7 text-orange-accent" />
          Advanced Analytics
        </h1>
        <p className="text-white-muted text-sm">
          Win rates, P&L breakdown, feature importance, signal accuracy, and
          drawdown analysis
        </p>
      </div>
      <TimeAgo timestamp={fetchedAt} staleAfterMs={3600000} />
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="bg-black-card rounded-xl p-8 border border-border-subtle text-center">
      <BarChart3 className="w-12 h-12 text-orange-accent/50 mx-auto mb-4" />
      <h2 className="text-xl font-bold mb-2">Analytics Building Up</h2>
      <p className="text-white-muted mb-4 max-w-md mx-auto">{message}</p>
      <p className="text-white-dim text-sm mb-4">Charts populate automatically as the system trades. You can also seed data with a backtest:</p>
      <code className="bg-black-deep px-4 py-2 rounded text-sm">
        python3 -m core.orchestrator --paper
      </code>
    </div>
  );
}

function WinRateByRegimeChart({ data }: { data: WinRateByRegime[] }) {
  if (data.length === 0) {
    return (
      <ChartCard
        title="Win Rate by Regime"
        icon={<BarChart3 className="w-4 h-4 text-orange-accent" />}
      >
        <p className="text-white-dim text-center py-8">
          Regime data appears after trades across different market conditions (trending, ranging, crisis).
        </p>
      </ChartCard>
    );
  }

  return (
    <ChartCard
      title="Win Rate by Regime"
      icon={<BarChart3 className="w-4 h-4 text-orange-accent" />}
    >
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="regime"
            tick={{ fill: "#a1a1aa", fontSize: 12 }}
            axisLine={{ stroke: "#3f3f46" }}
          />
          <YAxis
            tick={{ fill: "#a1a1aa", fontSize: 12 }}
            axisLine={{ stroke: "#3f3f46" }}
            domain={[0, 100]}
            tickFormatter={(v: number) => `${v}%`}
          />
          <Tooltip content={<RegimeTooltip />} />
          <Bar dataKey="win_rate" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={index}
                fill={
                  entry.win_rate >= 60
                    ? "#10b981"
                    : entry.win_rate >= 45
                      ? "#eab308"
                      : "#ef4444"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-2 mt-3">
        {data.map((d) => (
          <span
            key={d.regime}
            className="text-xs bg-black-deep px-2 py-1 rounded text-white-muted"
          >
            {d.regime}: {d.total_trades} trades
          </span>
        ))}
      </div>
    </ChartCard>
  );
}

function PnlByExitReasonChart({ data }: { data: PnlByExitReason[] }) {
  if (data.length === 0) {
    return (
      <ChartCard
        title="P&L by Exit Reason"
        icon={<Activity className="w-4 h-4 text-orange-accent" />}
      >
        <p className="text-white-dim text-center py-8">
          Exit reason breakdown appears after closed trades (stop-loss, take-profit, trailing stop).
        </p>
      </ChartCard>
    );
  }

  const chartData = data.map((d) => ({
    ...d,
    displayReason: formatExitReason(d.reason),
  }));

  return (
    <ChartCard
      title="P&L by Exit Reason"
      icon={<Activity className="w-4 h-4 text-orange-accent" />}
    >
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={chartData}
          margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="displayReason"
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            axisLine={{ stroke: "#3f3f46" }}
          />
          <YAxis
            tick={{ fill: "#a1a1aa", fontSize: 12 }}
            axisLine={{ stroke: "#3f3f46" }}
            tickFormatter={(v: number) => `$${v}`}
          />
          <Tooltip content={<PnlTooltip />} />
          <Bar dataKey="total_pnl" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.total_pnl >= 0 ? "#10b981" : "#ef4444"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function FeatureImportanceChart({
  data,
}: {
  data: FeatureImportanceEntry[];
}) {
  if (data.length === 0) {
    return (
      <ChartCard
        title="Feature Importance"
        icon={<Layers className="w-4 h-4 text-orange-accent" />}
      >
        <p className="text-white-dim text-center py-8">
          No feature importance data. Train an ML model first.
        </p>
      </ChartCard>
    );
  }

  const chartData = data.map((d) => ({
    ...d,
    displayName: formatFeatureName(d.feature),
  }));

  return (
    <ChartCard
      title="Feature Importance (Top 20)"
      icon={<Layers className="w-4 h-4 text-orange-accent" />}
    >
      <ResponsiveContainer width="100%" height={Math.max(300, data.length * 28)}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 5, right: 20, bottom: 5, left: 120 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: "#a1a1aa", fontSize: 12 }}
            axisLine={{ stroke: "#3f3f46" }}
            tickFormatter={(v: number) => `${v}%`}
          />
          <YAxis
            type="category"
            dataKey="displayName"
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            axisLine={{ stroke: "#3f3f46" }}
            width={115}
          />
          <Tooltip content={<FeatureTooltip />} />
          <Bar dataKey="importance" fill="#10b981" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function SignalAccuracyChart({ data }: { data: SignalAccuracyPoint[] }) {
  if (data.length === 0) {
    return (
      <ChartCard
        title="Signal Accuracy Over Time"
        icon={<Target className="w-4 h-4 text-orange-accent" />}
      >
        <p className="text-white-dim text-center py-8">
          Signal accuracy tracking starts after the nightly evaluation cycle matches signals to trade outcomes.
        </p>
      </ChartCard>
    );
  }

  const avgAccuracy =
    data.reduce((sum, d) => sum + d.accuracy, 0) / data.length;

  return (
    <ChartCard
      title="Signal Accuracy Over Time"
      icon={<Target className="w-4 h-4 text-orange-accent" />}
      subtitle={`Average: ${avgAccuracy.toFixed(1)}%`}
    >
      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={data}
          margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            axisLine={{ stroke: "#3f3f46" }}
            tickFormatter={(d: string) => {
              const date = new Date(d);
              return `${date.getMonth() + 1}/${date.getDate()}`;
            }}
          />
          <YAxis
            tick={{ fill: "#a1a1aa", fontSize: 12 }}
            axisLine={{ stroke: "#3f3f46" }}
            domain={[0, 100]}
            tickFormatter={(v: number) => `${v}%`}
          />
          <Tooltip content={<AccuracyTooltip />} />
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke="#10b981"
            strokeWidth={2}
            dot={{ fill: "#10b981", r: 3 }}
            activeDot={{ r: 5, fill: "#34d399" }}
          />
          {/* Reference line at 50% */}
          <Line
            type="monotone"
            dataKey={() => 50}
            stroke="#3f3f46"
            strokeDasharray="5 5"
            strokeWidth={1}
            dot={false}
            activeDot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function DrawdownChart({ data }: { data: DrawdownPoint[] }) {
  if (data.length === 0) {
    return (
      <ChartCard
        title="Drawdown"
        icon={<TrendingDown className="w-4 h-4 text-red-hot" />}
      >
        <p className="text-white-dim text-center py-8">
          Drawdown chart builds as the equity curve develops over multiple trading sessions.
        </p>
      </ChartCard>
    );
  }

  const maxDrawdown = Math.min(...data.map((d) => d.drawdown_pct));
  const currentDrawdown = data[data.length - 1]?.drawdown_pct ?? 0;

  return (
    <ChartCard
      title="Drawdown Analysis"
      icon={<TrendingDown className="w-4 h-4 text-red-hot" />}
      subtitle={`Max: ${maxDrawdown.toFixed(2)}% | Current: ${currentDrawdown.toFixed(2)}%`}
    >
      <ResponsiveContainer width="100%" height={250}>
        <AreaChart
          data={data}
          margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            axisLine={{ stroke: "#3f3f46" }}
            tickFormatter={(d: string) => {
              const date = new Date(d);
              return `${date.getMonth() + 1}/${date.getDate()}`;
            }}
          />
          <YAxis
            tick={{ fill: "#a1a1aa", fontSize: 12 }}
            axisLine={{ stroke: "#3f3f46" }}
            tickFormatter={(v: number) => `${v}%`}
          />
          <Tooltip content={<DrawdownTooltip />} />
          <defs>
            <linearGradient id="drawdownGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <Area
            type="monotone"
            dataKey="drawdown_pct"
            stroke="#ef4444"
            strokeWidth={2}
            fill="url(#drawdownGrad)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function PositionHeatMap({ data }: { data: PositionHeatMapEntry[] }) {
  if (data.length === 0) {
    return null;
  }

  // Find max values for normalization
  const maxCount = Math.max(...data.map((d) => d.trade_count));
  const maxPnl = Math.max(...data.map((d) => Math.abs(d.total_pnl)));

  return (
    <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
      <h3 className="font-semibold mb-4 flex items-center gap-2">
        <Grid3X3 className="w-4 h-4 text-orange-accent" />
        Position Heat Map
        <span className="text-xs text-white-dim font-normal ml-2">
          Size = trade count, Color = P&L
        </span>
      </h3>
      <div className="flex flex-wrap gap-2">
        {data.map((entry) => {
          const sizeScale = 0.6 + (entry.trade_count / maxCount) * 0.4;
          const intensity =
            maxPnl > 0 ? Math.abs(entry.total_pnl) / maxPnl : 0;
          const isPositive = entry.total_pnl >= 0;

          const bgColor = isPositive
            ? `rgba(16, 185, 129, ${0.1 + intensity * 0.4})`
            : `rgba(239, 68, 68, ${0.1 + intensity * 0.4})`;
          const borderColor = isPositive
            ? `rgba(16, 185, 129, ${0.2 + intensity * 0.4})`
            : `rgba(239, 68, 68, ${0.2 + intensity * 0.4})`;

          return (
            <div
              key={entry.symbol}
              className="rounded-lg p-3 border transition-transform hover:scale-105 cursor-default"
              style={{
                backgroundColor: bgColor,
                borderColor: borderColor,
                minWidth: `${Math.max(80, 80 * sizeScale)}px`,
              }}
              title={`${entry.symbol}: ${entry.trade_count} trades, $${entry.total_pnl.toFixed(2)} P&L, ${entry.win_rate}% WR`}
            >
              <div className="font-bold text-sm">{entry.symbol}</div>
              <div className="text-xs text-white-muted">
                {entry.trade_count} trades
              </div>
              <div
                className={`text-xs font-medium ${
                  isPositive ? "text-orange-accent" : "text-red-hot"
                }`}
              >
                ${entry.total_pnl >= 0 ? "+" : ""}
                {entry.total_pnl.toFixed(0)}
              </div>
              <div className="text-xs text-white-dim">
                {entry.win_rate}% WR
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// --- Reusable Card ---

function ChartCard({
  title,
  icon,
  subtitle,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold flex items-center gap-2">
          {icon}
          {title}
        </h3>
        {subtitle && (
          <span className="text-xs text-white-dim">{subtitle}</span>
        )}
      </div>
      {children}
    </div>
  );
}
