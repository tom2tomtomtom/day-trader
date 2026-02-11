"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  DollarSign,
  BarChart3,
  Target,
  Shield,
  Zap,
  Clock,
} from "lucide-react";

interface BacktestMetrics {
  total_return_pct: number;
  annualized_return_pct: number;
  max_drawdown_pct: number;
  max_drawdown_duration_days: number;
  volatility_annualized: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  avg_win_pct: number;
  avg_loss_pct: number;
  largest_win_pct: number;
  largest_loss_pct: number;
  avg_hold_days: number;
  profit_factor: number;
  expectancy: number;
}

interface BacktestTrade {
  symbol: string;
  direction: string;
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  shares: number;
  pnl_dollars: number;
  return_pct: number;
  hold_days: number;
  exit_reason: string;
  entry_score: number;
}

interface EquityPoint {
  date: string;
  portfolio_value: number;
  drawdown: number;
  cumulative_return: number;
}

interface BacktestData {
  timestamp: string;
  strategy: string;
  symbols: string[];
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_value: number;
  metrics: BacktestMetrics;
  trades: BacktestTrade[];
  equity_curve: EquityPoint[];
  parameters: Record<string, number>;
}

interface PaperData {
  timestamp: string;
  portfolio: {
    initial_capital: number;
    current_value: number;
    cash: number;
    total_return_pct: number;
    daily_pnl: number;
    max_drawdown_pct: number;
  };
  stats: {
    total_trades: number;
    win_rate: number;
    avg_win_pct: number;
    avg_loss_pct: number;
    profit_factor: number;
  };
  positions: Array<{
    symbol: string;
    direction: string;
    entry_price: number;
    current_price: number;
    shares: number;
    unrealized_pnl: number;
    unrealized_pnl_pct: number;
    entry_score: number;
  }>;
  recent_trades: BacktestTrade[];
  equity_curve: Array<{ date: string; value: number }>;
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

export default function PerformancePage() {
  const [backtest, setBacktest] = useState<BacktestData | null>(null);
  const [paper, setPaper] = useState<PaperData | null>(null);
  const [btError, setBtError] = useState<string | null>(null);
  const [paperError, setPaperError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"backtest" | "paper">("backtest");

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const btRes = await fetch("/api/backtest");
        if (btRes.ok) setBacktest(await btRes.json());
        else setBtError((await btRes.json()).error);
      } catch {
        setBtError("Failed to fetch backtest data");
      }

      try {
        const ppRes = await fetch("/api/paper");
        if (ppRes.ok) setPaper(await ppRes.json());
        else setPaperError((await ppRes.json()).error);
      } catch {
        setPaperError("Failed to fetch paper trading data");
      }

      setLoading(false);
    };
    fetchAll();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-hot"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Activity className="w-7 h-7 text-orange-accent" />
            Performance & Backtesting
          </h1>
          <p className="text-white-muted text-sm">
            Backtest results and paper trading performance
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setTab("backtest")}
            className={`px-4 py-2 rounded-lg text-sm ${
              tab === "backtest"
                ? "bg-red-hot text-white"
                : "bg-black-deep text-white-muted hover:text-white"
            }`}
          >
            Backtest
          </button>
          <button
            onClick={() => setTab("paper")}
            className={`px-4 py-2 rounded-lg text-sm ${
              tab === "paper"
                ? "bg-red-hot text-white"
                : "bg-black-deep text-white-muted hover:text-white"
            }`}
          >
            Paper Trading
          </button>
        </div>
      </div>

      {/* Backtest Tab */}
      {tab === "backtest" && (
        <>
          {btError ? (
            <div className="bg-black-card rounded-xl p-8 border border-border-subtle text-center">
              <AlertTriangle className="w-12 h-12 text-orange-accent mx-auto mb-4" />
              <h2 className="text-xl font-bold mb-2">No Backtest Results</h2>
              <p className="text-white-muted mb-4">{btError}</p>
              <code className="bg-black-deep px-4 py-2 rounded text-sm">
                python3 -m core.backtester --symbol SPY --period 1y
              </code>
            </div>
          ) : backtest ? (
            <BacktestView data={backtest} />
          ) : null}
        </>
      )}

      {/* Paper Trading Tab */}
      {tab === "paper" && (
        <>
          {paperError ? (
            <div className="bg-black-card rounded-xl p-8 border border-border-subtle text-center">
              <AlertTriangle className="w-12 h-12 text-orange-accent mx-auto mb-4" />
              <h2 className="text-xl font-bold mb-2">No Paper Trading Data</h2>
              <p className="text-white-muted mb-4">{paperError}</p>
              <code className="bg-black-deep px-4 py-2 rounded text-sm">
                python3 -m core.paper_trader
              </code>
            </div>
          ) : paper ? (
            <PaperView data={paper} />
          ) : null}
        </>
      )}
    </div>
  );
}

function BacktestView({ data }: { data: BacktestData }) {
  const m = data.metrics;

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="bg-gradient-to-r from-red-hot/20 to-black-card rounded-xl p-6 border border-red-hot/30">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold">
              {data.symbols.join(", ")} Backtest
            </h2>
            <p className="text-sm text-white-muted">
              {data.start_date} to {data.end_date} | {data.strategy}
            </p>
          </div>
          <div className="text-right">
            <div className={`text-3xl font-bold ${m.total_return_pct >= 0 ? "text-orange-accent" : "text-red-hot"}`}>
              {m.total_return_pct >= 0 ? "+" : ""}
              {m.total_return_pct.toFixed(2)}%
            </div>
            <div className="text-sm text-white-muted">
              ${data.initial_capital.toLocaleString()} → ${data.final_value.toLocaleString()}
            </div>
          </div>
        </div>

        {/* Equity Curve (text-based sparkline) */}
        {data.equity_curve && data.equity_curve.length > 0 && (
          <div className="h-16 flex items-end gap-px">
            {data.equity_curve.map((pt, i) => {
              const min = Math.min(...data.equity_curve.map((p) => p.portfolio_value));
              const max = Math.max(...data.equity_curve.map((p) => p.portfolio_value));
              const range = max - min || 1;
              const height = ((pt.portfolio_value - min) / range) * 100;
              return (
                <div
                  key={i}
                  className={`flex-1 rounded-t ${
                    pt.portfolio_value >= data.initial_capital
                      ? "bg-red-hot/40"
                      : "bg-red-hot/40"
                  }`}
                  style={{ height: `${Math.max(2, height)}%` }}
                  title={`${pt.date}: $${pt.portfolio_value.toLocaleString()}`}
                />
              );
            })}
          </div>
        )}
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <MetricCard label="Annualized Return" value={`${m.annualized_return_pct >= 0 ? "+" : ""}${m.annualized_return_pct.toFixed(1)}%`} positive={m.annualized_return_pct > 0} />
        <MetricCard label="Max Drawdown" value={`${m.max_drawdown_pct.toFixed(1)}%`} positive={m.max_drawdown_pct < 10 ? true : m.max_drawdown_pct < 20 ? null : false} />
        <MetricCard label="Sharpe Ratio" value={m.sharpe_ratio.toFixed(2)} positive={m.sharpe_ratio > 1 ? true : m.sharpe_ratio > 0 ? null : false} />
        <MetricCard label="Sortino Ratio" value={m.sortino_ratio.toFixed(2)} positive={m.sortino_ratio > 1.5 ? true : m.sortino_ratio > 0 ? null : false} />
        <MetricCard label="Win Rate" value={`${(m.win_rate * 100).toFixed(1)}%`} positive={m.win_rate > 0.55 ? true : m.win_rate > 0.45 ? null : false} />
        <MetricCard label="Profit Factor" value={m.profit_factor.toFixed(2)} positive={m.profit_factor > 1.5 ? true : m.profit_factor > 1 ? null : false} />
      </div>

      {/* Detailed Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Metrics */}
        <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Shield className="w-4 h-4 text-red-hot" />
            Risk Metrics
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-white-muted">Max Drawdown</span>
              <span className="text-red-hot">{m.max_drawdown_pct.toFixed(2)}%</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-white-muted">DD Duration</span>
              <span>{m.max_drawdown_duration_days} days</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-white-muted">Volatility (ann.)</span>
              <span>{m.volatility_annualized.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-white-muted">Calmar Ratio</span>
              <span>{m.calmar_ratio.toFixed(2)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-white-muted">Largest Loss</span>
              <span className="text-red-hot">{m.largest_loss_pct.toFixed(2)}%</span>
            </div>
          </div>
        </div>

        {/* Trade Stats */}
        <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-orange-accent" />
            Trade Statistics
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-white-muted">Total Trades</span>
              <span>{m.total_trades}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-white-muted">Winners / Losers</span>
              <span>
                <span className="text-orange-accent">{m.winning_trades}</span> /{" "}
                <span className="text-red-hot">{m.losing_trades}</span>
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-white-muted">Avg Win / Loss</span>
              <span>
                <span className="text-orange-accent">+{m.avg_win_pct.toFixed(1)}%</span> /{" "}
                <span className="text-red-hot">{m.avg_loss_pct.toFixed(1)}%</span>
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-white-muted">Avg Hold</span>
              <span>{m.avg_hold_days.toFixed(1)} days</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-white-muted">Expectancy</span>
              <span className={m.expectancy >= 0 ? "text-orange-accent" : "text-red-hot"}>
                ${m.expectancy.toFixed(2)}/trade
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Trade Log */}
      {data.trades && data.trades.length > 0 && (
        <div className="bg-black-card rounded-xl border border-border-subtle overflow-hidden">
          <div className="p-4 border-b border-border-subtle">
            <h3 className="font-semibold">Trade Log ({data.trades.length} trades)</h3>
          </div>
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-black-deep/50 sticky top-0">
                <tr className="text-white-muted">
                  <th className="text-left p-3">Symbol</th>
                  <th className="text-left p-3">Dir</th>
                  <th className="text-left p-3">Entry</th>
                  <th className="text-left p-3">Exit</th>
                  <th className="text-right p-3">P&L</th>
                  <th className="text-right p-3">Return</th>
                  <th className="text-left p-3">Reason</th>
                  <th className="text-right p-3">Days</th>
                  <th className="text-right p-3">Score</th>
                </tr>
              </thead>
              <tbody>
                {data.trades.map((t, i) => (
                  <tr key={i} className="border-b border-border-subtle/30 hover:bg-black-deep/20">
                    <td className="p-3 font-bold">{t.symbol}</td>
                    <td className="p-3">
                      {t.direction === "long" ? (
                        <span className="text-orange-accent flex items-center gap-1">
                          <TrendingUp className="w-3 h-3" /> Long
                        </span>
                      ) : (
                        <span className="text-red-hot flex items-center gap-1">
                          <TrendingDown className="w-3 h-3" /> Short
                        </span>
                      )}
                    </td>
                    <td className="p-3">${t.entry_price.toFixed(2)}</td>
                    <td className="p-3">${t.exit_price.toFixed(2)}</td>
                    <td className={`p-3 text-right font-medium ${t.pnl_dollars >= 0 ? "text-orange-accent" : "text-red-hot"}`}>
                      ${t.pnl_dollars >= 0 ? "+" : ""}{t.pnl_dollars.toFixed(2)}
                    </td>
                    <td className={`p-3 text-right ${t.return_pct >= 0 ? "text-orange-accent" : "text-red-hot"}`}>
                      {t.return_pct >= 0 ? "+" : ""}{t.return_pct.toFixed(2)}%
                    </td>
                    <td className="p-3">
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        t.exit_reason === "take_profit" ? "bg-red-hot/10 text-orange-accent" :
                        t.exit_reason === "stop_loss" ? "bg-red-hot/10 text-red-hot" :
                        "bg-black-card text-white-muted"
                      }`}>
                        {t.exit_reason}
                      </span>
                    </td>
                    <td className="p-3 text-right text-white-muted">{t.hold_days}d</td>
                    <td className="p-3 text-right text-white-muted">{t.entry_score}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Parameters */}
      <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
        <h3 className="font-semibold mb-3 text-sm text-white-muted">Model Parameters</h3>
        <div className="flex flex-wrap gap-3">
          {Object.entries(data.parameters).map(([key, val]) => (
            <div key={key} className="bg-black-deep px-3 py-1.5 rounded text-xs">
              <span className="text-white-dim">{key}: </span>
              <span className="text-white-full">{typeof val === "number" ? (val < 1 && val > 0 ? `${(val * 100).toFixed(1)}%` : val) : val}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PaperView({ data }: { data: PaperData }) {
  return (
    <div className="space-y-6">
      {/* Portfolio Summary */}
      <div className="bg-gradient-to-r from-red-hot/20 to-black-card rounded-xl p-6 border border-red-hot/30">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold">Paper Portfolio</h2>
            <p className="text-sm text-white-muted">
              Updated: {data.timestamp && new Date(data.timestamp).toLocaleString()}
            </p>
          </div>
          <div className="text-right">
            <div className={`text-3xl font-bold ${data.portfolio.total_return_pct >= 0 ? "text-orange-accent" : "text-red-hot"}`}>
              ${data.portfolio.current_value.toLocaleString()}
            </div>
            <div className={`text-sm ${data.portfolio.total_return_pct >= 0 ? "text-orange-accent" : "text-red-hot"}`}>
              {data.portfolio.total_return_pct >= 0 ? "+" : ""}
              {data.portfolio.total_return_pct.toFixed(2)}%
            </div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="Cash" value={`$${data.portfolio.cash.toLocaleString()}`} />
        <MetricCard label="Daily P&L" value={`$${data.portfolio.daily_pnl >= 0 ? "+" : ""}${data.portfolio.daily_pnl.toFixed(0)}`} positive={data.portfolio.daily_pnl > 0 ? true : data.portfolio.daily_pnl < 0 ? false : null} />
        <MetricCard label="Win Rate" value={`${(data.stats.win_rate * 100).toFixed(0)}%`} positive={data.stats.win_rate > 0.5} />
        <MetricCard label="Trades" value={`${data.stats.total_trades}`} />
        <MetricCard label="Max DD" value={`${data.portfolio.max_drawdown_pct.toFixed(1)}%`} positive={data.portfolio.max_drawdown_pct < 5 ? true : data.portfolio.max_drawdown_pct < 15 ? null : false} />
      </div>

      {/* Open Positions */}
      {data.positions.length > 0 && (
        <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Target className="w-4 h-4 text-orange-accent" />
            Open Positions ({data.positions.length})
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {data.positions.map((pos) => (
              <div key={pos.symbol} className="bg-black-deep rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-bold">{pos.symbol}</span>
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      pos.direction === "long" ? "bg-red-hot/10 text-orange-accent" : "bg-red-hot/10 text-red-hot"
                    }`}>
                      {pos.direction.toUpperCase()}
                    </span>
                  </div>
                  <span className={`font-bold ${pos.unrealized_pnl >= 0 ? "text-orange-accent" : "text-red-hot"}`}>
                    {pos.unrealized_pnl_pct >= 0 ? "+" : ""}{pos.unrealized_pnl_pct.toFixed(1)}%
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs text-white-muted">
                  <div>Entry: ${pos.entry_price.toFixed(2)}</div>
                  <div>Current: ${pos.current_price.toFixed(2)}</div>
                  <div>P&L: ${pos.unrealized_pnl >= 0 ? "+" : ""}{pos.unrealized_pnl.toFixed(0)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Trades */}
      {data.recent_trades && data.recent_trades.length > 0 && (
        <div className="bg-black-card rounded-xl border border-border-subtle overflow-hidden">
          <div className="p-4 border-b border-border-subtle">
            <h3 className="font-semibold">Recent Trades</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-black-deep/50">
                <tr className="text-white-muted">
                  <th className="text-left p-3">Symbol</th>
                  <th className="text-left p-3">Dir</th>
                  <th className="text-left p-3">Entry</th>
                  <th className="text-left p-3">Exit</th>
                  <th className="text-right p-3">P&L</th>
                  <th className="text-left p-3">Reason</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_trades.slice(-10).reverse().map((t, i) => (
                  <tr key={i} className="border-b border-border-subtle/30">
                    <td className="p-3 font-bold">{t.symbol}</td>
                    <td className="p-3">
                      <span className={t.direction === "long" ? "text-orange-accent" : "text-red-hot"}>
                        {t.direction}
                      </span>
                    </td>
                    <td className="p-3">${t.entry_price.toFixed(2)}</td>
                    <td className="p-3">${t.exit_price.toFixed(2)}</td>
                    <td className={`p-3 text-right font-medium ${t.pnl_dollars >= 0 ? "text-orange-accent" : "text-red-hot"}`}>
                      ${t.pnl_dollars >= 0 ? "+" : ""}{t.pnl_dollars.toFixed(0)}
                    </td>
                    <td className="p-3">
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        t.exit_reason === "take_profit" ? "bg-red-hot/10 text-orange-accent" :
                        t.exit_reason === "stop_loss" ? "bg-red-hot/10 text-red-hot" :
                        "bg-black-card text-white-muted"
                      }`}>
                        {t.exit_reason}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
