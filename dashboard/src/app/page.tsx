"use client";

import { useEffect, useState, useCallback } from "react";
import { TrendingUp, TrendingDown, DollarSign, Activity, BarChart3, Zap } from "lucide-react";
import Link from "next/link";
import { PriceChart } from "@/components/PriceChart";
import { AutomationControl } from "@/components/AutomationControl";
import { FearGreedGauge } from "@/components/FearGreedGauge";
import { useRealtimeSubscription } from "@/hooks/useRealtimeSubscription";

interface Signal {
  symbol: string;
  price: number;
  signal_score: number;
  action: string;
  position_size: number;
  reasons: string[];
}

interface SignalsData {
  timestamp: string;
  market_context: {
    fear_greed: number;
    fear_greed_label: string;
    vix: number;
    vix_regime: string;
  };
  signals: Signal[];
}

interface DayStatus {
  portfolio_value: number;
  cash: number;
  day_pnl: number;
  day_pnl_pct: number;
  total_trades: number;
  winners: number;
  losers: number;
  open_positions: number;
  positions: Record<string, Position>;
}

interface Position {
  symbol: string;
  direction: string;
  shares: number;
  entry_price: number;
  current_price?: number;
  pnl?: number;
  pnl_pct?: number;
}

interface MarketStatus {
  active_markets: string[];
  global_regime: string;
}

export default function Dashboard() {
  const [status, setStatus] = useState<DayStatus | null>(null);
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null);
  const [signals, setSignals] = useState<SignalsData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const [statusRes, marketRes, signalsRes] = await Promise.all([
        fetch("/api/status"),
        fetch("/api/markets"),
        fetch("/api/signals"),
      ]);

      if (statusRes.ok) {
        const data = await statusRes.json();
        setStatus(data);
      }
      if (marketRes.ok) {
        const data = await marketRes.json();
        setMarketStatus(data);
      }
      if (signalsRes.ok) {
        const data = await signalsRes.json();
        setSignals(data);
      }
    } catch (error) {
      console.error("Failed to fetch data:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    // Fallback polling (2 min) for when Supabase realtime isn't configured
    const interval = setInterval(fetchData, 120000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Real-time: refetch when positions, signals, or market snapshots change
  useRealtimeSubscription([
    { table: "portfolio_state", onchange: fetchData },
    { table: "positions", onchange: fetchData },
    { table: "signals", onchange: fetchData },
    { table: "market_snapshots", onchange: fetchData },
  ]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  const winRate = status && status.total_trades > 0
    ? ((status.winners / status.total_trades) * 100).toFixed(1)
    : "0";

  // Get top picks from signals
  const topPicks = signals?.signals
    .filter(s => s.action === "STRONG_BUY" || (s.action === "BUY" && s.signal_score > 0.3))
    .sort((a, b) => b.signal_score - a.signal_score)
    .slice(0, 3) || [];

  return (
    <div className="space-y-6">
      {/* Automation Control */}
      <AutomationControl />

      {/* Fear & Greed + Top Picks Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Fear & Greed */}
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-2 text-center">Market Sentiment</h2>
          {signals?.market_context ? (
            <FearGreedGauge 
              value={signals.market_context.fear_greed} 
              label={signals.market_context.fear_greed_label}
            />
          ) : (
            <div className="flex items-center justify-center h-40 text-zinc-500">
              Loading...
            </div>
          )}
        </div>

        {/* Top Picks */}
        <div className="lg:col-span-2 bg-gradient-to-r from-emerald-900/20 to-zinc-900 rounded-xl p-6 border border-emerald-800/30">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-500" />
              🎯 Top Picks
            </h2>
            <Link href="/signals" className="text-sm text-emerald-500 hover:text-emerald-400">
              View all →
            </Link>
          </div>
          
          {topPicks.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {topPicks.map((signal, idx) => (
                <div key={signal.symbol} className="bg-zinc-800/80 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-lg font-bold text-zinc-500">#{idx + 1}</span>
                      <span className="font-bold">{signal.symbol}</span>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      signal.action === "STRONG_BUY" 
                        ? "bg-emerald-600 text-white" 
                        : "bg-emerald-600/50 text-emerald-300"
                    }`}>
                      {signal.action.replace("_", " ")}
                    </span>
                  </div>
                  <div className="text-xl font-semibold">${signal.price.toFixed(2)}</div>
                  <div className="text-sm text-zinc-400 mt-1">
                    {(signal.signal_score * 100).toFixed(0)}% confidence • {signal.position_size} shares
                  </div>
                  <div className="text-xs text-zinc-500 mt-2 line-clamp-1">
                    {signal.reasons[0]}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-zinc-500 text-center py-8">
              No strong signals right now
            </div>
          )}
        </div>
      </div>

      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Portfolio Value"
          value={`$${status?.portfolio_value?.toLocaleString() ?? "100,000"}`}
          icon={DollarSign}
          trend={status?.day_pnl ?? 0}
          trendLabel={`${status?.day_pnl_pct?.toFixed(2) ?? "0.00"}% today`}
        />
        <StatCard
          title="Day P&L"
          value={`$${status?.day_pnl?.toLocaleString() ?? "0"}`}
          icon={status?.day_pnl && status.day_pnl >= 0 ? TrendingUp : TrendingDown}
          trend={status?.day_pnl ?? 0}
          positive={status?.day_pnl ? status.day_pnl >= 0 : true}
        />
        <StatCard
          title="Win Rate"
          value={`${winRate}%`}
          icon={Activity}
          subtitle={`${status?.winners ?? 0}W / ${status?.losers ?? 0}L`}
        />
        <StatCard
          title="Open Positions"
          value={String(status?.open_positions ?? 0)}
          icon={BarChart3}
          subtitle={`of 5 max`}
        />
      </div>

      {/* Market Status */}
      <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
          Market Status
        </h2>
        <div className="flex flex-wrap gap-4">
          {marketStatus?.active_markets?.length ? (
            <>
              <span className="text-zinc-400">Open now:</span>
              {marketStatus.active_markets.map((market) => (
                <span
                  key={market}
                  className="px-3 py-1 bg-emerald-600/20 text-emerald-400 rounded-full text-sm"
                >
                  {market}
                </span>
              ))}
            </>
          ) : (
            <span className="text-zinc-400">All markets closed</span>
          )}
          {marketStatus?.global_regime && (
            <span className="ml-auto text-zinc-400">
              Global: <span className="text-white">{marketStatus.global_regime}</span>
            </span>
          )}
        </div>
      </div>

      {/* Chart and Positions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-4">SPY Chart</h2>
          <div className="h-80">
            <PriceChart symbol="SPY" />
          </div>
        </div>

        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-4">Open Positions</h2>
          {status?.positions && Object.keys(status.positions).length > 0 ? (
            <div className="space-y-3">
              {Object.entries(status.positions).map(([sym, pos]) => (
                <PositionCard key={sym} position={{ ...pos, symbol: sym }} />
              ))}
            </div>
          ) : (
            <p className="text-zinc-500 text-center py-8">No open positions</p>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  icon: Icon,
  trend,
  trendLabel,
  subtitle,
  positive,
}: {
  title: string;
  value: string;
  icon: React.ComponentType<{ className?: string }>;
  trend?: number;
  trendLabel?: string;
  subtitle?: string;
  positive?: boolean;
}) {
  const isPositive = positive ?? (trend ? trend >= 0 : true);

  return (
    <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
      <div className="flex items-center justify-between mb-2">
        <span className="text-zinc-400 text-sm">{title}</span>
        <Icon className={`w-5 h-5 ${isPositive ? "text-emerald-500" : "text-red-500"}`} />
      </div>
      <div className="text-2xl font-bold">{value}</div>
      {(trendLabel || subtitle) && (
        <div className={`text-sm mt-1 ${isPositive ? "text-emerald-500" : "text-red-500"}`}>
          {trendLabel || subtitle}
        </div>
      )}
    </div>
  );
}

function PositionCard({ position }: { position: Position }) {
  const isLong = position.direction === "LONG";
  const isProfitable = (position.pnl ?? 0) >= 0;

  return (
    <div className="bg-zinc-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold">{position.symbol}</span>
        <span
          className={`text-xs px-2 py-1 rounded ${
            isLong ? "bg-emerald-600/20 text-emerald-400" : "bg-red-600/20 text-red-400"
          }`}
        >
          {position.direction}
        </span>
      </div>
      <div className="text-sm text-zinc-400">
        {position.shares} shares @ ${position.entry_price?.toFixed(2)}
      </div>
      {position.pnl !== undefined && (
        <div className={`text-sm mt-1 ${isProfitable ? "text-emerald-500" : "text-red-500"}`}>
          {isProfitable ? "+" : ""}${position.pnl?.toFixed(2)} ({position.pnl_pct?.toFixed(2)}%)
        </div>
      )}
    </div>
  );
}
