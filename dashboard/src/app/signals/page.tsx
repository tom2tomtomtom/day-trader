"use client";

import { useEffect, useState, useCallback } from "react";
import { TrendingUp, TrendingDown, Minus, AlertTriangle, CheckCircle2, XCircle, Zap } from "lucide-react";
import { FearGreedGauge } from "@/components/FearGreedGauge";
import { TimeAgo } from "@/components/TimeAgo";
import { useTableSubscription } from "@/hooks/useRealtimeSubscription";

interface Signal {
  symbol: string;
  price: number;
  signal_score: number;
  action: string;
  position_size: number;
  reasons: string[];
  indicators: {
    bollinger: { position: number; width: number };
    rsi: { value: number; is_oversold: boolean; is_overbought: boolean };
    volume: { relative_volume: number; is_confirming: boolean };
    trend: { trend: string; strength: number };
  };
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

export default function SignalsPage() {
  const [data, setData] = useState<SignalsData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/signals");
      if (res.ok) {
        const signals = await res.json();
        setData(signals);
      }
    } catch (error) {
      console.error("Failed to fetch signals:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 120000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Real-time: refetch when signals change
  useTableSubscription("signals", fetchData);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-16">
        <Zap className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
        <h2 className="text-lg font-semibold mb-2">No Signal Data Yet</h2>
        <p className="text-zinc-500 mb-4 max-w-sm mx-auto">
          The scheduler generates signals every 5 minutes during market hours and every 15 minutes for crypto.
        </p>
        <p className="text-zinc-600 text-sm">Signals will appear here automatically once the next scan completes.</p>
      </div>
    );
  }

  // Sort signals by score, filter actionable ones
  const actionableSignals = data.signals
    .filter(s => s.action !== "HOLD" && Math.abs(s.signal_score) > 0.1)
    .sort((a, b) => Math.abs(b.signal_score) - Math.abs(a.signal_score));

  const strongBuys = actionableSignals.filter(s => s.action === "STRONG_BUY" || (s.action === "BUY" && s.signal_score > 0.4));
  const buys = actionableSignals.filter(s => s.action === "BUY" && s.signal_score <= 0.4);
  const sells = actionableSignals.filter(s => s.action === "SELL" || s.action === "STRONG_SELL");

  return (
    <div className="space-y-6">
      {/* Market Context Header */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Fear & Greed */}
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-4 text-center">Fear & Greed Index</h2>
          <FearGreedGauge 
            value={data.market_context.fear_greed} 
            label={data.market_context.fear_greed_label}
          />
        </div>

        {/* VIX */}
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-4">Volatility (VIX)</h2>
          <div className="flex flex-col items-center justify-center h-40">
            <div className="text-5xl font-bold">{data.market_context.vix.toFixed(1)}</div>
            <div className={`text-lg mt-2 ${
              data.market_context.vix_regime === "High" ? "text-red-500" :
              data.market_context.vix_regime === "Normal" ? "text-yellow-500" : "text-emerald-500"
            }`}>
              {data.market_context.vix_regime} Volatility
            </div>
            <div className="text-zinc-500 text-sm mt-2">
              {data.market_context.vix < 15 ? "Complacency zone" :
               data.market_context.vix < 20 ? "Normal range" :
               data.market_context.vix < 30 ? "Elevated fear" : "Panic levels"}
            </div>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-4">Signal Summary</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-zinc-400">Strong Buys</span>
              <span className="text-emerald-500 font-bold text-xl">{strongBuys.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-zinc-400">Buys</span>
              <span className="text-emerald-400 font-bold text-xl">{buys.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-zinc-400">Sells</span>
              <span className="text-red-500 font-bold text-xl">{sells.length}</span>
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-zinc-800">
              <span className="text-zinc-400">Total Scanned</span>
              <span className="font-bold">{data.signals.length}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Last Updated */}
      <div className="text-sm">
        <TimeAgo timestamp={data.timestamp} staleAfterMs={600000} />
      </div>

      {/* Top Picks */}
      {strongBuys.length > 0 && (
        <div className="bg-gradient-to-r from-emerald-900/30 to-zinc-900 rounded-xl p-6 border border-emerald-800/50">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <CheckCircle2 className="w-6 h-6 text-emerald-500" />
            🎯 Top Picks
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {strongBuys.slice(0, 6).map((signal, idx) => (
              <SignalCard key={signal.symbol} signal={signal} rank={idx + 1} featured />
            ))}
          </div>
        </div>
      )}

      {/* All Signals Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Buy Signals */}
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-emerald-500" />
            Buy Signals ({buys.length})
          </h2>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {buys.length > 0 ? buys.map((signal) => (
              <SignalCard key={signal.symbol} signal={signal} />
            )) : (
              <p className="text-zinc-500 text-center py-6 text-sm">No buy signals right now. The market may be bearish or between scan cycles.</p>
            )}
          </div>
        </div>

        {/* Sell Signals */}
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingDown className="w-5 h-5 text-red-500" />
            Sell Signals ({sells.length})
          </h2>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {sells.length > 0 ? sells.map((signal) => (
              <SignalCard key={signal.symbol} signal={signal} />
            )) : (
              <p className="text-zinc-500 text-center py-6 text-sm">No sell signals right now. The market may be bullish or between scan cycles.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function SignalCard({ signal, rank, featured }: { signal: Signal; rank?: number; featured?: boolean }) {
  const isBuy = signal.action.includes("BUY");
  const confidence = Math.abs(signal.signal_score * 100).toFixed(0);
  
  const actionColors: Record<string, string> = {
    STRONG_BUY: "bg-emerald-600 text-white",
    BUY: "bg-emerald-600/50 text-emerald-300",
    HOLD: "bg-zinc-600 text-zinc-300",
    SELL: "bg-red-600/50 text-red-300",
    STRONG_SELL: "bg-red-600 text-white",
  };

  return (
    <div className={`rounded-lg p-4 ${featured ? "bg-zinc-800/80" : "bg-zinc-800"}`}>
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          {rank && (
            <span className="text-2xl font-bold text-zinc-500">#{rank}</span>
          )}
          <div>
            <span className="font-bold text-lg">{signal.symbol}</span>
            <div className="text-zinc-400 text-sm">${signal.price.toFixed(2)}</div>
          </div>
        </div>
        <div className="text-right">
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${actionColors[signal.action] || actionColors.HOLD}`}>
            {signal.action.replace("_", " ")}
          </span>
          <div className="text-sm mt-1 text-zinc-400">
            {confidence}% confidence
          </div>
        </div>
      </div>

      {/* Position sizing */}
      <div className="text-sm text-zinc-400 mb-2">
        Suggested: {signal.position_size} shares
      </div>

      {/* Indicators mini-display */}
      <div className="flex gap-2 flex-wrap mb-2">
        <IndicatorPill 
          label="RSI" 
          value={signal.indicators.rsi.value.toFixed(0)}
          status={signal.indicators.rsi.is_oversold ? "good" : signal.indicators.rsi.is_overbought ? "bad" : "neutral"}
        />
        <IndicatorPill 
          label="Vol" 
          value={`${signal.indicators.volume.relative_volume.toFixed(1)}x`}
          status={signal.indicators.volume.is_confirming ? "good" : "neutral"}
        />
        <IndicatorPill 
          label="Trend" 
          value={signal.indicators.trend.trend}
          status={signal.indicators.trend.trend === "up" ? "good" : signal.indicators.trend.trend === "down" ? "bad" : "neutral"}
        />
      </div>

      {/* Reasons */}
      <div className="text-xs text-zinc-500 space-y-1">
        {signal.reasons.slice(0, 3).map((reason, i) => (
          <div key={i} className="flex items-start gap-1">
            <span className={isBuy ? "text-emerald-500" : "text-red-500"}>•</span>
            <span>{reason}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function IndicatorPill({ label, value, status }: { label: string; value: string; status: "good" | "bad" | "neutral" }) {
  const colors = {
    good: "bg-emerald-900/50 text-emerald-400 border-emerald-800",
    bad: "bg-red-900/50 text-red-400 border-red-800",
    neutral: "bg-zinc-800 text-zinc-400 border-zinc-700",
  };

  return (
    <span className={`text-xs px-2 py-1 rounded border ${colors[status]}`}>
      {label}: {value}
    </span>
  );
}
