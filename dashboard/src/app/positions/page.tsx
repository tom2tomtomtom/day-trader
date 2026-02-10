"use client";

import { useEffect, useState, useCallback } from "react";
import { TrendingUp, TrendingDown, Clock, DollarSign } from "lucide-react";
import { useRealtimeSubscription } from "@/hooks/useRealtimeSubscription";

interface Position {
  symbol: string;
  direction: string;
  shares: number;
  entry_price: number;
  entry_time: string;
  stop_price: number;
  target_price: number;
  current_price?: number;
  pnl?: number;
  pnl_pct?: number;
}

interface ClosedTrade extends Position {
  exit_price: number;
  exit_time: string;
  exit_reason: string;
  pnl: number;
}

interface PositionsData {
  positions: Record<string, Position>;
  closed_trades: ClosedTrade[];
  total_trades: number;
  winners: number;
  losers: number;
  gross_pnl: number;
}

export default function PositionsPage() {
  const [data, setData] = useState<PositionsData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/positions");
      if (res.ok) {
        const positions = await res.json();
        setData(positions);
      }
    } catch (error) {
      console.error("Failed to fetch positions:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 120000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Real-time: refetch when positions or trades change
  useRealtimeSubscription([
    { table: "positions", onchange: fetchData },
    { table: "trades", onchange: fetchData },
  ]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  const openPositions = data?.positions ? Object.entries(data.positions) : [];
  const closedTrades = data?.closed_trades || [];
  const winRate = data && data.total_trades > 0 
    ? ((data.winners / data.total_trades) * 100).toFixed(1) 
    : "0";

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Positions</h1>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-zinc-900 rounded-xl p-4 border border-zinc-800">
          <div className="text-zinc-400 text-sm">Total Trades</div>
          <div className="text-2xl font-bold">{data?.total_trades || 0}</div>
        </div>
        <div className="bg-zinc-900 rounded-xl p-4 border border-zinc-800">
          <div className="text-zinc-400 text-sm">Win Rate</div>
          <div className="text-2xl font-bold text-emerald-500">{winRate}%</div>
        </div>
        <div className="bg-zinc-900 rounded-xl p-4 border border-zinc-800">
          <div className="text-zinc-400 text-sm">Winners</div>
          <div className="text-2xl font-bold text-emerald-500">{data?.winners || 0}</div>
        </div>
        <div className="bg-zinc-900 rounded-xl p-4 border border-zinc-800">
          <div className="text-zinc-400 text-sm">Losers</div>
          <div className="text-2xl font-bold text-red-500">{data?.losers || 0}</div>
        </div>
      </div>

      {/* Open Positions */}
      <div className="bg-zinc-900 rounded-xl border border-zinc-800">
        <div className="p-4 border-b border-zinc-800">
          <h2 className="font-semibold flex items-center gap-2">
            <Clock className="w-5 h-5 text-emerald-500" />
            Open Positions ({openPositions.length})
          </h2>
        </div>
        
        {openPositions.length > 0 ? (
          <div className="divide-y divide-zinc-800">
            {openPositions.map(([sym, pos]) => (
              <OpenPositionRow key={sym} position={{ ...pos, symbol: sym }} />
            ))}
          </div>
        ) : (
          <div className="p-8 text-center">
            <TrendingUp className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
            <p className="text-zinc-400 font-medium mb-1">No open positions</p>
            <p className="text-zinc-600 text-sm">Positions open automatically when the system detects strong signals with high confidence.</p>
          </div>
        )}
      </div>

      {/* Closed Trades */}
      <div className="bg-zinc-900 rounded-xl border border-zinc-800">
        <div className="p-4 border-b border-zinc-800">
          <h2 className="font-semibold flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-zinc-400" />
            Today&apos;s Closed Trades ({closedTrades.length})
          </h2>
        </div>
        
        {closedTrades.length > 0 ? (
          <div className="divide-y divide-zinc-800">
            {closedTrades.map((trade, idx) => (
              <ClosedTradeRow key={idx} trade={trade} />
            ))}
          </div>
        ) : (
          <div className="p-8 text-center">
            <DollarSign className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
            <p className="text-zinc-400 font-medium mb-1">No closed trades today</p>
            <p className="text-zinc-600 text-sm">Trades close when stop-loss, take-profit, or trailing stop conditions are met.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function OpenPositionRow({ position }: { position: Position }) {
  const isLong = position.direction === "LONG";
  const isProfitable = (position.pnl ?? 0) >= 0;

  return (
    <div className="p-4 hover:bg-zinc-800/50">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg">{position.symbol}</span>
              <span
                className={`text-xs px-2 py-0.5 rounded ${
                  isLong
                    ? "bg-emerald-600/20 text-emerald-400"
                    : "bg-red-600/20 text-red-400"
                }`}
              >
                {position.direction}
              </span>
            </div>
            <div className="text-sm text-zinc-400">
              {position.shares} shares @ ${position.entry_price.toFixed(2)}
            </div>
          </div>
        </div>
        
        <div className="text-right">
          <div className={`text-lg font-semibold ${isProfitable ? "text-emerald-500" : "text-red-500"}`}>
            {isProfitable ? "+" : ""}${(position.pnl ?? 0).toFixed(2)}
          </div>
          <div className="text-sm text-zinc-400">
            Stop: ${position.stop_price.toFixed(2)} | Target: ${position.target_price.toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
}

function ClosedTradeRow({ trade }: { trade: ClosedTrade }) {
  const isProfitable = trade.pnl >= 0;

  return (
    <div className="p-4 hover:bg-zinc-800/50">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-semibold">{trade.symbol}</span>
            <span className="text-xs text-zinc-400">{trade.direction}</span>
          </div>
          <div className="text-sm text-zinc-400">
            {trade.shares} @ ${trade.entry_price.toFixed(2)} → ${trade.exit_price.toFixed(2)}
          </div>
        </div>
        
        <div className="text-right">
          <div className={`font-semibold ${isProfitable ? "text-emerald-500" : "text-red-500"}`}>
            {isProfitable ? "+" : ""}${trade.pnl.toFixed(2)} ({trade.pnl_pct?.toFixed(1)}%)
          </div>
          <div className="text-xs text-zinc-400">{trade.exit_reason}</div>
        </div>
      </div>
    </div>
  );
}
