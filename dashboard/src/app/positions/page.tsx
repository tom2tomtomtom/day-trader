"use client";

import { useEffect, useState, useCallback } from "react";
import { TrendingUp, TrendingDown, Clock, DollarSign } from "lucide-react";
import { TimeAgo } from "@/components/TimeAgo";
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
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/positions");
      if (res.ok) {
        const positions = await res.json();
        setData(positions);
        setFetchedAt(new Date().toISOString());
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
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-hot"></div>
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
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Positions</h1>
        <TimeAgo timestamp={fetchedAt} staleAfterMs={600000} />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-black-card rounded-xl p-4 border border-border-subtle">
          <div className="text-white-muted text-sm">Total Trades</div>
          <div className="text-2xl font-bold">{data?.total_trades || 0}</div>
        </div>
        <div className="bg-black-card rounded-xl p-4 border border-border-subtle">
          <div className="text-white-muted text-sm">Win Rate</div>
          <div className="text-2xl font-bold text-orange-accent">{winRate}%</div>
        </div>
        <div className="bg-black-card rounded-xl p-4 border border-border-subtle">
          <div className="text-white-muted text-sm">Winners</div>
          <div className="text-2xl font-bold text-orange-accent">{data?.winners || 0}</div>
        </div>
        <div className="bg-black-card rounded-xl p-4 border border-border-subtle">
          <div className="text-white-muted text-sm">Losers</div>
          <div className="text-2xl font-bold text-red-hot">{data?.losers || 0}</div>
        </div>
      </div>

      {/* Open Positions */}
      <div className="bg-black-card rounded-xl border border-border-subtle">
        <div className="p-4 border-b border-border-subtle">
          <h2 className="font-semibold flex items-center gap-2">
            <Clock className="w-5 h-5 text-orange-accent" />
            Open Positions ({openPositions.length})
          </h2>
        </div>
        
        {openPositions.length > 0 ? (
          <div className="divide-y divide-border-subtle">
            {openPositions.map(([sym, pos]) => (
              <OpenPositionRow key={sym} position={{ ...pos, symbol: sym }} />
            ))}
          </div>
        ) : (
          <div className="p-8 text-center">
            <TrendingUp className="w-10 h-10 text-white-faint mx-auto mb-3" />
            <p className="text-white-muted font-medium mb-1">No open positions</p>
            <p className="text-white-muted text-sm">Positions open automatically when the system detects strong signals with high confidence.</p>
          </div>
        )}
      </div>

      {/* Closed Trades */}
      <div className="bg-black-card rounded-xl border border-border-subtle">
        <div className="p-4 border-b border-border-subtle">
          <h2 className="font-semibold flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-white-muted" />
            Today&apos;s Closed Trades ({closedTrades.length})
          </h2>
        </div>
        
        {closedTrades.length > 0 ? (
          <div className="divide-y divide-border-subtle">
            {closedTrades.map((trade, idx) => (
              <ClosedTradeRow key={idx} trade={trade} />
            ))}
          </div>
        ) : (
          <div className="p-8 text-center">
            <DollarSign className="w-10 h-10 text-white-faint mx-auto mb-3" />
            <p className="text-white-muted font-medium mb-1">No closed trades today</p>
            <p className="text-white-muted text-sm">Trades close when stop-loss, take-profit, or trailing stop conditions are met.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function OpenPositionRow({ position }: { position: Position }) {
  const isLong = position.direction.toLowerCase() === "long";
  const isProfitable = (position.pnl ?? 0) >= 0;

  return (
    <div className="p-4 hover:bg-black-deep/50">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg">{position.symbol}</span>
              <span
                className={`text-xs px-2 py-0.5 rounded ${
                  isLong
                    ? "bg-red-hot/20 text-orange-accent"
                    : "bg-red-hot/20 text-red-hot"
                }`}
              >
                {position.direction}
              </span>
            </div>
            <div className="text-sm text-white-muted">
              {position.shares} shares @ ${position.entry_price.toFixed(2)}
            </div>
          </div>
        </div>
        
        <div className="text-right">
          <div className={`text-lg font-semibold ${isProfitable ? "text-orange-accent" : "text-red-hot"}`}>
            {isProfitable ? "+" : ""}${(position.pnl ?? 0).toFixed(2)}
          </div>
          <div className="text-sm text-white-muted">
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
    <div className="p-4 hover:bg-black-deep/50">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-semibold">{trade.symbol}</span>
            <span className="text-xs text-white-muted">{trade.direction}</span>
          </div>
          <div className="text-sm text-white-muted">
            {trade.shares} @ ${trade.entry_price.toFixed(2)} → ${trade.exit_price.toFixed(2)}
          </div>
        </div>
        
        <div className="text-right">
          <div className={`font-semibold ${isProfitable ? "text-orange-accent" : "text-red-hot"}`}>
            {isProfitable ? "+" : ""}${trade.pnl.toFixed(2)} ({trade.pnl_pct?.toFixed(1)}%)
          </div>
          <div className="text-xs text-white-muted">{trade.exit_reason}</div>
        </div>
      </div>
    </div>
  );
}
