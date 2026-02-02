"use client";

import { useEffect, useState } from "react";
import { Zap, TrendingUp, Flame, Target, AlertTriangle, Users } from "lucide-react";

interface EdgeOpportunity {
  symbol: string;
  edge_score: number;
  reasons: string[];
  short_interest?: {
    short_percent: number;
    days_to_cover: number;
    squeeze_potential: string;
  };
  options?: {
    put_call_ratio: number;
    options_signal: string;
    unusual_activity: boolean;
  };
  earnings?: {
    beat_streak: number;
    earnings_momentum: string;
  };
}

interface WSBStock {
  symbol: string;
  mentions_24h: number;
  mention_change_pct: number;
  rank: number;
}

interface SectorData {
  symbol: string;
  name: string;
  pct_1d: number;
  pct_5d: number;
  momentum_score: number;
}

interface EdgeData {
  timestamp: string;
  edge_opportunities: EdgeOpportunity[];
  wsb_momentum: WSBStock[];
  wsb_trending: WSBStock[];
  squeeze_setups: { symbol: string; short_percent: number; days_to_cover: number }[];
  sector_rotation: {
    rotation_signal: string;
    leaders: SectorData[];
    laggards: SectorData[];
    sectors: SectorData[];
  };
  summary: {
    total_scanned: number;
    edge_opportunities: number;
    squeeze_setups: number;
    wsb_momentum: number;
    sector_signal: string;
  };
}

export default function EdgePage() {
  const [data, setData] = useState<EdgeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/api/edge");
        if (res.ok) {
          const edge = await res.json();
          setData(edge);
        } else {
          const err = await res.json();
          setError(err.error);
        }
      } catch (e) {
        setError("Failed to fetch edge data");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-zinc-900 rounded-xl p-8 border border-zinc-800 text-center">
        <AlertTriangle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
        <h2 className="text-xl font-bold mb-2">Edge Scanner Not Run</h2>
        <p className="text-zinc-400 mb-4">{error}</p>
        <code className="bg-zinc-800 px-4 py-2 rounded text-sm">
          python3 edge_scanner.py
        </code>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Target className="w-7 h-7 text-yellow-500" />
            Edge Scanner
          </h1>
          <p className="text-zinc-400 text-sm">
            Last scan: {new Date(data.timestamp).toLocaleString()}
          </p>
        </div>
        <div className="flex gap-4 text-sm">
          <div className="bg-zinc-800 px-3 py-2 rounded-lg">
            <span className="text-zinc-400">Opportunities:</span>{" "}
            <span className="text-yellow-500 font-bold">{data.summary.edge_opportunities}</span>
          </div>
          <div className="bg-zinc-800 px-3 py-2 rounded-lg">
            <span className="text-zinc-400">Squeezes:</span>{" "}
            <span className="text-orange-500 font-bold">{data.summary.squeeze_setups}</span>
          </div>
        </div>
      </div>

      {/* Top Row: Sector Rotation + WSB */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sector Rotation */}
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-500" />
            Sector Rotation
            <span className={`ml-auto text-sm px-2 py-1 rounded ${
              data.sector_rotation?.rotation_signal === "RISK_ON" ? "bg-emerald-600/20 text-emerald-400" :
              data.sector_rotation?.rotation_signal === "RISK_OFF" ? "bg-red-600/20 text-red-400" :
              "bg-zinc-700 text-zinc-400"
            }`}>
              {data.sector_rotation?.rotation_signal || "NEUTRAL"}
            </span>
          </h2>
          
          <div className="space-y-2">
            {data.sector_rotation?.sectors?.map((sector) => (
              <div key={sector.symbol} className="flex items-center gap-3">
                <span className="w-12 font-mono text-sm">{sector.symbol}</span>
                <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${sector.pct_5d >= 0 ? "bg-emerald-500" : "bg-red-500"}`}
                    style={{ 
                      width: `${Math.min(100, Math.abs(sector.pct_5d) * 10)}%`,
                      marginLeft: sector.pct_5d < 0 ? "auto" : 0
                    }}
                  />
                </div>
                <span className={`w-16 text-right text-sm ${
                  sector.pct_5d >= 0 ? "text-emerald-500" : "text-red-500"
                }`}>
                  {sector.pct_5d >= 0 ? "+" : ""}{sector.pct_5d.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* WSB Momentum */}
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Users className="w-5 h-5 text-orange-500" />
            🦍 WSB Trending
          </h2>
          
          <div className="space-y-3">
            {data.wsb_trending?.slice(0, 8).map((stock, idx) => (
              <div key={stock.symbol} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-zinc-500 text-sm w-6">#{idx + 1}</span>
                  <span className="font-semibold">{stock.symbol}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-zinc-400 text-sm">{stock.mentions_24h} mentions</span>
                  <span className={`text-sm font-medium ${
                    stock.mention_change_pct > 50 ? "text-orange-500" :
                    stock.mention_change_pct > 0 ? "text-emerald-500" : "text-red-500"
                  }`}>
                    {stock.mention_change_pct > 0 ? "+" : ""}{stock.mention_change_pct.toFixed(0)}%
                    {stock.mention_change_pct > 100 && " 🔥"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Edge Opportunities */}
      <div className="bg-gradient-to-r from-yellow-900/20 to-zinc-900 rounded-xl p-6 border border-yellow-800/30">
        <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Zap className="w-6 h-6 text-yellow-500" />
          Top Edge Opportunities
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.edge_opportunities?.slice(0, 9).map((opp, idx) => (
            <EdgeCard key={`${opp.symbol}-${idx}`} opportunity={opp} rank={idx + 1} />
          ))}
        </div>
      </div>

      {/* Squeeze Setups */}
      {data.squeeze_setups && data.squeeze_setups.length > 0 && (
        <div className="bg-zinc-900 rounded-xl p-6 border border-orange-800/30">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Flame className="w-5 h-5 text-orange-500" />
            🚀 Squeeze Setups
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[...new Map(data.squeeze_setups.map(s => [s.symbol, s])).values()].map((setup) => (
              <div key={setup.symbol} className="bg-zinc-800 rounded-lg p-4">
                <div className="text-lg font-bold">{setup.symbol}</div>
                <div className="text-sm text-orange-400">
                  {setup.short_percent?.toFixed(1)}% Short Interest
                </div>
                <div className="text-sm text-zinc-400">
                  {setup.days_to_cover?.toFixed(1)} Days to Cover
                </div>
                <div className="mt-2 h-2 bg-zinc-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-yellow-500 to-orange-500"
                    style={{ width: `${Math.min(100, setup.short_percent * 2)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function EdgeCard({ opportunity, rank }: { opportunity: EdgeOpportunity; rank: number }) {
  const scoreColor = opportunity.edge_score >= 4 ? "text-yellow-500" :
                     opportunity.edge_score >= 3 ? "text-emerald-500" : "text-zinc-400";
  
  return (
    <div className="bg-zinc-800/80 rounded-lg p-4">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xl font-bold text-zinc-500">#{rank}</span>
          <span className="font-bold text-lg">{opportunity.symbol}</span>
        </div>
        <div className={`text-lg font-bold ${scoreColor}`}>
          {opportunity.edge_score}⭐
        </div>
      </div>
      
      {/* Key metrics */}
      <div className="space-y-2 mb-3">
        {opportunity.short_interest?.short_percent && opportunity.short_interest.short_percent > 15 && (
          <div className="flex justify-between text-sm">
            <span className="text-zinc-400">Short Interest</span>
            <span className="text-orange-400 font-medium">
              {opportunity.short_interest.short_percent.toFixed(1)}%
            </span>
          </div>
        )}
        {opportunity.options?.put_call_ratio && (
          <div className="flex justify-between text-sm">
            <span className="text-zinc-400">Put/Call</span>
            <span className={`font-medium ${
              opportunity.options.put_call_ratio < 0.7 ? "text-emerald-400" :
              opportunity.options.put_call_ratio > 1.3 ? "text-red-400" : "text-zinc-300"
            }`}>
              {opportunity.options.put_call_ratio.toFixed(2)}
            </span>
          </div>
        )}
        {opportunity.earnings?.beat_streak && opportunity.earnings.beat_streak >= 3 && (
          <div className="flex justify-between text-sm">
            <span className="text-zinc-400">Beat Streak</span>
            <span className="text-emerald-400 font-medium">
              {opportunity.earnings.beat_streak} quarters
            </span>
          </div>
        )}
      </div>
      
      {/* Reasons */}
      <div className="text-xs text-zinc-500 space-y-1">
        {opportunity.reasons.slice(0, 3).map((reason, i) => (
          <div key={i} className="flex items-start gap-1">
            <span className="text-yellow-500">•</span>
            <span>{reason}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
