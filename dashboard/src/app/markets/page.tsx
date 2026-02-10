"use client";

import { useEffect, useState, useCallback } from "react";
import { Globe, TrendingUp, TrendingDown } from "lucide-react";
import { PriceChart } from "@/components/PriceChart";
import { useTableSubscription } from "@/hooks/useRealtimeSubscription";

interface MarketStatus {
  open: boolean;
  hours: string;
}

interface RegionalData {
  regime: string;
  score: number;
  change_1d: number;
  market_open: boolean;
}

interface MarketsData {
  market_status: Record<string, MarketStatus>;
  active_markets: string[];
  global_regime: string;
  regional?: {
    regions: Record<string, RegionalData>;
  };
}

const MARKET_FLAGS: Record<string, string> = {
  US: "🇺🇸",
  Europe: "🇪🇺",
  Japan: "🇯🇵",
  HongKong: "🇭🇰",
  Australia: "🇦🇺",
  Korea: "🇰🇷",
};

const MARKET_INDICES: Record<string, { symbol: string; name: string }> = {
  US: { symbol: "SPY", name: "S&P 500" },
  Europe: { symbol: "FEZ", name: "Euro Stoxx" },
  Japan: { symbol: "EWJ", name: "Nikkei (ETF)" },
  HongKong: { symbol: "FXI", name: "China/HK" },
  Australia: { symbol: "EWA", name: "ASX (ETF)" },
  Korea: { symbol: "EWY", name: "KOSPI (ETF)" },
};

export default function MarketsPage() {
  const [data, setData] = useState<MarketsData | null>(null);
  const [selectedMarket, setSelectedMarket] = useState<string>("US");
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/markets");
      if (res.ok) {
        const markets = await res.json();
        setData(markets);
      }
    } catch (error) {
      console.error("Failed to fetch markets:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 120000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Real-time: refetch when market snapshots change
  useTableSubscription("market_snapshots", fetchData);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Global Markets</h1>
        <div className="flex items-center gap-2">
          <Globe className="w-5 h-5 text-zinc-400" />
          <span className={`font-semibold ${
            data?.global_regime === "GLOBAL_RISK_ON" 
              ? "text-emerald-500" 
              : data?.global_regime === "GLOBAL_RISK_OFF"
              ? "text-red-500"
              : "text-zinc-400"
          }`}>
            {data?.global_regime || "LOADING"}
          </span>
        </div>
      </div>

      {/* Market Status Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {Object.entries(data?.market_status || {}).map(([market, status]) => (
          <button
            key={market}
            onClick={() => setSelectedMarket(market)}
            className={`bg-zinc-900 rounded-xl p-4 border transition-colors ${
              selectedMarket === market
                ? "border-emerald-500"
                : "border-zinc-800 hover:border-zinc-700"
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-2xl">{MARKET_FLAGS[market] || "🌍"}</span>
              <span className="font-semibold">{market}</span>
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  status.open ? "bg-emerald-500 animate-pulse" : "bg-zinc-600"
                }`}
              />
              <span className={`text-sm ${status.open ? "text-emerald-500" : "text-zinc-500"}`}>
                {status.open ? "OPEN" : "CLOSED"}
              </span>
            </div>
            <div className="text-xs text-zinc-500 mt-1">{status.hours}</div>
          </button>
        ))}
      </div>

      {/* Selected Market Chart */}
      <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">
            {MARKET_FLAGS[selectedMarket]} {MARKET_INDICES[selectedMarket]?.name || selectedMarket}
          </h2>
          <span className="text-zinc-400 text-sm">
            {MARKET_INDICES[selectedMarket]?.symbol}
          </span>
        </div>
        <div className="h-80">
          <PriceChart 
            key={selectedMarket}
            symbol={MARKET_INDICES[selectedMarket]?.symbol || "SPY"} 
          />
        </div>
      </div>

      {/* Regional Breakdown */}
      <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
        <h2 className="text-lg font-semibold mb-4">Regional Regimes</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-zinc-400 text-sm">
                <th className="pb-3">Region</th>
                <th className="pb-3">Status</th>
                <th className="pb-3">Regime</th>
                <th className="pb-3 text-right">1D Change</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {Object.entries(data?.market_status || {}).map(([market, status]) => {
                const index = MARKET_INDICES[market];
                return (
                  <tr key={market} className="hover:bg-zinc-800/50">
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        <span>{MARKET_FLAGS[market]}</span>
                        <span className="font-medium">{market}</span>
                      </div>
                    </td>
                    <td className="py-3">
                      <span
                        className={`inline-flex items-center gap-1 text-sm ${
                          status.open ? "text-emerald-500" : "text-zinc-500"
                        }`}
                      >
                        <span className={`w-2 h-2 rounded-full ${
                          status.open ? "bg-emerald-500" : "bg-zinc-600"
                        }`} />
                        {status.open ? "Open" : "Closed"}
                      </span>
                    </td>
                    <td className="py-3">
                      <span className="text-sm text-zinc-400">
                        {index?.symbol || "-"}
                      </span>
                    </td>
                    <td className="py-3 text-right">
                      <span className="text-zinc-400">-</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
