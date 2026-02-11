"use client";

import { useEffect, useState, useCallback } from "react";
import { Globe, TrendingUp, TrendingDown } from "lucide-react";
import { PriceChart } from "@/components/PriceChart";
import { TimeAgo } from "@/components/TimeAgo";
import { useTableSubscription } from "@/hooks/useRealtimeSubscription";

interface MarketStatus {
  open: boolean;
  hours: string;
}

interface RegionalData {
  regime: string;
  change_1d: number | null;
}

interface MarketsData {
  market_status: Record<string, MarketStatus>;
  active_markets: string[];
  global_regime: string;
  regional?: Record<string, RegionalData>;
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
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/markets");
      if (res.ok) {
        const markets = await res.json();
        setData(markets);
        setFetchedAt(new Date().toISOString());
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
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-hot"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold">Global Markets</h1>
          <TimeAgo timestamp={fetchedAt} staleAfterMs={600000} />
        </div>
        <div className="flex items-center gap-2">
          <Globe className="w-5 h-5 text-white-muted" />
          <span className={`font-semibold ${
            data?.global_regime === "GLOBAL_RISK_ON"
              ? "text-orange-accent"
              : data?.global_regime === "GLOBAL_RISK_OFF"
              ? "text-red-hot"
              : "text-white-muted"
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
            className={`bg-black-card rounded-xl p-4 border transition-colors ${
              selectedMarket === market
                ? "border-red-hot"
                : "border-border-subtle hover:border-border-subtle"
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-2xl">{MARKET_FLAGS[market] || "🌍"}</span>
              <span className="font-semibold">{market}</span>
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  status.open ? "bg-red-hot animate-pulse" : "bg-white-dim"
                }`}
              />
              <span className={`text-sm ${status.open ? "text-orange-accent" : "text-white-dim"}`}>
                {status.open ? "OPEN" : "CLOSED"}
              </span>
            </div>
            <div className="text-xs text-white-dim mt-1">{status.hours}</div>
          </button>
        ))}
      </div>

      {/* Selected Market Chart */}
      <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">
            {MARKET_FLAGS[selectedMarket]} {MARKET_INDICES[selectedMarket]?.name || selectedMarket}
          </h2>
          <span className="text-white-muted text-sm">
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
      <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
        <h2 className="text-lg font-semibold mb-4">Regional Regimes</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-white-muted text-sm">
                <th className="pb-3">Region</th>
                <th className="pb-3">Status</th>
                <th className="pb-3">Regime</th>
                <th className="pb-3 text-right">1D Change</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle">
              {Object.entries(data?.market_status || {}).map(([market, status]) => {
                const regionData = data?.regional?.[market];
                const regime = regionData?.regime || data?.global_regime || "N/A";
                const change1d = regionData?.change_1d;
                const hasChange = change1d !== null && change1d !== undefined;

                return (
                  <tr key={market} className="hover:bg-black-deep/50">
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        <span>{MARKET_FLAGS[market]}</span>
                        <span className="font-medium">{market}</span>
                      </div>
                    </td>
                    <td className="py-3">
                      <span
                        className={`inline-flex items-center gap-1 text-sm ${
                          status.open ? "text-orange-accent" : "text-white-dim"
                        }`}
                      >
                        <span className={`w-2 h-2 rounded-full ${
                          status.open ? "bg-red-hot" : "bg-white-dim"
                        }`} />
                        {status.open ? "Open" : "Closed"}
                      </span>
                    </td>
                    <td className="py-3">
                      <span className={`inline-flex items-center gap-1.5 text-sm font-medium ${
                        regime === "BULLISH" ? "text-orange-accent"
                          : regime === "BEARISH" ? "text-red-hot"
                          : "text-white-muted"
                      }`}>
                        {regime === "BULLISH" && <TrendingUp className="w-3.5 h-3.5" />}
                        {regime === "BEARISH" && <TrendingDown className="w-3.5 h-3.5" />}
                        {regime}
                      </span>
                    </td>
                    <td className="py-3 text-right">
                      {hasChange ? (
                        <span className={`font-medium ${
                          change1d > 0 ? "text-orange-accent" : change1d < 0 ? "text-red-hot" : "text-white-muted"
                        }`}>
                          {change1d > 0 ? "+" : ""}{change1d.toFixed(2)}%
                        </span>
                      ) : (
                        <span className="text-white-dim">N/A</span>
                      )}
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
