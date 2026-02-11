"use client";

import { useEffect, useState } from "react";
import { RefreshCw, TrendingUp, TrendingDown, Zap, BarChart3 } from "lucide-react";

interface WatchlistItem {
  symbol: string;
  name: string;
  price: number;
  change_pct: number;
  setups: string[];
  trade_ideas: string[];
}

interface WatchlistData {
  timestamp: string;
  scanned: number;
  watchlist: WatchlistItem[];
  setups: Record<string, SetupItem[]>;
}

interface SetupItem {
  symbol: string;
  name: string;
  price: number;
  change_pct: number;
  setup: string;
  trade_idea: string;
  signal_strength: number;
}

export default function WatchlistPage() {
  const [data, setData] = useState<WatchlistData | null>(null);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);

  const fetchData = async () => {
    try {
      const res = await fetch("/api/watchlist");
      if (res.ok) {
        const watchlist = await res.json();
        setData(watchlist);
      }
    } catch (error) {
      console.error("Failed to fetch watchlist:", error);
    } finally {
      setLoading(false);
    }
  };

  const runScan = async () => {
    setScanning(true);
    try {
      const res = await fetch("/api/scan", { method: "POST" });
      if (res.ok) {
        await fetchData();
      }
    } catch (error) {
      console.error("Scan failed:", error);
    } finally {
      setScanning(false);
    }
  };

  useEffect(() => {
    fetchData();
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Watchlist</h1>
          {data?.timestamp && (
            <p className="text-white-muted text-sm">
              Last scan: {new Date(data.timestamp).toLocaleString()}
            </p>
          )}
        </div>
        <button
          onClick={runScan}
          disabled={scanning}
          className="flex items-center gap-2 px-4 py-2 bg-red-hot hover:bg-red-dim disabled:bg-black-card rounded-lg transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${scanning ? "animate-spin" : ""}`} />
          {scanning ? "Scanning..." : "Run Scan"}
        </button>
      </div>

      {/* Today's Picks */}
      <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-electric" />
          Today&apos;s Top Picks
        </h2>
        
        {data?.watchlist && data.watchlist.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.watchlist.map((item, idx) => (
              <WatchlistCard key={item.symbol} item={item} rank={idx + 1} />
            ))}
          </div>
        ) : (
          <p className="text-white-dim text-center py-8">
            No watchlist items. Run a scan to find opportunities.
          </p>
        )}
      </div>

      {/* Setups by Category */}
      {data?.setups && Object.keys(data.setups).length > 0 && (
        <div className="space-y-6">
          <h2 className="text-lg font-semibold">Setups by Category</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {Object.entries(data.setups).map(([category, items]) => (
              <SetupCategory key={category} category={category} items={items} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function WatchlistCard({ item, rank }: { item: WatchlistItem; rank: number }) {
  const isPositive = item.change_pct >= 0;
  
  return (
    <div className="bg-black-deep rounded-lg p-4 hover:bg-black-card transition-colors">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-white-dim text-sm">#{rank}</span>
          <span className="font-bold text-lg">{item.symbol}</span>
        </div>
        <div className={`flex items-center gap-1 ${isPositive ? "text-orange-accent" : "text-red-hot"}`}>
          {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          <span className="font-semibold">{item.change_pct.toFixed(2)}%</span>
        </div>
      </div>
      
      <div className="text-white-muted text-sm mb-3">{item.name}</div>
      <div className="text-xl font-semibold mb-3">${item.price.toFixed(2)}</div>
      
      <div className="flex flex-wrap gap-1 mb-3">
        {item.setups.slice(0, 3).map((setup) => (
          <span
            key={setup}
            className="text-xs px-2 py-1 bg-black-card rounded-full"
          >
            {setup.replace(/_/g, " ")}
          </span>
        ))}
      </div>
      
      <div className="text-sm text-white-muted line-clamp-2">
        {item.trade_ideas[0]}
      </div>
    </div>
  );
}

function SetupCategory({ category, items }: { category: string; items: SetupItem[] }) {
  const categoryIcons: Record<string, string> = {
    gaps: "📊",
    volume_surges: "📈",
    breakouts: "🚀",
    breakdowns: "📉",
    mean_reversion: "↩️",
    high_volatility: "⚡",
  };

  return (
    <div className="bg-black-card rounded-xl p-4 border border-border-subtle">
      <h3 className="font-semibold mb-3 flex items-center gap-2">
        <span>{categoryIcons[category] || "📋"}</span>
        {category.replace(/_/g, " ").toUpperCase()}
      </h3>
      <div className="space-y-2">
        {items.slice(0, 5).map((item) => (
          <div
            key={item.symbol}
            className="flex items-center justify-between text-sm py-2 border-b border-border-subtle last:border-0"
          >
            <div className="flex items-center gap-3">
              <span className="font-medium">{item.symbol}</span>
              <span className="text-white-dim">${item.price.toFixed(2)}</span>
            </div>
            <span className={item.change_pct >= 0 ? "text-orange-accent" : "text-red-hot"}>
              {item.change_pct >= 0 ? "+" : ""}{item.change_pct.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
