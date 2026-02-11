"use client";

import { useEffect, useState } from "react";
import {
  Landmark,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Clock,
  Shield,
  Users,
  Flame,
} from "lucide-react";

interface Trade {
  member: string;
  party: string;
  chamber: string;
  symbol: string;
  company: string;
  trade_type: string;
  amount_range: string;
  amount_low: number;
  amount_high: number;
  trade_date: string;
  disclosure_date: string;
  filing_delay_days: number;
  committees: string[];
  asset_type: string;
}

interface CongressSignals {
  total_trades: number;
  signals: number;
  cluster_buys: number;
  hot_symbols: { symbol: string; trade_count: number }[];
  notable_activity: string[];
}

interface CongressData {
  timestamp: string;
  trades: Trade[];
  signals: CongressSignals | null;
}

function partyColor(party: string): string {
  if (party === "D") return "text-orange-accent";
  if (party === "R") return "text-red-hot";
  return "text-white-muted";
}

function partyBg(party: string): string {
  if (party === "D") return "bg-orange-accent/10 border-blue-500/20";
  if (party === "R") return "bg-red-hot/10 border-red-500/20";
  return "bg-black-deep border-border-subtle";
}

function delayColor(days: number): string {
  if (days > 60) return "text-red-hot";
  if (days > 45) return "text-orange-accent";
  if (days > 30) return "text-yellow-electric";
  return "text-orange-accent";
}

function tradeTypeIcon(type: string) {
  if (type.toLowerCase().includes("purchase") || type.toLowerCase().includes("buy"))
    return <TrendingUp className="w-4 h-4 text-orange-accent" />;
  return <TrendingDown className="w-4 h-4 text-red-hot" />;
}

export default function CongressPage() {
  const [data, setData] = useState<CongressData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterParty, setFilterParty] = useState<string>("all");
  const [filterType, setFilterType] = useState<string>("all");

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/api/congress");
        if (res.ok) {
          const congress = await res.json();
          setData(congress);
        } else {
          const err = await res.json();
          setError(err.error);
        }
      } catch {
        setError("Failed to fetch congressional data");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-accent"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-black-card rounded-xl p-8 border border-border-subtle text-center">
        <AlertTriangle className="w-12 h-12 text-orange-accent mx-auto mb-4" />
        <h2 className="text-xl font-bold mb-2">Congressional Data Not Loaded</h2>
        <p className="text-white-muted mb-4">{error}</p>
        <code className="bg-black-deep px-4 py-2 rounded text-sm">
          python3 -m core.orchestrator --intel
        </code>
      </div>
    );
  }

  if (!data) return null;

  // Filter trades
  let filtered = data.trades;
  if (filterParty !== "all") {
    filtered = filtered.filter((t) => t.party === filterParty);
  }
  if (filterType !== "all") {
    filtered = filtered.filter((t) =>
      filterType === "buy"
        ? t.trade_type.toLowerCase().includes("purchase")
        : t.trade_type.toLowerCase().includes("sale")
    );
  }

  // Stats
  const totalBuys = data.trades.filter((t) =>
    t.trade_type.toLowerCase().includes("purchase")
  ).length;
  const totalSells = data.trades.filter((t) =>
    t.trade_type.toLowerCase().includes("sale")
  ).length;
  const avgDelay =
    data.trades.reduce((sum, t) => sum + t.filing_delay_days, 0) /
    (data.trades.length || 1);
  const uniqueMembers = new Set(data.trades.map((t) => t.member)).size;

  // Group by symbol
  const symbolCounts: Record<string, { buys: number; sells: number; total: number }> = {};
  data.trades.forEach((t) => {
    if (!symbolCounts[t.symbol]) symbolCounts[t.symbol] = { buys: 0, sells: 0, total: 0 };
    symbolCounts[t.symbol].total++;
    if (t.trade_type.toLowerCase().includes("purchase")) symbolCounts[t.symbol].buys++;
    else symbolCounts[t.symbol].sells++;
  });
  const hotSymbols = Object.entries(symbolCounts)
    .sort((a, b) => b[1].total - a[1].total)
    .slice(0, 8);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Landmark className="w-7 h-7 text-orange-accent" />
            Congressional Intelligence
          </h1>
          <p className="text-white-muted text-sm">
            Track what Congress is trading - follow the smart money
          </p>
        </div>
        <div className="text-sm text-white-dim">
          {data.timestamp && new Date(data.timestamp).toLocaleString()}
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-black-card rounded-xl p-4 border border-border-subtle">
          <div className="text-white-muted text-sm mb-1">Total Trades</div>
          <div className="text-2xl font-bold">{data.trades.length}</div>
        </div>
        <div className="bg-black-card rounded-xl p-4 border border-border-subtle">
          <div className="text-white-muted text-sm mb-1">Buy/Sell Ratio</div>
          <div className="text-2xl font-bold">
            <span className="text-orange-accent">{totalBuys}</span>
            <span className="text-white-dim mx-1">/</span>
            <span className="text-red-hot">{totalSells}</span>
          </div>
        </div>
        <div className="bg-black-card rounded-xl p-4 border border-border-subtle">
          <div className="text-white-muted text-sm mb-1">Active Members</div>
          <div className="text-2xl font-bold">{uniqueMembers}</div>
        </div>
        <div className="bg-black-card rounded-xl p-4 border border-border-subtle">
          <div className="text-white-muted text-sm mb-1">Avg Filing Delay</div>
          <div className={`text-2xl font-bold ${delayColor(avgDelay)}`}>
            {avgDelay.toFixed(0)}d
          </div>
        </div>
      </div>

      {/* Hot Symbols + Signals */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Hot Symbols */}
        <div className="bg-gradient-to-r from-orange-accent/20 to-black-card rounded-xl p-6 border border-orange-accent/30">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Flame className="w-5 h-5 text-orange-accent" />
            Most Traded by Congress
          </h2>
          <div className="space-y-3">
            {hotSymbols.map(([symbol, counts], idx) => (
              <div key={symbol} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-white-dim text-sm w-6">#{idx + 1}</span>
                  <span className="font-bold">{symbol}</span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-orange-accent">{counts.buys} buys</span>
                    <span className="text-white-dim">|</span>
                    <span className="text-red-hot">{counts.sells} sells</span>
                  </div>
                  <div className="w-24 h-2 bg-black-deep rounded-full overflow-hidden flex">
                    <div
                      className="bg-red-hot h-full"
                      style={{
                        width: `${(counts.buys / counts.total) * 100}%`,
                      }}
                    />
                    <div
                      className="bg-red-hot h-full"
                      style={{
                        width: `${(counts.sells / counts.total) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Notable Activity */}
        <div className="bg-black-card rounded-xl p-6 border border-border-subtle">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Users className="w-5 h-5 text-orange-accent" />
            Notable Activity
          </h2>
          {data.signals?.notable_activity &&
          data.signals.notable_activity.length > 0 ? (
            <div className="space-y-3">
              {data.signals.notable_activity.map((note, i) => (
                <div
                  key={i}
                  className="text-sm text-white-muted bg-black-deep/50 rounded-lg p-3"
                >
                  <Shield className="w-4 h-4 text-orange-accent inline mr-2" />
                  {note}
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-3">
              {data.trades
                .filter((t) => t.amount_high > 250000)
                .slice(0, 5)
                .map((t, i) => (
                  <div
                    key={i}
                    className="text-sm text-white-muted bg-black-deep/50 rounded-lg p-3"
                  >
                    <Shield className="w-4 h-4 text-orange-accent inline mr-2" />
                    {t.member} ({t.party}) - {t.trade_type} {t.symbol}{" "}
                    {t.amount_range}
                  </div>
                ))}
            </div>
          )}
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-4 items-center">
        <div className="flex gap-2">
          <button
            onClick={() => setFilterParty("all")}
            className={`px-3 py-1.5 rounded-lg text-sm ${
              filterParty === "all"
                ? "bg-black-card text-white"
                : "bg-black-deep text-white-muted hover:text-white"
            }`}
          >
            All Parties
          </button>
          <button
            onClick={() => setFilterParty("D")}
            className={`px-3 py-1.5 rounded-lg text-sm ${
              filterParty === "D"
                ? "bg-orange-accent/20 text-orange-accent border border-orange-accent/30"
                : "bg-black-deep text-white-muted hover:text-orange-accent"
            }`}
          >
            Democrat
          </button>
          <button
            onClick={() => setFilterParty("R")}
            className={`px-3 py-1.5 rounded-lg text-sm ${
              filterParty === "R"
                ? "bg-red-hot/20 text-red-hot border border-red-hot/30"
                : "bg-black-deep text-white-muted hover:text-red-hot"
            }`}
          >
            Republican
          </button>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setFilterType("all")}
            className={`px-3 py-1.5 rounded-lg text-sm ${
              filterType === "all"
                ? "bg-black-card text-white"
                : "bg-black-deep text-white-muted hover:text-white"
            }`}
          >
            All Types
          </button>
          <button
            onClick={() => setFilterType("buy")}
            className={`px-3 py-1.5 rounded-lg text-sm ${
              filterType === "buy"
                ? "bg-red-hot/20 text-orange-accent"
                : "bg-black-deep text-white-muted hover:text-orange-accent"
            }`}
          >
            Buys Only
          </button>
          <button
            onClick={() => setFilterType("sell")}
            className={`px-3 py-1.5 rounded-lg text-sm ${
              filterType === "sell"
                ? "bg-red-hot/20 text-red-hot"
                : "bg-black-deep text-white-muted hover:text-red-hot"
            }`}
          >
            Sells Only
          </button>
        </div>
      </div>

      {/* Trade Table */}
      <div className="bg-black-card rounded-xl border border-border-subtle overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border-subtle text-white-muted">
                <th className="text-left p-4">Member</th>
                <th className="text-left p-4">Symbol</th>
                <th className="text-left p-4">Type</th>
                <th className="text-left p-4">Amount</th>
                <th className="text-left p-4">Trade Date</th>
                <th className="text-left p-4">Delay</th>
                <th className="text-left p-4">Committees</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((trade, i) => (
                <tr
                  key={i}
                  className="border-b border-border-subtle hover:bg-black-deep/30"
                >
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      <span
                        className={`inline-block w-2 h-2 rounded-full ${
                          trade.party === "D" ? "bg-orange-accent" : "bg-red-hot"
                        }`}
                      />
                      <div>
                        <div className={`font-medium ${partyColor(trade.party)}`}>
                          {trade.member}
                        </div>
                        <div className="text-xs text-white-dim">{trade.chamber}</div>
                      </div>
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="font-bold">{trade.symbol}</div>
                    <div className="text-xs text-white-dim truncate max-w-[140px]">
                      {trade.company}
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-1">
                      {tradeTypeIcon(trade.trade_type)}
                      <span className="text-white-muted">{trade.trade_type}</span>
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-1">
                      <DollarSign className="w-3 h-3 text-white-dim" />
                      <span
                        className={`${
                          trade.amount_high > 500000
                            ? "text-orange-accent font-semibold"
                            : "text-white-muted"
                        }`}
                      >
                        {trade.amount_range}
                      </span>
                    </div>
                  </td>
                  <td className="p-4 text-white-muted">{trade.trade_date}</td>
                  <td className="p-4">
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3 text-white-dim" />
                      <span className={delayColor(trade.filing_delay_days)}>
                        {trade.filing_delay_days}d
                      </span>
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="flex flex-wrap gap-1">
                      {trade.committees.slice(0, 2).map((c) => (
                        <span
                          key={c}
                          className="text-xs bg-black-deep text-white-muted px-2 py-0.5 rounded"
                        >
                          {c}
                        </span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
