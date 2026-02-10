"use client";

import { useEffect, useState } from "react";
import {
  Brain,
  AlertTriangle,
  Shield,
  Zap,
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  Clock,
  BarChart3,
  Flame,
} from "lucide-react";
import { TimeAgo } from "@/components/TimeAgo";

interface Trigger {
  type: string;
  severity: string;
  title: string;
  description: string;
  direction: string;
}

interface Opportunity {
  symbol: string;
  action: string;
  confidence: number;
  opportunity_score: number;
  conviction_label: string;
  score_breakdown: Record<string, number>;
  council_action: string;
  council_score: number;
  council_conviction: string;
  council_bulls: number;
  council_bears: number;
  congress_buying: number;
  congress_selling: number;
  headline: string;
  thesis: string;
  bull_case: string;
  bear_case: string;
  risk_briefing: string;
  smart_money_note: string;
  timing_note: string;
  tags: string[];
  key_drivers: string[];
  key_risks: string[];
  position_size_pct: number;
}

interface IntelData {
  timestamp: string;
  market: {
    regime: string;
    regime_score: number;
    fear_greed: number;
    vix: number;
    risk_score: number;
    opportunity_score: number;
  };
  triggers: Trigger[];
  critical_triggers: number;
  digest: {
    headline: string;
    mood: string;
    regime_narrative: string;
    risk_warnings: string[];
    smart_money: string;
    closing_thought: string;
  };
  opportunities: Opportunity[];
  congress: {
    total_trades: number;
    signals: number;
    cluster_buys: number;
    hot_symbols: { symbol: string; trade_count: number }[];
    notable_activity: string[];
  };
  stats: {
    symbols_analyzed: number;
    actionable_signals: number;
  };
}

function severityColor(severity: string): string {
  if (severity === "critical") return "bg-red-600/20 text-red-400 border-red-500/30";
  if (severity === "high") return "bg-orange-600/20 text-orange-400 border-orange-500/30";
  if (severity === "medium") return "bg-yellow-600/20 text-yellow-400 border-yellow-500/30";
  return "bg-zinc-700/50 text-zinc-400 border-zinc-600/30";
}

function scoreColor(score: number): string {
  if (score >= 80) return "text-emerald-400";
  if (score >= 60) return "text-blue-400";
  if (score >= 40) return "text-yellow-400";
  if (score >= 20) return "text-orange-400";
  return "text-red-400";
}

function convictionBadge(label: string): string {
  const styles: Record<string, string> = {
    Maximum: "bg-emerald-600/20 text-emerald-400",
    High: "bg-blue-600/20 text-blue-400",
    Moderate: "bg-yellow-600/20 text-yellow-400",
    Low: "bg-orange-600/20 text-orange-400",
    "No Trade": "bg-red-600/20 text-red-400",
  };
  return styles[label] || "bg-zinc-700 text-zinc-400";
}

const SCORE_LABELS: Record<string, { label: string; color: string }> = {
  technical: { label: "Technical", color: "bg-blue-500" },
  sentiment: { label: "Sentiment", color: "bg-purple-500" },
  smart_money: { label: "Smart $", color: "bg-green-500" },
  council: { label: "Council", color: "bg-yellow-500" },
  macro: { label: "Macro", color: "bg-cyan-500" },
  quality: { label: "Quality", color: "bg-pink-500" },
};

export default function IntelPage() {
  const [data, setData] = useState<IntelData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedOpp, setExpandedOpp] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/api/intel");
        if (res.ok) {
          const intel = await res.json();
          setData(intel);
        } else {
          const err = await res.json();
          setError(err.error);
        }
      } catch {
        setError("Failed to fetch intelligence data");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-zinc-900 rounded-xl p-8 border border-zinc-800 text-center">
        <AlertTriangle className="w-12 h-12 text-cyan-500/50 mx-auto mb-4" />
        <h2 className="text-xl font-bold mb-2">Intelligence Briefing Not Available</h2>
        <p className="text-zinc-400 mb-4 max-w-md mx-auto">{error}</p>
        <p className="text-zinc-500 text-sm mb-4">Briefings run automatically at 9:00 AM and 4:30 PM ET. You can also trigger one manually:</p>
        <code className="bg-zinc-800 px-4 py-2 rounded text-sm">
          python3 -m core.orchestrator --intel
        </code>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-6">
      {/* Hero Header */}
      <div className="bg-gradient-to-r from-cyan-900/30 via-purple-900/20 to-zinc-900 rounded-xl p-6 border border-cyan-800/30">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2 mb-2">
              <Brain className="w-7 h-7 text-cyan-500" />
              Intelligence Briefing
            </h1>
            <p className="text-xl font-semibold text-zinc-200 mb-2">
              {data.digest.headline}
            </p>
            <p className="text-sm text-zinc-400 max-w-2xl">{data.digest.mood}</p>
            <TimeAgo timestamp={data.timestamp} staleAfterMs={3600000} className="mt-2 inline-block" />
          </div>
          <div className="text-right space-y-2">
            <div className="flex gap-4">
              <div className="text-center">
                <div className="text-xs text-zinc-500 mb-1">Risk</div>
                <div
                  className={`text-2xl font-bold ${
                    data.market.risk_score > 60
                      ? "text-red-400"
                      : data.market.risk_score > 40
                        ? "text-yellow-400"
                        : "text-emerald-400"
                  }`}
                >
                  {data.market.risk_score.toFixed(0)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-zinc-500 mb-1">Opportunity</div>
                <div
                  className={`text-2xl font-bold ${
                    data.market.opportunity_score > 60
                      ? "text-emerald-400"
                      : data.market.opportunity_score > 40
                        ? "text-yellow-400"
                        : "text-red-400"
                  }`}
                >
                  {data.market.opportunity_score.toFixed(0)}
                </div>
              </div>
            </div>
            <div className="text-xs text-zinc-500">
              {data.stats.actionable_signals} actionable / {data.stats.symbols_analyzed} analyzed
            </div>
          </div>
        </div>
      </div>

      {/* Market Context Row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="bg-zinc-900 rounded-xl p-4 border border-zinc-800">
          <div className="text-xs text-zinc-500 mb-1">Regime</div>
          <div className="font-semibold text-sm capitalize">
            {data.market.regime.replace(/_/g, " ")}
          </div>
          <div className="text-xs text-zinc-500">Score: {data.market.regime_score}</div>
        </div>
        <div className="bg-zinc-900 rounded-xl p-4 border border-zinc-800">
          <div className="text-xs text-zinc-500 mb-1">Fear & Greed</div>
          <div
            className={`font-semibold text-lg ${
              data.market.fear_greed < 25
                ? "text-red-400"
                : data.market.fear_greed > 75
                  ? "text-emerald-400"
                  : "text-yellow-400"
            }`}
          >
            {data.market.fear_greed}
          </div>
        </div>
        <div className="bg-zinc-900 rounded-xl p-4 border border-zinc-800">
          <div className="text-xs text-zinc-500 mb-1">VIX</div>
          <div
            className={`font-semibold text-lg ${
              data.market.vix > 30
                ? "text-red-400"
                : data.market.vix > 20
                  ? "text-yellow-400"
                  : "text-emerald-400"
            }`}
          >
            {data.market.vix.toFixed(1)}
          </div>
        </div>
        <div className="bg-zinc-900 rounded-xl p-4 border border-zinc-800">
          <div className="text-xs text-zinc-500 mb-1">Triggers</div>
          <div className="font-semibold text-lg">
            {data.triggers.length}
            {data.critical_triggers > 0 && (
              <span className="text-red-400 text-sm ml-1">
                ({data.critical_triggers} critical)
              </span>
            )}
          </div>
        </div>
        <div className="bg-zinc-900 rounded-xl p-4 border border-zinc-800">
          <div className="text-xs text-zinc-500 mb-1">Congress Signals</div>
          <div className="font-semibold text-lg text-amber-400">
            {data.congress?.signals || 0}
          </div>
        </div>
      </div>

      {/* Triggers */}
      {data.triggers.length > 0 && (
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-orange-400" />
            Active Macro Triggers
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {data.triggers.slice(0, 6).map((trigger, i) => (
              <div
                key={i}
                className={`rounded-lg p-4 border ${severityColor(trigger.severity)}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-sm">{trigger.title}</span>
                  <span className="text-xs uppercase px-2 py-0.5 rounded bg-zinc-800/50">
                    {trigger.severity}
                  </span>
                </div>
                <p className="text-xs opacity-80">{trigger.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risk Warnings + Regime */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-400" />
            Regime Analysis
          </h2>
          <p className="text-sm text-zinc-300">{data.digest.regime_narrative}</p>
        </div>
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <Shield className="w-5 h-5 text-red-400" />
            Risk Warnings
          </h2>
          <div className="space-y-2">
            {data.digest.risk_warnings.map((w, i) => (
              <div key={i} className="text-sm text-zinc-300 flex gap-2">
                <span className="text-red-500 mt-0.5">!</span>
                <span>{w}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Top Opportunities */}
      <div className="bg-gradient-to-r from-emerald-900/20 to-zinc-900 rounded-xl p-6 border border-emerald-800/30">
        <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Target className="w-6 h-6 text-emerald-400" />
          Top Opportunities
        </h2>

        <div className="space-y-4">
          {data.opportunities.slice(0, 8).map((opp) => (
            <div key={opp.symbol} className="bg-zinc-800/50 rounded-xl overflow-hidden">
              {/* Summary Row */}
              <button
                onClick={() =>
                  setExpandedOpp(expandedOpp === opp.symbol ? null : opp.symbol)
                }
                className="w-full text-left p-4 hover:bg-zinc-700/30 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div>
                      <span className="text-lg font-bold">{opp.symbol}</span>
                      <span
                        className={`ml-2 text-sm ${
                          opp.action.includes("BUY")
                            ? "text-emerald-400"
                            : opp.action.includes("SELL")
                              ? "text-red-400"
                              : "text-zinc-400"
                        }`}
                      >
                        {opp.action}
                      </span>
                    </div>
                    <div className={`text-sm px-2 py-0.5 rounded ${convictionBadge(opp.conviction_label)}`}>
                      {opp.conviction_label}
                    </div>
                    {opp.congress_buying > 0 && (
                      <div className="text-xs bg-amber-500/10 text-amber-400 px-2 py-0.5 rounded border border-amber-500/20">
                        Congress: {opp.congress_buying}B
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-6">
                    {/* Score Breakdown Mini Bar */}
                    <div className="hidden md:flex items-center gap-1">
                      {Object.entries(opp.score_breakdown).map(([key, val]) => {
                        const config = SCORE_LABELS[key];
                        return (
                          <div
                            key={key}
                            className="flex flex-col items-center"
                            title={`${config?.label}: ${val.toFixed(1)}`}
                          >
                            <div className="w-8 h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${config?.color || "bg-zinc-500"}`}
                                style={{ width: `${Math.min(100, val * 4)}%` }}
                              />
                            </div>
                            <span className="text-[10px] text-zinc-500 mt-0.5">
                              {config?.label?.substring(0, 3)}
                            </span>
                          </div>
                        );
                      })}
                    </div>

                    {/* Overall Score */}
                    <div className="text-right">
                      <div className={`text-2xl font-bold ${scoreColor(opp.opportunity_score)}`}>
                        {opp.opportunity_score.toFixed(0)}
                      </div>
                      <div className="text-[10px] text-zinc-500">/ 100</div>
                    </div>
                  </div>
                </div>

                <div className="text-sm text-zinc-400 mt-2">{opp.headline}</div>

                {/* Tags */}
                <div className="flex gap-1.5 mt-2">
                  {opp.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-[10px] bg-zinc-700/50 text-zinc-500 px-1.5 py-0.5 rounded"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </button>

              {/* Expanded Detail */}
              {expandedOpp === opp.symbol && (
                <div className="border-t border-zinc-700/50 p-4 space-y-4">
                  {/* Thesis */}
                  <div>
                    <h4 className="text-sm font-semibold text-zinc-400 mb-1">Trade Thesis</h4>
                    <p className="text-sm text-zinc-300">{opp.thesis}</p>
                  </div>

                  {/* Bull / Bear / Risk */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div className="bg-emerald-500/5 rounded-lg p-3 border border-emerald-500/10">
                      <h5 className="text-xs font-semibold text-emerald-400 mb-1 flex items-center gap-1">
                        <TrendingUp className="w-3 h-3" /> Bull Case
                      </h5>
                      <p className="text-xs text-zinc-300">{opp.bull_case}</p>
                    </div>
                    <div className="bg-red-500/5 rounded-lg p-3 border border-red-500/10">
                      <h5 className="text-xs font-semibold text-red-400 mb-1 flex items-center gap-1">
                        <TrendingDown className="w-3 h-3" /> Bear Case
                      </h5>
                      <p className="text-xs text-zinc-300">{opp.bear_case}</p>
                    </div>
                    <div className="bg-orange-500/5 rounded-lg p-3 border border-orange-500/10">
                      <h5 className="text-xs font-semibold text-orange-400 mb-1 flex items-center gap-1">
                        <Shield className="w-3 h-3" /> Risk Briefing
                      </h5>
                      <p className="text-xs text-zinc-300">{opp.risk_briefing}</p>
                    </div>
                  </div>

                  {/* Smart Money & Timing */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="text-sm">
                      <span className="text-zinc-500">Smart Money: </span>
                      <span className="text-zinc-300">{opp.smart_money_note}</span>
                    </div>
                    <div className="text-sm">
                      <span className="text-zinc-500">Timing: </span>
                      <span className="text-zinc-300">{opp.timing_note}</span>
                    </div>
                  </div>

                  {/* Score Breakdown Full */}
                  <div>
                    <h4 className="text-sm font-semibold text-zinc-400 mb-2">Score Breakdown</h4>
                    <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
                      {Object.entries(opp.score_breakdown).map(([key, val]) => {
                        const config = SCORE_LABELS[key];
                        return (
                          <div key={key} className="bg-zinc-800 rounded-lg p-2 text-center">
                            <div className="text-xs text-zinc-500">{config?.label || key}</div>
                            <div className={`text-lg font-bold ${scoreColor(val * 4)}`}>
                              {val.toFixed(1)}
                            </div>
                            <div className="h-1 bg-zinc-700 rounded-full mt-1 overflow-hidden">
                              <div
                                className={`h-full rounded-full ${config?.color || "bg-zinc-500"}`}
                                style={{ width: `${Math.min(100, val * 4)}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Position Size */}
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-zinc-500">Suggested Position:</span>
                    <span className="font-semibold text-zinc-200">
                      {opp.position_size_pct}% of portfolio
                    </span>
                    {opp.council_conviction && (
                      <>
                        <span className="text-zinc-600">|</span>
                        <span className="text-zinc-500">Council:</span>
                        <span className="text-zinc-200">
                          {opp.council_conviction} ({opp.council_bulls}B/{opp.council_bears}S)
                        </span>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Closing Thought */}
      <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800 text-center">
        <p className="text-zinc-300 italic">{data.digest.closing_thought}</p>
        <p className="text-xs mt-2">
          <TimeAgo timestamp={data.timestamp} staleAfterMs={3600000} prefix="Analysis generated" />
        </p>
      </div>
    </div>
  );
}
