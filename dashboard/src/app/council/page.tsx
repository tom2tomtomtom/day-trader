"use client";

import { useEffect, useState } from "react";
import {
  Users,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  Shield,
  Zap,
  Eye,
  BarChart3,
  Brain,
} from "lucide-react";
import { TimeAgo } from "@/components/TimeAgo";

interface PersonaVerdict {
  name: string;
  style: string;
  stance: string;
  conviction: number;
  reasoning: string[];
  key_metric: string;
  key_value: string;
  risk_flag: string | null;
}

interface CouncilOpportunity {
  symbol: string;
  action: string;
  council_action: string;
  council_score: number;
  council_conviction: string;
  council_bulls: number;
  council_bears: number;
  persona_verdicts: PersonaVerdict[];
  opportunity_score: number;
  headline: string;
  thesis: string;
  key_drivers: string[];
  key_risks: string[];
}

interface CouncilData {
  timestamp: string;
  opportunities: CouncilOpportunity[];
}

const PERSONA_CONFIG: Record<
  string,
  { emoji: string; color: string; bgColor: string; borderColor: string }
> = {
  Warren: {
    emoji: "🎩",
    color: "text-orange-accent",
    bgColor: "bg-orange-accent/10",
    borderColor: "border-orange-accent/30",
  },
  Michael: {
    emoji: "🔍",
    color: "text-red-hot",
    bgColor: "bg-red-hot/10",
    borderColor: "border-red-hot/30",
  },
  Cathie: {
    emoji: "🚀",
    color: "text-red-hot",
    bgColor: "bg-red-hot/10",
    borderColor: "border-red-hot/30",
  },
  Ray: {
    emoji: "🌍",
    color: "text-orange-accent",
    bgColor: "bg-orange-accent/10",
    borderColor: "border-orange-accent/30",
  },
  Nancy: {
    emoji: "💰",
    color: "text-orange-accent",
    bgColor: "bg-orange-accent/10",
    borderColor: "border-green-500/30",
  },
  Jesse: {
    emoji: "📈",
    color: "text-yellow-electric",
    bgColor: "bg-yellow-500/10",
    borderColor: "border-yellow-500/30",
  },
};

function StanceIcon({ stance }: { stance: string }) {
  if (stance === "STRONG_BUY" || stance === "BUY")
    return <TrendingUp className="w-4 h-4 text-orange-accent" />;
  if (stance === "STRONG_SELL" || stance === "SELL")
    return <TrendingDown className="w-4 h-4 text-red-hot" />;
  return <Minus className="w-4 h-4 text-white-muted" />;
}

function stanceColor(stance: string): string {
  if (stance === "STRONG_BUY") return "text-orange-accent font-bold";
  if (stance === "BUY") return "text-orange-accent";
  if (stance === "STRONG_SELL") return "text-red-hot font-bold";
  if (stance === "SELL") return "text-red-300";
  return "text-white-muted";
}

function convictionBadge(level: string): string {
  const styles: Record<string, string> = {
    Unanimous: "bg-red-hot/20 text-orange-accent border border-red-hot/30",
    "Strong Majority":
      "bg-orange-accent/20 text-orange-accent border border-orange-accent/30",
    Majority: "bg-yellow-600/20 text-yellow-electric border border-yellow-500/30",
    Split: "bg-orange-accent/20 text-orange-accent border border-orange-500/30",
    Contested: "bg-red-hot/20 text-red-hot border border-red-hot/30",
  };
  return styles[level] || "bg-black-card text-white-muted";
}

export default function CouncilPage() {
  const [data, setData] = useState<CouncilData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/api/council");
        if (res.ok) {
          const council = await res.json();
          setData(council);
          if (council.opportunities?.length > 0) {
            setSelectedSymbol(council.opportunities[0].symbol);
          }
        } else {
          const err = await res.json();
          setError(err.error);
        }
      } catch {
        setError("Failed to fetch council data");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-hot"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-black-card rounded-xl p-8 border border-border-subtle text-center">
        <AlertTriangle className="w-12 h-12 text-red-hot/50 mx-auto mb-4" />
        <h2 className="text-xl font-bold mb-2">Phantom Council Not Yet Convened</h2>
        <p className="text-white-muted mb-4 max-w-md mx-auto">{error}</p>
        <p className="text-white-dim text-sm mb-4">AI investor personas debate trade ideas during intel briefings (9 AM and 4:30 PM ET):</p>
        <code className="bg-black-deep px-4 py-2 rounded text-sm">
          python3 -m core.orchestrator --intel
        </code>
      </div>
    );
  }

  if (!data) return null;

  const selected = data.opportunities.find((o) => o.symbol === selectedSymbol);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="w-7 h-7 text-red-hot" />
            Phantom Council
          </h1>
          <p className="text-white-muted text-sm">
            6 AI investor personas debate every trade opportunity
          </p>
        </div>
        <TimeAgo timestamp={data.timestamp} staleAfterMs={3600000} />
      </div>

      {/* Persona Legend */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {Object.entries(PERSONA_CONFIG).map(([name, config]) => (
          <div
            key={name}
            className={`${config.bgColor} ${config.borderColor} border rounded-lg p-3 text-center`}
          >
            <div className="text-2xl mb-1">{config.emoji}</div>
            <div className={`font-semibold text-sm ${config.color}`}>{name}</div>
            <div className="text-xs text-white-dim">
              {name === "Warren" && "Value"}
              {name === "Michael" && "Contrarian"}
              {name === "Cathie" && "Growth"}
              {name === "Ray" && "Macro"}
              {name === "Nancy" && "Flow"}
              {name === "Jesse" && "Momentum"}
            </div>
          </div>
        ))}
      </div>

      {/* Symbol Selector + Council View */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Symbol List */}
        <div className="bg-black-card rounded-xl border border-border-subtle overflow-hidden">
          <div className="p-4 border-b border-border-subtle">
            <h2 className="font-semibold flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-red-hot" />
              Analyzed Symbols
            </h2>
          </div>
          <div className="max-h-[600px] overflow-y-auto">
            {data.opportunities.map((opp) => (
              <button
                key={opp.symbol}
                onClick={() => setSelectedSymbol(opp.symbol)}
                className={`w-full text-left px-4 py-3 border-b border-border-subtle hover:bg-black-deep/50 transition-colors ${
                  selectedSymbol === opp.symbol ? "bg-red-hot/20 border-l-2 border-l-red-hot" : ""
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <span className="font-bold">{opp.symbol}</span>
                    <span className={`ml-2 text-sm ${stanceColor(opp.council_action)}`}>
                      {opp.council_action}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-mono">
                      {opp.council_score > 0 ? "+" : ""}
                      {opp.council_score.toFixed(0)}
                    </div>
                    <div className={`text-xs px-2 py-0.5 rounded-full ${convictionBadge(opp.council_conviction)}`}>
                      {opp.council_conviction}
                    </div>
                  </div>
                </div>
                <div className="flex gap-1 mt-2">
                  {opp.persona_verdicts?.map((v) => {
                    const config = PERSONA_CONFIG[v.name] || { emoji: "?", color: "text-white-muted" };
                    return (
                      <div
                        key={v.name}
                        className="flex items-center gap-0.5"
                        title={`${v.name}: ${v.stance}`}
                      >
                        <span className="text-xs">{config.emoji}</span>
                        <StanceIcon stance={v.stance} />
                      </div>
                    );
                  })}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Council Detail View */}
        <div className="lg:col-span-2 space-y-4">
          {selected ? (
            <>
              {/* Header Card */}
              <div className="bg-gradient-to-r from-red-hot/30 to-black-card rounded-xl p-6 border border-red-hot/30">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h2 className="text-2xl font-bold">{selected.symbol}</h2>
                    <p className="text-white-muted text-sm mt-1">{selected.headline}</p>
                  </div>
                  <div className="text-right">
                    <div className={`text-3xl font-bold ${
                      selected.council_score > 0 ? "text-orange-accent" :
                      selected.council_score < 0 ? "text-red-hot" : "text-white-muted"
                    }`}>
                      {selected.council_score > 0 ? "+" : ""}
                      {selected.council_score.toFixed(0)}
                    </div>
                    <div className={`text-sm px-3 py-1 rounded-full mt-1 inline-block ${convictionBadge(selected.council_conviction)}`}>
                      {selected.council_conviction}
                    </div>
                  </div>
                </div>

                {/* Vote Bar */}
                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-orange-accent">
                      {selected.council_bulls} Bulls
                    </span>
                    <span className="text-red-hot">
                      {selected.council_bears} Bears
                    </span>
                  </div>
                  <div className="flex h-3 rounded-full overflow-hidden bg-black-deep">
                    <div
                      className="bg-red-hot transition-all"
                      style={{
                        width: `${(selected.council_bulls / 6) * 100}%`,
                      }}
                    />
                    <div
                      className="bg-white-dim transition-all"
                      style={{
                        width: `${((6 - selected.council_bulls - selected.council_bears) / 6) * 100}%`,
                      }}
                    />
                    <div
                      className="bg-red-hot transition-all"
                      style={{
                        width: `${(selected.council_bears / 6) * 100}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Thesis */}
                <p className="text-sm text-white-muted">{selected.thesis}</p>
              </div>

              {/* Persona Verdicts Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {selected.persona_verdicts?.map((verdict) => {
                  const config = PERSONA_CONFIG[verdict.name] || {
                    emoji: "?",
                    color: "text-white-muted",
                    bgColor: "bg-black-deep",
                    borderColor: "border-border-subtle",
                  };
                  return (
                    <div
                      key={verdict.name}
                      className={`${config.bgColor} ${config.borderColor} border rounded-xl p-4`}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span className="text-xl">{config.emoji}</span>
                          <div>
                            <div className={`font-semibold ${config.color}`}>
                              {verdict.name}
                            </div>
                            <div className="text-xs text-white-dim">
                              {verdict.style}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`font-bold ${stanceColor(verdict.stance)}`}>
                            {verdict.stance.replace("_", " ")}
                          </div>
                          <div className="text-xs text-white-muted">
                            {verdict.conviction.toFixed(0)}% conviction
                          </div>
                        </div>
                      </div>

                      {/* Conviction Bar */}
                      <div className="h-1.5 bg-black-deep rounded-full mb-3 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            verdict.stance.includes("BUY")
                              ? "bg-red-hot"
                              : verdict.stance.includes("SELL")
                                ? "bg-red-hot"
                                : "bg-white-dim"
                          }`}
                          style={{ width: `${verdict.conviction}%` }}
                        />
                      </div>

                      {/* Key Metric */}
                      <div className="flex justify-between text-xs mb-2">
                        <span className="text-white-dim">{verdict.key_metric}</span>
                        <span className="text-white-muted">{verdict.key_value}</span>
                      </div>

                      {/* Reasoning */}
                      <div className="space-y-1">
                        {verdict.reasoning.slice(0, 3).map((r, i) => (
                          <div key={i} className="text-xs text-white-muted flex gap-1">
                            <span className={config.color}>•</span>
                            <span>{r}</span>
                          </div>
                        ))}
                      </div>

                      {/* Risk Flag */}
                      {verdict.risk_flag && (
                        <div className="mt-2 flex items-center gap-1 text-xs text-red-hot">
                          <AlertTriangle className="w-3 h-3" />
                          {verdict.risk_flag}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Drivers & Risks */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-black-card rounded-xl p-4 border border-border-subtle">
                  <h3 className="font-semibold text-sm mb-3 flex items-center gap-2">
                    <Zap className="w-4 h-4 text-orange-accent" />
                    Key Drivers
                  </h3>
                  {selected.key_drivers.map((d, i) => (
                    <div key={i} className="text-sm text-white-muted mb-1 flex gap-2">
                      <span className="text-orange-accent">+</span> {d}
                    </div>
                  ))}
                </div>
                <div className="bg-black-card rounded-xl p-4 border border-border-subtle">
                  <h3 className="font-semibold text-sm mb-3 flex items-center gap-2">
                    <Shield className="w-4 h-4 text-red-hot" />
                    Key Risks
                  </h3>
                  {selected.key_risks.map((r, i) => (
                    <div key={i} className="text-sm text-white-muted mb-1 flex gap-2">
                      <span className="text-red-hot">!</span> {r}
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="bg-black-card rounded-xl p-8 text-center text-white-dim">
              <Eye className="w-12 h-12 mx-auto mb-4 opacity-50" />
              Select a symbol to view council analysis
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
