"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Settings2,
  TrendingUp,
  BarChart3,
  Zap,
  Save,
  Check,
  AlertTriangle,
  Loader2,
  Info,
} from "lucide-react";

interface StrategyParameters {
  [key: string]: number;
}

interface StrategyConfig {
  name: string;
  enabled: boolean;
  parameters: StrategyParameters;
}

interface StrategiesResponse {
  strategies: StrategyConfig[];
  source: string;
}

const STRATEGY_META: Record<
  string,
  {
    label: string;
    description: string;
    regime: string;
    regimeColor: string;
    icon: React.ReactNode;
    gradient: string;
  }
> = {
  momentum: {
    label: "Momentum",
    description:
      "Follows strong price trends using moving average crossovers. Buys when the fast MA crosses above the slow MA and rides the trend with trailing stops.",
    regime: "Trending",
    regimeColor: "text-orange-accent bg-red-hot/20",
    icon: <TrendingUp className="w-5 h-5" />,
    gradient: "from-red-hot/30 to-black-card",
  },
  mean_reversion: {
    label: "Mean Reversion",
    description:
      "Bets on price returning to the mean when it moves too far from Bollinger Bands. Buys at the lower band and sells at the upper band.",
    regime: "Ranging",
    regimeColor: "text-orange-accent bg-orange-accent/20",
    icon: <BarChart3 className="w-5 h-5" />,
    gradient: "from-red-hot/30 to-black-card",
  },
  breakout: {
    label: "Breakout",
    description:
      "Detects when price breaks above resistance levels after consolidation. Enters on confirmed breakouts with volume confirmation.",
    regime: "Breakout",
    regimeColor: "text-yellow-electric bg-yellow-600/20",
    icon: <Zap className="w-5 h-5" />,
    gradient: "from-orange-accent/30 to-black-card",
  },
};

const PARAMETER_INFO: Record<string, Record<string, { label: string; tooltip: string; step: number; min: number; max: number }>> = {
  momentum: {
    fast_ma: {
      label: "Fast MA Period",
      tooltip: "Short-term moving average period. Lower values react faster to price changes.",
      step: 1,
      min: 2,
      max: 50,
    },
    slow_ma: {
      label: "Slow MA Period",
      tooltip: "Long-term moving average period. Defines the trend direction baseline.",
      step: 1,
      min: 10,
      max: 200,
    },
    stop_loss_pct: {
      label: "Stop Loss %",
      tooltip: "Maximum allowed loss per trade before automatic exit.",
      step: 0.01,
      min: 0.01,
      max: 0.2,
    },
    take_profit_pct: {
      label: "Take Profit %",
      tooltip: "Target profit percentage to lock in gains.",
      step: 0.01,
      min: 0.02,
      max: 0.5,
    },
  },
  mean_reversion: {
    bb_period: {
      label: "Bollinger Period",
      tooltip: "Lookback period for Bollinger Band calculation. Standard is 20.",
      step: 1,
      min: 5,
      max: 50,
    },
    bb_std: {
      label: "Bollinger Std Dev",
      tooltip: "Number of standard deviations for band width. Higher = wider bands, fewer signals.",
      step: 0.1,
      min: 0.5,
      max: 4.0,
    },
    stop_loss_pct: {
      label: "Stop Loss %",
      tooltip: "Maximum allowed loss per trade before automatic exit.",
      step: 0.01,
      min: 0.01,
      max: 0.2,
    },
    take_profit_pct: {
      label: "Take Profit %",
      tooltip: "Target profit percentage to lock in gains.",
      step: 0.01,
      min: 0.02,
      max: 0.5,
    },
  },
  breakout: {
    lookback: {
      label: "Lookback Period",
      tooltip: "Number of bars to look back for resistance level detection.",
      step: 1,
      min: 5,
      max: 100,
    },
    stop_loss_pct: {
      label: "Stop Loss %",
      tooltip: "Maximum allowed loss per trade before automatic exit.",
      step: 0.01,
      min: 0.01,
      max: 0.2,
    },
    take_profit_pct: {
      label: "Take Profit %",
      tooltip: "Target profit percentage to lock in gains.",
      step: 0.01,
      min: 0.02,
      max: 0.5,
    },
  },
};

export default function StrategiesPage() {
  const [strategies, setStrategies] = useState<StrategyConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [source, setSource] = useState<string>("");
  const [savingMap, setSavingMap] = useState<Record<string, boolean>>({});
  const [feedbackMap, setFeedbackMap] = useState<
    Record<string, { type: "success" | "error"; message: string }>
  >({});

  const fetchStrategies = useCallback(async () => {
    try {
      const res = await fetch("/api/strategies");
      if (res.ok) {
        const data: StrategiesResponse = await res.json();
        setStrategies(data.strategies);
        setSource(data.source);
      }
    } catch {
      // Leave defaults empty, will show error state
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStrategies();
  }, [fetchStrategies]);

  const handleToggle = (strategyName: string) => {
    setStrategies((prev) =>
      prev.map((s) =>
        s.name === strategyName ? { ...s, enabled: !s.enabled } : s
      )
    );
  };

  const handleParamChange = (
    strategyName: string,
    paramKey: string,
    value: number
  ) => {
    setStrategies((prev) =>
      prev.map((s) =>
        s.name === strategyName
          ? { ...s, parameters: { ...s.parameters, [paramKey]: value } }
          : s
      )
    );
  };

  const handleSave = async (strategy: StrategyConfig) => {
    setSavingMap((prev) => ({ ...prev, [strategy.name]: true }));
    setFeedbackMap((prev) => {
      const next = { ...prev };
      delete next[strategy.name];
      return next;
    });

    try {
      const res = await fetch("/api/strategies", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(strategy),
      });

      const data = await res.json();

      if (res.ok) {
        setFeedbackMap((prev) => ({
          ...prev,
          [strategy.name]: { type: "success", message: "Strategy saved" },
        }));
      } else {
        setFeedbackMap((prev) => ({
          ...prev,
          [strategy.name]: {
            type: "error",
            message: data.error || "Failed to save",
          },
        }));
      }
    } catch {
      setFeedbackMap((prev) => ({
        ...prev,
        [strategy.name]: { type: "error", message: "Network error" },
      }));
    } finally {
      setSavingMap((prev) => ({ ...prev, [strategy.name]: false }));
      // Clear feedback after 3 seconds
      setTimeout(() => {
        setFeedbackMap((prev) => {
          const next = { ...prev };
          delete next[strategy.name];
          return next;
        });
      }, 3000);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-hot"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Settings2 className="w-7 h-7 text-orange-accent" />
            Strategy Configuration
          </h1>
          <p className="text-white-muted text-sm mt-1">
            Configure trading strategy parameters and enable/disable strategies
          </p>
        </div>
        <div className="flex gap-3 text-sm">
          <div className="bg-black-deep px-3 py-2 rounded-lg">
            <span className="text-white-muted">Source:</span>{" "}
            <span className="text-orange-accent font-medium">{source}</span>
          </div>
          <div className="bg-black-deep px-3 py-2 rounded-lg">
            <span className="text-white-muted">Active:</span>{" "}
            <span className="text-orange-accent font-bold">
              {strategies.filter((s) => s.enabled).length}
            </span>
            <span className="text-white-dim">/{strategies.length}</span>
          </div>
        </div>
      </div>

      {/* Strategy Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {strategies.map((strategy) => {
          const meta = STRATEGY_META[strategy.name] || {
            label: strategy.name,
            description: "",
            regime: "Unknown",
            regimeColor: "text-white-muted bg-white-dim/20",
            icon: <Settings2 className="w-5 h-5" />,
            gradient: "from-black-deep to-black-card",
          };
          const paramInfo = PARAMETER_INFO[strategy.name] || {};
          const isSaving = savingMap[strategy.name] || false;
          const feedback = feedbackMap[strategy.name];

          return (
            <div
              key={strategy.name}
              className={`bg-gradient-to-br ${meta.gradient} rounded-xl border ${
                strategy.enabled
                  ? "border-border-subtle"
                  : "border-border-subtle opacity-60"
              } transition-all`}
            >
              {/* Card Header */}
              <div className="p-5 border-b border-border-subtle">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span
                      className={
                        strategy.enabled ? "text-white-full" : "text-white-dim"
                      }
                    >
                      {meta.icon}
                    </span>
                    <div>
                      <h2 className="text-lg font-bold">{meta.label}</h2>
                      <span
                        className={`text-xs font-medium px-2 py-0.5 rounded-full ${meta.regimeColor}`}
                      >
                        {meta.regime} Regime
                      </span>
                    </div>
                  </div>

                  {/* Toggle */}
                  <button
                    onClick={() => handleToggle(strategy.name)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      strategy.enabled ? "bg-red-hot" : "bg-black-card"
                    }`}
                    aria-label={`Toggle ${meta.label} strategy`}
                  >
                    <span
                      className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                        strategy.enabled ? "translate-x-6" : "translate-x-0"
                      }`}
                    />
                  </button>
                </div>

                <p className="text-sm text-white-muted leading-relaxed">
                  {meta.description}
                </p>
              </div>

              {/* Parameters */}
              <div className="p-5 space-y-4">
                {Object.entries(strategy.parameters).map(([key, value]) => {
                  const info = paramInfo[key] || {
                    label: key.replace(/_/g, " "),
                    tooltip: "",
                    step: key.includes("pct") ? 0.01 : 1,
                    min: 0,
                    max: 999,
                  };

                  const isPercentage = key.includes("pct");

                  return (
                    <div key={key}>
                      <div className="flex items-center justify-between mb-1.5">
                        <label className="text-sm font-medium text-white-muted flex items-center gap-1.5">
                          {info.label}
                          {info.tooltip && (
                            <span className="group relative">
                              <Info className="w-3.5 h-3.5 text-white-dim cursor-help" />
                              <span className="invisible group-hover:visible absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-48 p-2 text-xs bg-black-deep border border-border-subtle rounded-lg text-white-muted z-10 shadow-xl">
                                {info.tooltip}
                              </span>
                            </span>
                          )}
                        </label>
                        <span className="text-sm font-mono text-white-muted">
                          {isPercentage
                            ? `${(value * 100).toFixed(0)}%`
                            : value}
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min={info.min}
                          max={info.max}
                          step={info.step}
                          value={value}
                          onChange={(e) =>
                            handleParamChange(
                              strategy.name,
                              key,
                              parseFloat(e.target.value)
                            )
                          }
                          className="flex-1 h-1.5 bg-black-card rounded-full appearance-none cursor-pointer accent-red-hot
                            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
                            [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-red-hot [&::-webkit-slider-thumb]:cursor-pointer
                            [&::-webkit-slider-thumb]:shadow-lg"
                        />
                        <input
                          type="number"
                          min={info.min}
                          max={info.max}
                          step={info.step}
                          value={value}
                          onChange={(e) =>
                            handleParamChange(
                              strategy.name,
                              key,
                              parseFloat(e.target.value) || 0
                            )
                          }
                          className="w-20 bg-black-deep border border-border-subtle rounded-lg px-2 py-1 text-sm font-mono text-white-full text-right
                            focus:outline-none focus:border-red-hot focus:ring-1 focus:ring-red-hot/30"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Save Button + Feedback */}
              <div className="px-5 pb-5">
                <button
                  onClick={() => handleSave(strategy)}
                  disabled={isSaving}
                  className="w-full flex items-center justify-center gap-2 bg-black-deep hover:bg-black-card
                    border border-border-subtle rounded-lg px-4 py-2.5 text-sm font-medium transition-colors
                    disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSaving ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4" />
                      Save {meta.label}
                    </>
                  )}
                </button>

                {feedback && (
                  <div
                    className={`mt-3 flex items-center gap-2 text-sm rounded-lg px-3 py-2 ${
                      feedback.type === "success"
                        ? "bg-red-hot/10 text-orange-accent border border-red-hot/20"
                        : "bg-red-hot/10 text-red-hot border border-red-hot/20"
                    }`}
                  >
                    {feedback.type === "success" ? (
                      <Check className="w-4 h-4 flex-shrink-0" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                    )}
                    {feedback.message}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Info Footer */}
      <div className="bg-black-card/50 rounded-xl border border-border-subtle p-5 text-sm text-white-dim">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-white-muted mb-1">
              How Strategies Map to Market Regimes
            </p>
            <p>
              The regime engine detects the current market state (trending,
              ranging, or breakout) and automatically activates the matching
              strategy. You can tune each strategy&apos;s parameters here, or disable
              strategies you don&apos;t want the system to use. Changes are saved to
              the database and picked up by the trading engine on the next cycle.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
