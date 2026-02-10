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
    regimeColor: "text-emerald-400 bg-emerald-600/20",
    icon: <TrendingUp className="w-5 h-5" />,
    gradient: "from-emerald-900/30 to-zinc-900",
  },
  mean_reversion: {
    label: "Mean Reversion",
    description:
      "Bets on price returning to the mean when it moves too far from Bollinger Bands. Buys at the lower band and sells at the upper band.",
    regime: "Ranging",
    regimeColor: "text-blue-400 bg-blue-600/20",
    icon: <BarChart3 className="w-5 h-5" />,
    gradient: "from-blue-900/30 to-zinc-900",
  },
  breakout: {
    label: "Breakout",
    description:
      "Detects when price breaks above resistance levels after consolidation. Enters on confirmed breakouts with volume confirmation.",
    regime: "Breakout",
    regimeColor: "text-yellow-400 bg-yellow-600/20",
    icon: <Zap className="w-5 h-5" />,
    gradient: "from-yellow-900/30 to-zinc-900",
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
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Settings2 className="w-7 h-7 text-emerald-500" />
            Strategy Configuration
          </h1>
          <p className="text-zinc-400 text-sm mt-1">
            Configure trading strategy parameters and enable/disable strategies
          </p>
        </div>
        <div className="flex gap-3 text-sm">
          <div className="bg-zinc-800 px-3 py-2 rounded-lg">
            <span className="text-zinc-400">Source:</span>{" "}
            <span className="text-emerald-400 font-medium">{source}</span>
          </div>
          <div className="bg-zinc-800 px-3 py-2 rounded-lg">
            <span className="text-zinc-400">Active:</span>{" "}
            <span className="text-emerald-400 font-bold">
              {strategies.filter((s) => s.enabled).length}
            </span>
            <span className="text-zinc-500">/{strategies.length}</span>
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
            regimeColor: "text-zinc-400 bg-zinc-600/20",
            icon: <Settings2 className="w-5 h-5" />,
            gradient: "from-zinc-800 to-zinc-900",
          };
          const paramInfo = PARAMETER_INFO[strategy.name] || {};
          const isSaving = savingMap[strategy.name] || false;
          const feedback = feedbackMap[strategy.name];

          return (
            <div
              key={strategy.name}
              className={`bg-gradient-to-br ${meta.gradient} rounded-xl border ${
                strategy.enabled
                  ? "border-zinc-700"
                  : "border-zinc-800 opacity-60"
              } transition-all`}
            >
              {/* Card Header */}
              <div className="p-5 border-b border-zinc-800">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span
                      className={
                        strategy.enabled ? "text-zinc-200" : "text-zinc-500"
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
                      strategy.enabled ? "bg-emerald-600" : "bg-zinc-700"
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

                <p className="text-sm text-zinc-400 leading-relaxed">
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
                        <label className="text-sm font-medium text-zinc-300 flex items-center gap-1.5">
                          {info.label}
                          {info.tooltip && (
                            <span className="group relative">
                              <Info className="w-3.5 h-3.5 text-zinc-500 cursor-help" />
                              <span className="invisible group-hover:visible absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-48 p-2 text-xs bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-300 z-10 shadow-xl">
                                {info.tooltip}
                              </span>
                            </span>
                          )}
                        </label>
                        <span className="text-sm font-mono text-zinc-400">
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
                          className="flex-1 h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer accent-emerald-500
                            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
                            [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-emerald-500 [&::-webkit-slider-thumb]:cursor-pointer
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
                          className="w-20 bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1 text-sm font-mono text-zinc-200 text-right
                            focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/30"
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
                  className="w-full flex items-center justify-center gap-2 bg-zinc-800 hover:bg-zinc-700
                    border border-zinc-700 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors
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
                        ? "bg-emerald-600/10 text-emerald-400 border border-emerald-600/20"
                        : "bg-red-600/10 text-red-400 border border-red-600/20"
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
      <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-5 text-sm text-zinc-500">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-zinc-400 mb-1">
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
