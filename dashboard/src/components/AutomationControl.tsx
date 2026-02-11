"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Play, Pause, Square, ShieldCheck,
  RefreshCw, CheckCircle, XCircle,
  AlertTriangle, Zap
} from "lucide-react";
import { useRealtimeSubscription } from "@/hooks/useRealtimeSubscription";

interface PendingTrade {
  id: string;
  symbol: string;
  action: string;
  price: number;
  reason: string;
  queued_at: string;
}

interface AutomationStatus {
  mode: "FULL_AUTO" | "APPROVAL" | "PAUSED" | "STOPPED";
  active_markets: string[];
  pending_trades: PendingTrade[];
  pending_count: number;
  open_positions: number;
  trades_today: number;
  config: {
    max_daily_trades: number;
    max_daily_loss_pct: number;
  };
}

const MODE_CONFIG = {
  FULL_AUTO: {
    icon: Zap,
    label: "Full Auto",
    description: "Scanning and trading automatically",
    activeClass: "bg-red-hot text-white",
    iconClass: "text-red-hot",
    bgClass: "bg-red-hot/20",
  },
  APPROVAL: {
    icon: ShieldCheck,
    label: "Approval Mode",
    description: "Trades queued for your approval",
    activeClass: "bg-orange-accent text-white",
    iconClass: "text-orange-accent",
    bgClass: "bg-orange-accent/20",
  },
  PAUSED: {
    icon: Pause,
    label: "Paused",
    description: "Managing existing positions only",
    activeClass: "bg-white-dim text-white",
    iconClass: "text-white-dim",
    bgClass: "bg-white-faint",
  },
  STOPPED: {
    icon: Square,
    label: "Stopped",
    description: "All automation disabled",
    activeClass: "bg-red-dim text-white",
    iconClass: "text-red-dim",
    bgClass: "bg-red-dim/20",
  },
};

export function AutomationControl() {
  const [status, setStatus] = useState<AutomationStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/automation");
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (error) {
      console.error("Failed to fetch automation status:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 120000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Real-time: refetch when positions or pending trades change
  useRealtimeSubscription([
    { table: "positions", onchange: fetchStatus },
    { table: "pending_trades", onchange: fetchStatus },
    { table: "portfolio_state", onchange: fetchStatus },
  ]);

  const setMode = async (mode: string) => {
    setActionLoading(`mode-${mode}`);
    try {
      const res = await fetch("/api/automation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "set_mode", mode }),
      });
      if (res.ok) {
        await fetchStatus();
      }
    } finally {
      setActionLoading(null);
    }
  };

  const runScan = async () => {
    setActionLoading("scan");
    try {
      await fetch("/api/automation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "run_scan" }),
      });
      await fetchStatus();
    } finally {
      setActionLoading(null);
    }
  };

  const runTradeCycle = async () => {
    setActionLoading("trade");
    try {
      await fetch("/api/automation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "run_trade_cycle" }),
      });
      await fetchStatus();
    } finally {
      setActionLoading(null);
    }
  };

  const approveTrade = async (tradeId: string) => {
    setActionLoading(`approve-${tradeId}`);
    try {
      await fetch("/api/automation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "approve_trade", tradeId }),
      });
      await fetchStatus();
    } finally {
      setActionLoading(null);
    }
  };

  const rejectTrade = async (tradeId: string) => {
    setActionLoading(`reject-${tradeId}`);
    try {
      await fetch("/api/automation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "reject_trade", tradeId }),
      });
      await fetchStatus();
    } finally {
      setActionLoading(null);
    }
  };

  if (loading || !status) {
    return (
      <div className="bg-black-card p-6 border border-border-subtle animate-pulse">
        <div className="h-6 bg-black-deep w-1/3 mb-4"></div>
        <div className="h-10 bg-black-deep"></div>
      </div>
    );
  }

  const currentMode = MODE_CONFIG[status.mode];
  const ModeIcon = currentMode.icon;

  return (
    <div className="space-y-4">
      {/* Main Control Panel */}
      <div className="bg-black-card p-6 border border-border-subtle">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className={`p-2 ${currentMode.bgClass}`}>
              <ModeIcon className={`w-6 h-6 ${currentMode.iconClass}`} />
            </div>
            <div>
              <h2 className="font-semibold text-lg">{currentMode.label}</h2>
              <p className="text-sm text-white-muted">{currentMode.description}</p>
            </div>
          </div>

          {status.active_markets.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-red-hot animate-pulse"></span>
              <span className="text-sm text-white-muted">
                {status.active_markets.join(", ")} open
              </span>
            </div>
          )}
        </div>

        {/* Mode Buttons */}
        <div className="grid grid-cols-4 gap-2 mb-6">
          <button
            onClick={() => setMode("FULL_AUTO")}
            disabled={actionLoading !== null}
            className={`flex items-center justify-center gap-2 px-3 py-2 transition-colors ${
              status.mode === "FULL_AUTO"
                ? "bg-red-hot text-white"
                : "bg-black-deep hover:bg-black-card text-white-muted"
            }`}
          >
            <Zap className="w-4 h-4" />
            <span className="text-sm">Auto</span>
          </button>

          <button
            onClick={() => setMode("APPROVAL")}
            disabled={actionLoading !== null}
            className={`flex items-center justify-center gap-2 px-3 py-2 transition-colors ${
              status.mode === "APPROVAL"
                ? "bg-orange-accent text-white"
                : "bg-black-deep hover:bg-black-card text-white-muted"
            }`}
          >
            <ShieldCheck className="w-4 h-4" />
            <span className="text-sm">Approve</span>
          </button>

          <button
            onClick={() => setMode("PAUSED")}
            disabled={actionLoading !== null}
            className={`flex items-center justify-center gap-2 px-3 py-2 transition-colors ${
              status.mode === "PAUSED"
                ? "bg-white-dim text-white"
                : "bg-black-deep hover:bg-black-card text-white-muted"
            }`}
          >
            <Pause className="w-4 h-4" />
            <span className="text-sm">Pause</span>
          </button>

          <button
            onClick={() => setMode("STOPPED")}
            disabled={actionLoading !== null}
            className={`flex items-center justify-center gap-2 px-3 py-2 transition-colors ${
              status.mode === "STOPPED"
                ? "bg-red-dim text-white"
                : "bg-black-deep hover:bg-black-card text-white-muted"
            }`}
          >
            <Square className="w-4 h-4" />
            <span className="text-sm">Stop</span>
          </button>
        </div>

        {/* Manual Actions */}
        <div className="flex gap-2">
          <button
            onClick={runScan}
            disabled={actionLoading !== null}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-black-deep hover:bg-black-card transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${actionLoading === "scan" ? "animate-spin" : ""}`} />
            <span>Run Scan</span>
          </button>

          <button
            onClick={runTradeCycle}
            disabled={actionLoading !== null || status.mode === "STOPPED"}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-black-deep hover:bg-black-card disabled:opacity-50 transition-colors"
          >
            <Play className={`w-4 h-4 ${actionLoading === "trade" ? "animate-pulse" : ""}`} />
            <span>Run Trade Cycle</span>
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 mt-6 pt-4 border-t border-border-subtle">
          <div className="text-center">
            <div className="text-2xl font-bold">{status.trades_today}</div>
            <div className="text-xs text-white-muted">
              Trades Today / {status.config.max_daily_trades}
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{status.open_positions}</div>
            <div className="text-xs text-white-muted">Open Positions</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-accent">{status.pending_count}</div>
            <div className="text-xs text-white-muted">Pending Approval</div>
          </div>
        </div>
      </div>

      {/* Pending Trades */}
      {status.pending_trades.length > 0 && (
        <div className="bg-black-card border border-orange-accent/50">
          <div className="p-4 border-b border-border-subtle flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-orange-accent" />
            <h3 className="font-semibold">Pending Approval ({status.pending_trades.length})</h3>
          </div>
          <div className="divide-y divide-border-subtle">
            {status.pending_trades.map((trade) => (
              <div key={trade.id} className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <span className={`font-bold ${
                      trade.action === "LONG" || trade.action === "BUY"
                        ? "text-orange-accent"
                        : "text-red-hot"
                    }`}>
                      {trade.action}
                    </span>
                    <span className="ml-2 font-semibold">{trade.symbol}</span>
                    <span className="ml-2 text-white-muted">@ ${trade.price.toFixed(2)}</span>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => approveTrade(trade.id)}
                      disabled={actionLoading !== null}
                      className="flex items-center gap-1 px-3 py-1 bg-red-hot hover:bg-red-dim text-sm"
                    >
                      <CheckCircle className="w-4 h-4" />
                      Approve
                    </button>
                    <button
                      onClick={() => rejectTrade(trade.id)}
                      disabled={actionLoading !== null}
                      className="flex items-center gap-1 px-3 py-1 bg-red-dim hover:bg-red-hot text-sm"
                    >
                      <XCircle className="w-4 h-4" />
                      Reject
                    </button>
                  </div>
                </div>
                <div className="text-sm text-white-muted">{trade.reason}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
