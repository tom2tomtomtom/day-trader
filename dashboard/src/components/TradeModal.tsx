"use client";

import { useState } from "react";
import { X, TrendingUp, TrendingDown, AlertTriangle, Loader2 } from "lucide-react";

interface TradeModalProps {
  symbol: string;
  price: number;
  suggestedShares?: number;
  suggestedDirection?: "LONG" | "SHORT";
  reason?: string;
  onClose: () => void;
  onSuccess?: () => void;
}

export function TradeModal({
  symbol,
  price,
  suggestedShares = 100,
  suggestedDirection = "LONG",
  reason,
  onClose,
  onSuccess
}: TradeModalProps) {
  const [direction, setDirection] = useState<"LONG" | "SHORT">(suggestedDirection);
  const [shares, setShares] = useState(suggestedShares);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const estimatedCost = shares * price;
  const stopLoss = direction === "LONG"
    ? price * 0.98  // 2% stop for longs
    : price * 1.02; // 2% stop for shorts
  const target = direction === "LONG"
    ? price * 1.03  // 3% target for longs
    : price * 0.97; // 3% target for shorts

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/trade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "open",
          symbol,
          direction,
          shares,
          price,
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Trade failed");
      }

      onSuccess?.();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Trade failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-zinc-900 rounded-xl border border-zinc-700 w-full max-w-md shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            <span className="text-xl font-bold">{symbol}</span>
            <span className="text-zinc-400">${price.toFixed(2)}</span>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-zinc-400 hover:text-white rounded-lg hover:bg-zinc-800"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-4 space-y-4">
          {/* Reason if provided */}
          {reason && (
            <div className="text-sm text-zinc-400 bg-zinc-800/50 rounded-lg p-3">
              {reason}
            </div>
          )}

          {/* Direction Toggle */}
          <div>
            <label className="text-sm text-zinc-400 mb-2 block">Direction</label>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setDirection("LONG")}
                className={`flex items-center justify-center gap-2 py-3 rounded-lg font-medium transition-colors ${
                  direction === "LONG"
                    ? "bg-emerald-600 text-white"
                    : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                }`}
              >
                <TrendingUp className="w-5 h-5" />
                Long
              </button>
              <button
                onClick={() => setDirection("SHORT")}
                className={`flex items-center justify-center gap-2 py-3 rounded-lg font-medium transition-colors ${
                  direction === "SHORT"
                    ? "bg-red-600 text-white"
                    : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                }`}
              >
                <TrendingDown className="w-5 h-5" />
                Short
              </button>
            </div>
          </div>

          {/* Shares Input */}
          <div>
            <label className="text-sm text-zinc-400 mb-2 block">Shares</label>
            <div className="flex gap-2">
              <input
                type="number"
                value={shares}
                onChange={(e) => setShares(Math.max(1, parseInt(e.target.value) || 0))}
                className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 text-lg font-medium focus:outline-none focus:border-emerald-500"
              />
              <div className="flex flex-col gap-1">
                <button
                  onClick={() => setShares(Math.round(shares * 0.5))}
                  className="px-3 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 rounded"
                >
                  ½
                </button>
                <button
                  onClick={() => setShares(Math.round(shares * 2))}
                  className="px-3 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 rounded"
                >
                  2x
                </button>
              </div>
            </div>
          </div>

          {/* Order Summary */}
          <div className="bg-zinc-800 rounded-lg p-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-zinc-400">Estimated Cost</span>
              <span className="font-medium">${estimatedCost.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-zinc-400">🛑 Stop Loss (2%)</span>
              <span className="text-red-400">${stopLoss.toFixed(2)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-zinc-400">🎯 Target (3%)</span>
              <span className="text-emerald-400">${target.toFixed(2)}</span>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-center gap-2 text-red-400 text-sm bg-red-900/20 rounded-lg p-3">
              <AlertTriangle className="w-4 h-4" />
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-zinc-800 flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-3 rounded-lg bg-zinc-800 hover:bg-zinc-700 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={loading || shares <= 0}
            className={`flex-1 py-3 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 ${
              direction === "LONG"
                ? "bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-800"
                : "bg-red-600 hover:bg-red-700 disabled:bg-red-800"
            } disabled:opacity-50`}
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <>
                {direction === "LONG" ? "Buy" : "Sell Short"} {symbol}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
