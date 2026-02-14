import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

interface Trade {
  symbol: string;
  direction: string;
  entry_price: number;
  exit_price: number;
  pnl_dollars: number;
  pnl_pct: number;
  exit_reason: string;
  entry_date: string;
  exit_date: string;
  regime_at_entry: string | null;
  shares: number;
}

interface SignalEvaluation {
  symbol: string;
  signal_type: string;
  predicted_direction: string;
  actual_direction: string;
  was_correct: boolean;
  evaluated_at: string;
  confidence: number;
}

export async function GET() {
  try {
    if (!isSupabaseConfigured()) {
      return NextResponse.json({
        win_rate_by_regime: [],
        pnl_by_exit_reason: [],
        feature_importance: [],
        signal_accuracy_over_time: [],
        drawdown_data: [],
        position_heat_map: [],
        error: "Supabase not configured",
      });
    }

    const [tradesRes, mlModelRes, signalEvalsRes] = await Promise.all([
      supabase
        .from("trades")
        .select(
          "symbol, direction, entry_price, exit_price, pnl_dollars, pnl_pct, exit_reason, entry_date, exit_date, regime_at_entry, shares"
        )
        .order("exit_date", { ascending: true })
        .limit(500),
      supabase
        .from("ml_models")
        .select("feature_importance, model_name, version, trained_at")
        .eq("is_active", true)
        .eq("model_name", "signal_quality")
        .limit(1),
      supabase
        .from("signal_evaluations")
        .select(
          "symbol, signal_type, predicted_direction, actual_direction, was_correct, evaluated_at, confidence"
        )
        .order("evaluated_at", { ascending: true })
        .limit(500),
    ]);

    const trades: Trade[] = tradesRes.data || [];
    const signalEvals: SignalEvaluation[] = signalEvalsRes.data || [];
    const mlModel = mlModelRes.data?.[0] || null;

    // --- Win Rate by Regime ---
    const regimeMap: Record<string, { wins: number; total: number }> = {};
    for (const t of trades) {
      const regime = t.regime_at_entry || "unknown";
      if (!regimeMap[regime]) regimeMap[regime] = { wins: 0, total: 0 };
      regimeMap[regime].total++;
      if (t.pnl_dollars > 0) regimeMap[regime].wins++;
    }
    const winRateByRegime = Object.entries(regimeMap).map(
      ([regime, stats]) => ({
        regime,
        win_rate: stats.total > 0 ? (stats.wins / stats.total) * 100 : 0,
        total_trades: stats.total,
        wins: stats.wins,
        losses: stats.total - stats.wins,
      })
    );

    // --- P&L by Exit Reason ---
    const exitReasonMap: Record<
      string,
      { total_pnl: number; count: number; avg_pnl: number }
    > = {};
    for (const t of trades) {
      const reason = t.exit_reason || "unknown";
      if (!exitReasonMap[reason])
        exitReasonMap[reason] = { total_pnl: 0, count: 0, avg_pnl: 0 };
      exitReasonMap[reason].total_pnl += t.pnl_dollars;
      exitReasonMap[reason].count++;
    }
    const pnlByExitReason = Object.entries(exitReasonMap).map(
      ([reason, stats]) => ({
        reason,
        total_pnl: parseFloat(stats.total_pnl.toFixed(2)),
        count: stats.count,
        avg_pnl: parseFloat((stats.total_pnl / stats.count).toFixed(2)),
      })
    );

    // --- Feature Importance ---
    const featureImportance: { feature: string; importance: number }[] = [];
    if (mlModel?.feature_importance) {
      const fi =
        typeof mlModel.feature_importance === "string"
          ? JSON.parse(mlModel.feature_importance)
          : mlModel.feature_importance;
      const sorted = Object.entries(fi as Record<string, number>)
        .sort(([, a], [, b]) => (b as number) - (a as number))
        .slice(0, 20);
      for (const [feature, importance] of sorted) {
        featureImportance.push({
          feature,
          importance: parseFloat(((importance as number) * 100).toFixed(2)),
        });
      }
    }

    // --- Signal Accuracy Over Time ---
    // Group by week
    const weekMap: Record<
      string,
      { correct: number; total: number; date: string }
    > = {};
    for (const ev of signalEvals) {
      if (!ev.evaluated_at) continue;
      const d = new Date(ev.evaluated_at);
      // Get ISO week start (Monday)
      const day = d.getDay();
      const diff = d.getDate() - day + (day === 0 ? -6 : 1);
      const weekStart = new Date(d.setDate(diff));
      const weekKey = weekStart.toISOString().split("T")[0];
      if (!weekMap[weekKey])
        weekMap[weekKey] = { correct: 0, total: 0, date: weekKey };
      weekMap[weekKey].total++;
      if (ev.was_correct) weekMap[weekKey].correct++;
    }
    const signalAccuracyOverTime = Object.values(weekMap)
      .map((w) => ({
        date: w.date,
        accuracy: w.total > 0 ? parseFloat(((w.correct / w.total) * 100).toFixed(1)) : 0,
        total_signals: w.total,
        correct_signals: w.correct,
      }))
      .sort((a, b) => a.date.localeCompare(b.date));

    // --- Drawdown Data ---
    // Compute running portfolio value and drawdown from trades
    let cumulativePnl = 0;
    let peakPnl = 0;
    const drawdownData: {
      date: string;
      cumulative_pnl: number;
      drawdown: number;
      drawdown_pct: number;
    }[] = [];
    const initialCapital = 100000; // Default starting capital

    for (const t of trades) {
      cumulativePnl += t.pnl_dollars;
      const portfolioValue = initialCapital + cumulativePnl;
      if (cumulativePnl > peakPnl) peakPnl = cumulativePnl;
      const peakValue = initialCapital + peakPnl;
      const drawdown = cumulativePnl - peakPnl;
      const drawdownPct =
        peakValue > 0 ? (drawdown / peakValue) * 100 : 0;

      drawdownData.push({
        date: t.exit_date,
        cumulative_pnl: parseFloat(cumulativePnl.toFixed(2)),
        drawdown: parseFloat(drawdown.toFixed(2)),
        drawdown_pct: parseFloat(drawdownPct.toFixed(2)),
      });
    }

    // --- Position Heat Map ---
    // Aggregate by symbol: count, avg P&L, total P&L
    const symbolMap: Record<
      string,
      { count: number; total_pnl: number; wins: number }
    > = {};
    for (const t of trades) {
      if (!symbolMap[t.symbol])
        symbolMap[t.symbol] = { count: 0, total_pnl: 0, wins: 0 };
      symbolMap[t.symbol].count++;
      symbolMap[t.symbol].total_pnl += t.pnl_dollars;
      if (t.pnl_dollars > 0) symbolMap[t.symbol].wins++;
    }
    const positionHeatMap = Object.entries(symbolMap)
      .map(([symbol, stats]) => ({
        symbol,
        trade_count: stats.count,
        total_pnl: parseFloat(stats.total_pnl.toFixed(2)),
        avg_pnl: parseFloat((stats.total_pnl / stats.count).toFixed(2)),
        win_rate: parseFloat(
          ((stats.wins / stats.count) * 100).toFixed(1)
        ),
      }))
      .sort((a, b) => b.trade_count - a.trade_count)
      .slice(0, 30);

    return NextResponse.json({
      win_rate_by_regime: winRateByRegime,
      pnl_by_exit_reason: pnlByExitReason,
      feature_importance: featureImportance,
      signal_accuracy_over_time: signalAccuracyOverTime,
      drawdown_data: drawdownData,
      position_heat_map: positionHeatMap,
    });
  } catch (error) {
    console.error("Analytics API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch analytics data" },
      { status: 500 }
    );
  }
}
