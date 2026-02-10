import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const { data } = await supabase
        .from("trades")
        .select("*")
        .eq("is_backtest", true)
        .order("exit_date", { ascending: false })
        .limit(200);

      const trades = data || [];

      if (trades.length === 0) {
        // Return 404-like so the page shows "No Backtest Results" state
        return NextResponse.json(
          { error: "No backtest data yet. Run: python3 -m core.backtester --symbol SPY" },
          { status: 404 }
        );
      }

      const wins = trades.filter((t) => t.pnl_pct > 0);
      const losses = trades.filter((t) => t.pnl_pct <= 0);
      const totalPnl = trades.reduce((s, t) => s + t.pnl_pct, 0);

      return NextResponse.json({
        timestamp: trades[0]?.exit_date || new Date().toISOString(),
        strategy: "Ensemble ML",
        symbols: [...new Set(trades.map((t) => t.symbol))],
        start_date: trades[trades.length - 1]?.entry_date || "",
        end_date: trades[0]?.exit_date || "",
        initial_capital: 100000,
        final_value: 100000 * (1 + totalPnl / 100),
        metrics: {
          total_return_pct: totalPnl,
          annualized_return_pct: 0,
          max_drawdown_pct: 0,
          max_drawdown_duration_days: 0,
          volatility_annualized: 0,
          sharpe_ratio: 0,
          sortino_ratio: 0,
          calmar_ratio: 0,
          total_trades: trades.length,
          winning_trades: wins.length,
          losing_trades: losses.length,
          win_rate: (wins.length / trades.length) * 100,
          avg_win_pct: wins.length > 0 ? wins.reduce((s, t) => s + t.pnl_pct, 0) / wins.length : 0,
          avg_loss_pct: losses.length > 0 ? losses.reduce((s, t) => s + t.pnl_pct, 0) / losses.length : 0,
          largest_win_pct: wins.length > 0 ? Math.max(...wins.map((t) => t.pnl_pct)) : 0,
          largest_loss_pct: losses.length > 0 ? Math.min(...losses.map((t) => t.pnl_pct)) : 0,
          avg_hold_days: 0,
          profit_factor: 0,
          expectancy: 0,
        },
        trades: trades.map((t) => ({
          symbol: t.symbol,
          direction: t.direction,
          entry_date: t.entry_date,
          entry_price: t.entry_price,
          exit_date: t.exit_date,
          exit_price: t.exit_price,
          shares: t.shares,
          pnl_dollars: t.pnl_dollars,
          return_pct: t.pnl_pct,
          hold_days: 0,
          exit_reason: t.exit_reason,
          entry_score: t.entry_score || 0,
        })),
        equity_curve: [],
        parameters: {},
        source: "supabase",
      });
    }

    return NextResponse.json(
      { error: "No backtest data available" },
      { status: 404 }
    );
  } catch (error) {
    console.error("Backtest API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch backtest data" },
      { status: 500 }
    );
  }
}
