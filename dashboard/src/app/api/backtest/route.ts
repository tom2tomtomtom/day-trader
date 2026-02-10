import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const { data, error } = await supabase
        .from("trades")
        .select("*")
        .eq("is_backtest", true)
        .order("exit_date", { ascending: false })
        .limit(200);

      if (!error) {
        const trades = data || [];
        const wins = trades.filter((t) => t.pnl_pct > 0);
        const losses = trades.filter((t) => t.pnl_pct <= 0);
        return NextResponse.json({
          total_trades: trades.length,
          win_rate: trades.length > 0 ? (wins.length / trades.length) * 100 : 0,
          avg_win_pct: wins.length > 0 ? wins.reduce((s, t) => s + t.pnl_pct, 0) / wins.length : 0,
          avg_loss_pct: losses.length > 0 ? losses.reduce((s, t) => s + t.pnl_pct, 0) / losses.length : 0,
          trades: trades.map((t) => ({
            symbol: t.symbol,
            direction: t.direction,
            entry_price: t.entry_price,
            exit_price: t.exit_price,
            pnl_pct: t.pnl_pct,
            pnl_dollars: t.pnl_dollars,
            exit_reason: t.exit_reason,
            entry_date: t.entry_date,
            exit_date: t.exit_date,
          })),
          source: "supabase",
        });
      }
    }

    return NextResponse.json({
      total_trades: 0,
      win_rate: 0,
      trades: [],
      message: "No backtest data yet. Run backtester to generate historical trades.",
    });
  } catch (error) {
    console.error("Backtest API error:", error);
    return NextResponse.json({
      total_trades: 0,
      win_rate: 0,
      trades: [],
    });
  }
}
