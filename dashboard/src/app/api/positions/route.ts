import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const [posRes, tradesRes] = await Promise.all([
        supabase.from("positions").select("*").eq("status", "open"),
        supabase
          .from("trades")
          .select("*")
          .eq("is_backtest", false)
          .order("exit_date", { ascending: false })
          .limit(50),
      ]);

      const positions = posRes.data || [];
      const trades = tradesRes.data || [];
      const winners = trades.filter((t) => (t.pnl_dollars || 0) > 0).length;
      const losers = trades.filter((t) => (t.pnl_dollars || 0) < 0).length;
      const grossPnl = trades.reduce((sum, t) => sum + (t.pnl_dollars || 0), 0);

      return NextResponse.json({
        positions: Object.fromEntries(
          positions.map((pos) => [
            pos.symbol,
            {
              symbol: pos.symbol,
              direction: pos.direction,
              shares: pos.shares,
              entry_price: pos.entry_price,
              entry_time: pos.entry_date,
              current_price: pos.current_price,
              stop_price: pos.stop_loss,
              target_price: pos.take_profit,
              pnl: pos.unrealized_pnl,
              pnl_pct: pos.unrealized_pnl_pct,
            },
          ])
        ),
        closed_trades: trades.map((t) => ({
          symbol: t.symbol,
          direction: t.direction,
          shares: t.shares,
          entry_price: t.entry_price,
          entry_time: t.entry_date,
          exit_price: t.exit_price,
          exit_time: t.exit_date,
          exit_reason: t.exit_reason || "unknown",
          pnl: t.pnl_dollars || 0,
          pnl_pct: t.pnl_pct || 0,
        })),
        total_trades: trades.length,
        winners,
        losers,
        gross_pnl: grossPnl,
        source: "supabase",
      });
    }

    return NextResponse.json({
      positions: {},
      closed_trades: [],
      total_trades: 0,
      winners: 0,
      losers: 0,
      gross_pnl: 0,
      source: "none",
    });
  } catch (error) {
    console.error("Positions API error:", error);
    return NextResponse.json({
      positions: {},
      closed_trades: [],
      total_trades: 0,
      winners: 0,
      losers: 0,
      gross_pnl: 0,
    });
  }
}
