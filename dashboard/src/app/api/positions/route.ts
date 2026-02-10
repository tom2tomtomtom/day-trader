import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const { data, error } = await supabase
        .from("positions")
        .select("*")
        .eq("status", "open");

      const tradesRes = await supabase
        .from("trades")
        .select("*")
        .eq("is_backtest", false)
        .order("exit_date", { ascending: false })
        .limit(50);

      if (!error) {
        return NextResponse.json({
          positions: Object.fromEntries(
            (data || []).map((pos) => [
              pos.symbol,
              {
                symbol: pos.symbol,
                direction: pos.direction,
                shares: pos.shares,
                entry_price: pos.entry_price,
                current_price: pos.current_price,
                stop_loss: pos.stop_loss,
                take_profit: pos.take_profit,
                entry_date: pos.entry_date,
                unrealized_pnl: pos.unrealized_pnl,
                unrealized_pnl_pct: pos.unrealized_pnl_pct,
              },
            ])
          ),
          closed_trades: (tradesRes.data || []).map((t) => ({
            symbol: t.symbol,
            direction: t.direction,
            entry_price: t.entry_price,
            exit_price: t.exit_price,
            shares: t.shares,
            pnl_dollars: t.pnl_dollars,
            pnl_pct: t.pnl_pct,
            exit_reason: t.exit_reason,
            entry_date: t.entry_date,
            exit_date: t.exit_date,
          })),
          total_trades: (data || []).length + (tradesRes.data || []).length,
          source: "supabase",
        });
      }
    }

    return NextResponse.json({
      positions: {},
      closed_trades: [],
      total_trades: 0,
      source: "none",
    });
  } catch (error) {
    console.error("Positions API error:", error);
    return NextResponse.json({
      positions: {},
      closed_trades: [],
      total_trades: 0,
    });
  }
}
