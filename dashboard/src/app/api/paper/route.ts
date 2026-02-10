import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";
import { promises as fs } from "fs";
import path from "path";
import { DATA_DIR } from "@/lib/data-dir";

export async function GET() {
  try {
    // Try Supabase first
    if (isSupabaseConfigured()) {
      const [portfolioRes, positionsRes, tradesRes, equityRes] =
        await Promise.all([
          supabase
            .from("portfolio_state")
            .select("*")
            .order("snapshot_at", { ascending: false })
            .limit(1),
          supabase.from("positions").select("*").eq("status", "open"),
          supabase
            .from("trades")
            .select("*")
            .eq("is_backtest", false)
            .order("exit_date", { ascending: false })
            .limit(20),
          supabase
            .from("equity_curve")
            .select("*")
            .order("recorded_at", { ascending: false })
            .limit(100),
        ]);

      if (portfolioRes.data?.length) {
        const p = portfolioRes.data[0];
        return NextResponse.json({
          timestamp: p.snapshot_at,
          portfolio: {
            initial_capital: 100000,
            current_value: p.portfolio_value,
            cash: p.cash,
            total_return_pct: p.total_return_pct,
            daily_pnl: 0,
            max_drawdown_pct: p.max_drawdown_pct,
          },
          stats: {
            total_trades: p.total_trades,
            win_rate: p.win_rate,
            avg_win_pct: 0,
            avg_loss_pct: 0,
            profit_factor: p.profit_factor,
          },
          positions: (positionsRes.data || []).map((pos) => ({
            symbol: pos.symbol,
            direction: pos.direction,
            entry_price: pos.entry_price,
            current_price: pos.current_price,
            shares: pos.shares,
            stop_loss: pos.stop_loss,
            take_profit: pos.take_profit,
            entry_date: pos.entry_date,
            entry_score: pos.entry_score,
            unrealized_pnl: pos.unrealized_pnl,
            unrealized_pnl_pct: pos.unrealized_pnl_pct,
          })),
          recent_trades: (tradesRes.data || []).map((t) => ({
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
          equity_curve: (equityRes.data || [])
            .reverse()
            .map((e) => ({
              date: e.recorded_at,
              value: e.portfolio_value,
              cash: e.cash,
              positions: e.positions_value,
            })),
          source: "supabase",
        });
      }
    }

    // Fallback to JSON
    const paperPath = path.join(DATA_DIR, "paper_trades.json");
    const data = await fs.readFile(paperPath, "utf-8");
    return NextResponse.json({ ...JSON.parse(data), source: "json" });
  } catch (error) {
    console.error("Paper trader API error:", error);
    return NextResponse.json(
      { error: "No paper trading data. Run: python3 -m core.paper_trader" },
      { status: 500 }
    );
  }
}
