import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";
import { promises as fs } from "fs";
import path from "path";
import { DATA_DIR } from "@/lib/data-dir";

export async function GET() {
  try {
    // Try Supabase first
    if (isSupabaseConfigured()) {
      const [portfolioRes, positionsRes] = await Promise.all([
        supabase
          .from("portfolio_state")
          .select("*")
          .order("snapshot_at", { ascending: false })
          .limit(1),
        supabase.from("positions").select("*").eq("status", "open"),
      ]);

      if (portfolioRes.data?.length) {
        const p = portfolioRes.data[0];
        const positions = positionsRes.data || [];
        return NextResponse.json({
          portfolio_value: p.portfolio_value,
          cash: p.cash,
          day_pnl: p.portfolio_value - 100000,
          day_pnl_pct: ((p.portfolio_value - 100000) / 100000) * 100,
          total_trades: p.total_trades,
          winners: p.winning_trades,
          losers: p.losing_trades,
          win_rate: p.win_rate,
          profit_factor: p.profit_factor,
          max_drawdown_pct: p.max_drawdown_pct,
          open_positions: positions.length,
          positions: Object.fromEntries(
            positions.map((pos) => [
              pos.symbol,
              {
                symbol: pos.symbol,
                direction: pos.direction,
                shares: pos.shares,
                entry_price: pos.entry_price,
                current_price: pos.current_price,
                pnl: pos.unrealized_pnl,
                pnl_pct: pos.unrealized_pnl_pct,
                stop_loss: pos.stop_loss,
                take_profit: pos.take_profit,
              },
            ])
          ),
          source: "supabase",
        });
      }
    }

    // Fallback to JSON
    const positionsPath = path.join(DATA_DIR, "day_positions.json");
    const data = await fs.readFile(positionsPath, "utf-8");
    const positions = JSON.parse(data);

    let portfolioValue = positions.cash;
    for (const pos of Object.values(
      positions.positions as Record<
        string,
        { shares: number; entry_price: number }
      >
    )) {
      portfolioValue += pos.shares * pos.entry_price;
    }

    return NextResponse.json({
      portfolio_value: portfolioValue,
      cash: positions.cash,
      day_pnl: portfolioValue - 100000,
      day_pnl_pct: ((portfolioValue - 100000) / 100000) * 100,
      total_trades: positions.total_trades,
      winners: positions.winners,
      losers: positions.losers,
      open_positions: Object.keys(positions.positions).length,
      positions: positions.positions,
      source: "json",
    });
  } catch (error) {
    console.error("Status API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch status" },
      { status: 500 }
    );
  }
}
