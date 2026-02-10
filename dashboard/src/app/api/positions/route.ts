import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";
import { promises as fs } from "fs";
import path from "path";
import { DATA_DIR } from "@/lib/data-dir";

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const { data, error } = await supabase
        .from("positions")
        .select("*")
        .eq("status", "open");

      if (!error && data) {
        return NextResponse.json({
          positions: Object.fromEntries(
            data.map((pos) => [
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
          closed_trades: [],
          total_trades: data.length,
          source: "supabase",
        });
      }
    }

    // Fallback
    const positionsPath = path.join(DATA_DIR, "day_positions.json");
    const data = await fs.readFile(positionsPath, "utf-8");
    return NextResponse.json(JSON.parse(data));
  } catch (error) {
    console.error("Positions API error:", error);
    return NextResponse.json({
      positions: {},
      closed_trades: [],
      total_trades: 0,
    });
  }
}
