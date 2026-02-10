import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    // Try Supabase first
    if (isSupabaseConfigured()) {
      const [portfolioRes, positionsRes, lastSignalRes] = await Promise.all([
        supabase
          .from("portfolio_state")
          .select("*")
          .order("snapshot_at", { ascending: false })
          .limit(1),
        supabase.from("positions").select("*").eq("status", "open"),
        supabase
          .from("signals")
          .select("created_at")
          .order("created_at", { ascending: false })
          .limit(1),
      ]);

      const p = portfolioRes.data?.[0];
      const positions = positionsRes.data || [];
      const pv = p?.portfolio_value || 100000;

      // Determine engine status from last snapshot time
      const lastScanAt = p?.snapshot_at || null;
      const lastSignalAt = lastSignalRes.data?.[0]?.created_at || null;
      const now = new Date();
      const scanAge = lastScanAt
        ? (now.getTime() - new Date(lastScanAt).getTime()) / 60000
        : Infinity;
      const engineStatus =
        scanAge < 20
          ? "active"
          : scanAge < 60
            ? "idle"
            : "stale";

      return NextResponse.json({
        portfolio_value: pv,
        cash: p?.cash || 100000,
        day_pnl: pv - 100000,
        day_pnl_pct: ((pv - 100000) / 100000) * 100,
        total_trades: p?.total_trades || 0,
        winners: p?.winning_trades || 0,
        losers: p?.losing_trades || 0,
        win_rate: p?.win_rate || 0,
        profit_factor: p?.profit_factor || 0,
        max_drawdown_pct: p?.max_drawdown_pct || 0,
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
        engine: {
          status: engineStatus,
          last_scan: lastScanAt,
          last_signal: lastSignalAt,
          scan_age_min: Math.round(scanAge),
        },
        source: "supabase",
        last_updated: lastScanAt,
      });
    }

    return NextResponse.json({
      portfolio_value: 100000,
      cash: 100000,
      day_pnl: 0,
      day_pnl_pct: 0,
      total_trades: 0,
      winners: 0,
      losers: 0,
      win_rate: 0,
      open_positions: 0,
      positions: {},
      engine: {
        status: "disconnected",
        last_scan: null,
        last_signal: null,
        scan_age_min: null,
      },
      source: "none",
    });
  } catch (error) {
    console.error("Status API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch status" },
      { status: 500 }
    );
  }
}
