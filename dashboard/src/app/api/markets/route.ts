import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";
import { promises as fs } from "fs";
import path from "path";
import { DATA_DIR } from "@/lib/data-dir";

// Market schedules (UTC hours)
const MARKET_SCHEDULES: Record<string, { open: number; close: number }> = {
  US: { open: 14, close: 21 },
  Europe: { open: 8, close: 16 },
  Japan: { open: 0, close: 6 },
  HongKong: { open: 1, close: 8 },
  Australia: { open: 23, close: 5 },
  Korea: { open: 0, close: 6 },
};

function getMarketStatus() {
  const now = new Date();
  const hour = now.getUTCHours();
  const weekday = now.getUTCDay();

  const status: Record<string, { open: boolean; hours: string }> = {};
  const activeMarkets: string[] = [];

  for (const [market, schedule] of Object.entries(MARKET_SCHEDULES)) {
    let isOpen = false;

    if (schedule.open > schedule.close) {
      isOpen = hour >= schedule.open || hour < schedule.close;
    } else {
      isOpen = hour >= schedule.open && hour < schedule.close;
    }

    if (weekday === 0 || weekday === 6) {
      isOpen = false;
    }

    status[market] = {
      open: isOpen,
      hours: `${schedule.open.toString().padStart(2, "0")}:00-${schedule.close.toString().padStart(2, "0")}:00 UTC`,
    };

    if (isOpen) {
      activeMarkets.push(market);
    }
  }

  return { status, activeMarkets };
}

export async function GET() {
  try {
    const { status, activeMarkets } = getMarketStatus();

    // Try Supabase for regime data
    let globalRegime = "UNKNOWN";
    if (isSupabaseConfigured()) {
      const { data } = await supabase
        .from("market_snapshots")
        .select("regime")
        .order("created_at", { ascending: false })
        .limit(1);

      if (data?.length) {
        globalRegime = data[0].regime || "UNKNOWN";
      }
    } else {
      // Fallback to JSON
      try {
        const regimePath = path.join(DATA_DIR, "regime_state.json");
        const data = await fs.readFile(regimePath, "utf-8");
        const regime = JSON.parse(data);
        globalRegime =
          regime?.summary?.global ||
          regime?.regimes?.regional?.global_regime ||
          "MIXED";
      } catch {
        // File doesn't exist
      }
    }

    return NextResponse.json({
      market_status: status,
      active_markets: activeMarkets,
      global_regime: globalRegime,
    });
  } catch (error) {
    console.error("Markets API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch market status" },
      { status: 500 }
    );
  }
}
