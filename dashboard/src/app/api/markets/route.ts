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

// Index ETFs that represent each market region
const MARKET_INDEX_SYMBOLS: Record<string, string> = {
  US: "SPY",
  Europe: "FEZ",
  Japan: "EWJ",
  HongKong: "FXI",
  Australia: "EWA",
  Korea: "EWY",
};

interface RegionalInfo {
  regime: string;
  change_1d: number | null;
}

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

/**
 * Fetch 1D percentage change for a list of symbols from Yahoo Finance.
 * Returns a map of symbol -> change_pct (e.g., { SPY: 0.45, FEZ: -0.12 }).
 */
async function fetchYahoo1DChanges(
  symbols: string[]
): Promise<Record<string, number | null>> {
  const results: Record<string, number | null> = {};

  // Fetch all symbols in parallel
  const fetches = symbols.map(async (symbol) => {
    try {
      const endDate = Math.floor(Date.now() / 1000);
      // 7 days back to ensure we get at least 2 trading days
      const startDate = endDate - 7 * 86400;
      const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${startDate}&period2=${endDate}&interval=1d`;

      const response = await fetch(url, {
        headers: { "User-Agent": "Mozilla/5.0" },
        signal: AbortSignal.timeout(5000),
      });

      if (!response.ok) {
        results[symbol] = null;
        return;
      }

      const data = await response.json();
      const result = data.chart?.result?.[0];
      const closes = result?.indicators?.quote?.[0]?.close;

      if (closes && closes.length >= 2) {
        // Get last two valid closing prices
        const validCloses = closes.filter(
          (c: number | null) => c !== null && c > 0
        );
        if (validCloses.length >= 2) {
          const prev = validCloses[validCloses.length - 2];
          const curr = validCloses[validCloses.length - 1];
          results[symbol] = Number((((curr - prev) / prev) * 100).toFixed(2));
          return;
        }
      }

      results[symbol] = null;
    } catch {
      results[symbol] = null;
    }
  });

  await Promise.all(fetches);
  return results;
}

/**
 * Derive a simple regime label from 1D change percentage.
 * Positive > 0.5% = BULLISH, negative < -0.5% = BEARISH, else NEUTRAL.
 */
function deriveRegimeFromChange(
  changePct: number | null,
  globalRegime: string
): string {
  if (changePct === null) return globalRegime;
  if (changePct > 0.5) return "BULLISH";
  if (changePct < -0.5) return "BEARISH";
  return "NEUTRAL";
}

export async function GET() {
  try {
    const { status, activeMarkets } = getMarketStatus();

    // Try Supabase for regime data
    let globalRegime = "UNKNOWN";
    let spyChangePct: number | null = null;

    if (isSupabaseConfigured()) {
      const { data } = await supabase
        .from("market_snapshots")
        .select("regime, spy_change_pct, extra")
        .order("created_at", { ascending: false })
        .limit(1);

      if (data?.length) {
        globalRegime = data[0].regime || "UNKNOWN";
        spyChangePct =
          data[0].spy_change_pct !== null &&
          data[0].spy_change_pct !== undefined
            ? Number(data[0].spy_change_pct)
            : null;
      }
    } else {
      // Fallback to JSON
      try {
        const regimePath = path.join(DATA_DIR, "regime_state.json");
        const fileData = await fs.readFile(regimePath, "utf-8");
        const regime = JSON.parse(fileData);
        globalRegime =
          regime?.summary?.global ||
          regime?.regimes?.regional?.global_regime ||
          "MIXED";
      } catch {
        // File doesn't exist
      }
    }

    // Fetch 1D changes for all index ETFs from Yahoo Finance
    const symbols = Object.values(MARKET_INDEX_SYMBOLS);
    const yahooChanges = await fetchYahoo1DChanges(symbols);

    // Build per-region data
    const regional: Record<string, RegionalInfo> = {};

    for (const market of Object.keys(MARKET_SCHEDULES)) {
      const symbol = MARKET_INDEX_SYMBOLS[market];
      // Use Supabase spy_change_pct for US if available, otherwise use Yahoo
      let change1d: number | null = yahooChanges[symbol] ?? null;
      if (market === "US" && spyChangePct !== null) {
        change1d = spyChangePct;
      }

      regional[market] = {
        regime: deriveRegimeFromChange(change1d, globalRegime),
        change_1d: change1d,
      };
    }

    return NextResponse.json({
      market_status: status,
      active_markets: activeMarkets,
      global_regime: globalRegime,
      regional,
    });
  } catch (error) {
    console.error("Markets API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch market status" },
      { status: 500 }
    );
  }
}
