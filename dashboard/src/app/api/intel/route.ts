import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";
import { readFileSync } from "fs";
import { join } from "path";

const EMPTY_INTEL = {
  timestamp: new Date().toISOString(),
  market: {
    regime: "UNKNOWN",
    regime_score: 50,
    fear_greed: 50,
    vix: 0,
    risk_score: 50,
    opportunity_score: 50,
  },
  triggers: [],
  critical_triggers: 0,
  digest: {
    headline: "Engine Running — Awaiting First Intelligence Briefing",
    mood: "The trading engine is scanning markets. Intelligence briefings run at 9:00 AM and 4:30 PM ET.",
    regime_narrative: "No regime data available yet.",
    risk_warnings: [],
    smart_money: "No smart money signals yet.",
    closing_thought:
      "Signals and positions are being tracked. Full intel reports will appear after the next briefing cycle.",
  },
  opportunities: [],
  congress: {
    total_trades: 0,
    signals: 0,
    cluster_buys: 0,
    hot_symbols: [],
    notable_activity: [],
  },
  stats: {
    symbols_analyzed: 0,
    actionable_signals: 0,
  },
  source: "none",
};

export async function GET() {
  try {
    // Try Supabase intelligence_briefings table first
    if (isSupabaseConfigured()) {
      const { data, error } = await supabase
        .from("intelligence_briefings")
        .select("briefing_data, created_at")
        .order("created_at", { ascending: false })
        .limit(1);

      if (!error && data?.[0]) {
        const briefing =
          typeof data[0].briefing_data === "string"
            ? JSON.parse(data[0].briefing_data)
            : data[0].briefing_data;
        return NextResponse.json({
          ...briefing,
          source: "supabase",
          last_updated: data[0].created_at,
        });
      }
    }

    // Fall back to local JSON file
    const dataDir = process.env.TRADING_DATA_DIR;
    if (dataDir) {
      try {
        const filePath = join(dataDir, "intelligence_report.json");
        const raw = readFileSync(filePath, "utf-8");
        const briefing = JSON.parse(raw);
        return NextResponse.json({
          ...briefing,
          source: "file",
        });
      } catch {
        // File doesn't exist yet
      }
    }

    return NextResponse.json(EMPTY_INTEL);
  } catch (error) {
    console.error("Intel API error:", error);
    return NextResponse.json(EMPTY_INTEL);
  }
}
