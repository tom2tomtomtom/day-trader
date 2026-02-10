import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";
import { readFileSync } from "fs";
import { join } from "path";

export async function GET() {
  try {
    // Try Supabase intelligence_briefings table
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

        // Extract council data from opportunities
        const opportunities = (briefing.opportunities || []).map(
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (opp: any) => ({
            symbol: opp.symbol,
            action: opp.action,
            confidence: opp.confidence,
            opportunity_score: opp.opportunity_score,
            conviction_label: opp.conviction_label,
            council_action: opp.council_action,
            council_score: opp.council_score,
            council_conviction: opp.council_conviction,
            council_bulls: opp.council_bulls,
            council_bears: opp.council_bears,
            persona_verdicts: opp.persona_verdicts || [],
            headline: opp.headline,
            thesis: opp.thesis,
            bull_case: opp.bull_case,
            bear_case: opp.bear_case,
            key_drivers: opp.key_drivers || [],
            key_risks: opp.key_risks || [],
          })
        );

        return NextResponse.json({
          timestamp: briefing.timestamp || data[0].created_at,
          market: briefing.market || {},
          digest: briefing.digest || {},
          opportunities,
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
        const opportunities = (briefing.opportunities || []).map(
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (opp: any) => ({
            symbol: opp.symbol,
            action: opp.action,
            confidence: opp.confidence,
            opportunity_score: opp.opportunity_score,
            conviction_label: opp.conviction_label,
            council_action: opp.council_action,
            council_score: opp.council_score,
            council_conviction: opp.council_conviction,
            council_bulls: opp.council_bulls,
            council_bears: opp.council_bears,
            persona_verdicts: opp.persona_verdicts || [],
            headline: opp.headline,
            thesis: opp.thesis,
            bull_case: opp.bull_case,
            bear_case: opp.bear_case,
            key_drivers: opp.key_drivers || [],
            key_risks: opp.key_risks || [],
          })
        );
        return NextResponse.json({
          timestamp: briefing.timestamp,
          market: briefing.market || {},
          digest: briefing.digest || {},
          opportunities,
          source: "file",
        });
      } catch {
        // File doesn't exist yet
      }
    }

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      opportunities: [],
      source: "none",
      message:
        "Phantom Council convenes during intelligence briefings (9:00 AM and 4:30 PM ET). The engine is running and will populate this data.",
    });
  } catch (error) {
    console.error("Council API error:", error);
    return NextResponse.json({
      timestamp: new Date().toISOString(),
      opportunities: [],
      source: "error",
      message: "Failed to fetch council data.",
    });
  }
}
