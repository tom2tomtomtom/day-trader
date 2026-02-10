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

        const congress = briefing.congress || {};

        // Extract congressional data from opportunities
        const trades = (briefing.opportunities || [])
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .filter((opp: any) => opp.congress_buying > 0 || opp.congress_selling > 0)
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .map((opp: any) => ({
            symbol: opp.symbol,
            members_buying: opp.congress_buying || 0,
            members_selling: opp.congress_selling || 0,
            conviction: opp.congress_conviction || 0,
            notable: opp.congress_notable || [],
          }));

        return NextResponse.json({
          timestamp: briefing.timestamp || data[0].created_at,
          trades,
          signals: congress,
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
        const congress = briefing.congress || {};
        const trades = (briefing.opportunities || [])
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .filter((opp: any) => opp.congress_buying > 0 || opp.congress_selling > 0)
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .map((opp: any) => ({
            symbol: opp.symbol,
            members_buying: opp.congress_buying || 0,
            members_selling: opp.congress_selling || 0,
            conviction: opp.congress_conviction || 0,
            notable: opp.congress_notable || [],
          }));
        return NextResponse.json({
          timestamp: briefing.timestamp,
          trades,
          signals: congress,
          source: "file",
        });
      } catch {
        // File doesn't exist yet
      }
    }

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      trades: [],
      signals: null,
      source: "none",
      message:
        "Congressional intelligence runs during briefings (9:00 AM and 4:30 PM ET). Data will populate after the next cycle.",
    });
  } catch (error) {
    console.error("Congress API error:", error);
    return NextResponse.json({
      timestamp: new Date().toISOString(),
      trades: [],
      signals: null,
      source: "error",
      message: "Failed to fetch congressional data.",
    });
  }
}
