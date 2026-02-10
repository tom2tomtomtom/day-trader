import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    let edgeOpportunities: Record<string, unknown>[] = [];

    if (isSupabaseConfigured()) {
      const { data } = await supabase
        .from("signals")
        .select("*")
        .gte("score", 50)
        .order("created_at", { ascending: false })
        .limit(20);

      edgeOpportunities = (data || []).map((s) => ({
        symbol: s.symbol,
        edge_score: s.score,
        reasons: typeof s.reasons === "string" ? JSON.parse(s.reasons) : s.reasons || [],
      }));
    }

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      edge_opportunities: edgeOpportunities,
      wsb_momentum: [],
      wsb_trending: [],
      squeeze_setups: [],
      sector_rotation: {
        rotation_signal: "NEUTRAL",
        leaders: [],
        laggards: [],
        sectors: [],
      },
      summary: {
        total_scanned: 0,
        edge_opportunities: edgeOpportunities.length,
        squeeze_setups: 0,
        wsb_momentum: 0,
        sector_signal: "NEUTRAL",
      },
      source: "supabase",
    });
  } catch (error) {
    console.error("Edge API error:", error);
    return NextResponse.json({
      timestamp: new Date().toISOString(),
      edge_opportunities: [],
      wsb_momentum: [],
      wsb_trending: [],
      squeeze_setups: [],
      sector_rotation: { rotation_signal: "NEUTRAL", leaders: [], laggards: [], sectors: [] },
      summary: { total_scanned: 0, edge_opportunities: 0, squeeze_setups: 0, wsb_momentum: 0, sector_signal: "NEUTRAL" },
    });
  }
}
