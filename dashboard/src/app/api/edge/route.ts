import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      // Edge signals come from the signals table with high scores
      const { data, error } = await supabase
        .from("signals")
        .select("*")
        .gte("score", 50)
        .order("created_at", { ascending: false })
        .limit(20);

      if (!error) {
        return NextResponse.json({
          timestamp: new Date().toISOString(),
          signals: (data || []).map((s) => ({
            symbol: s.symbol,
            action: s.action,
            score: s.score,
            confidence: s.confidence,
            reasons: s.reasons,
            regime: s.regime,
          })),
          source: "supabase",
        });
      }
    }

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      signals: [],
      message: "No edge signals yet. The worker will generate them during market scans.",
    });
  } catch (error) {
    console.error("Edge API error:", error);
    return NextResponse.json({
      timestamp: new Date().toISOString(),
      signals: [],
    });
  }
}
