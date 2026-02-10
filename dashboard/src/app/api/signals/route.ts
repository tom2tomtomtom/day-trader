import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";
import { promises as fs } from "fs";
import path from "path";
import { DATA_DIR } from "@/lib/data-dir";

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const { data, error } = await supabase
        .from("signals")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(50);

      if (!error && data?.length) {
        return NextResponse.json({
          signals: data.map((s) => ({
            symbol: s.symbol,
            action: s.action,
            score: s.score,
            confidence: s.confidence,
            reasons: s.reasons,
            regime: s.regime,
            ml_quality_score: s.ml_quality_score,
            ml_size_multiplier: s.ml_size_multiplier,
            timestamp: s.created_at,
          })),
          source: "supabase",
        });
      }
    }

    // Fallback
    const signalsPath = path.join(DATA_DIR, "combined_signals.json");
    const data = await fs.readFile(signalsPath, "utf-8");
    return NextResponse.json(JSON.parse(data));
  } catch (error) {
    console.error("Signals API error:", error);
    return NextResponse.json({ error: "Failed to fetch signals" }, { status: 500 });
  }
}
