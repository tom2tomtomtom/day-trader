import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const { data, error } = await supabase
        .from("market_snapshots")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(1);

      if (!error) {
        const s = data?.[0];
        return NextResponse.json({
          regime: s?.regime || "UNKNOWN",
          fear_greed: s?.fear_greed || 50,
          vix: s?.vix || 0,
          spy_change_pct: s?.spy_change_pct || 0,
          portfolio_value: s?.portfolio_value || 0,
          extra: s?.extra || {},
          timestamp: s?.created_at || new Date().toISOString(),
          source: "supabase",
        });
      }
    }

    return NextResponse.json({
      regime: "UNKNOWN",
      fear_greed: 50,
      vix: 0,
      source: "none",
    });
  } catch (error) {
    console.error("Regime API error:", error);
    return NextResponse.json({ error: "Failed to fetch regime" }, { status: 500 });
  }
}
