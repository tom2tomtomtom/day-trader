import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";
import { promises as fs } from "fs";
import path from "path";
import { DATA_DIR } from "@/lib/data-dir";

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const { data, error } = await supabase
        .from("market_snapshots")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(1);

      if (!error && data?.length) {
        const s = data[0];
        return NextResponse.json({
          regime: s.regime,
          fear_greed: s.fear_greed,
          vix: s.vix,
          spy_change_pct: s.spy_change_pct,
          portfolio_value: s.portfolio_value,
          extra: s.extra,
          timestamp: s.created_at,
          source: "supabase",
        });
      }
    }

    // Fallback
    const [regimeState, currentState] = await Promise.all([
      fs.readFile(path.join(DATA_DIR, "regime_state.json"), "utf-8").catch(() => "{}"),
      fs.readFile(path.join(DATA_DIR, "current_state.json"), "utf-8").catch(() => "{}"),
    ]);

    return NextResponse.json({
      ...JSON.parse(regimeState),
      current_state: JSON.parse(currentState),
    });
  } catch (error) {
    console.error("Regime API error:", error);
    return NextResponse.json({ error: "Failed to fetch regime" }, { status: 500 });
  }
}
