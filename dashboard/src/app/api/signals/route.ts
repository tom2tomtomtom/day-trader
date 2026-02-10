import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    let signals: Record<string, unknown>[] = [];
    let fearGreed = 50;
    let vix = 20;
    let vixRegime = "Normal";

    if (isSupabaseConfigured()) {
      const [signalsRes, snapshotRes] = await Promise.all([
        supabase
          .from("signals")
          .select("*")
          .order("created_at", { ascending: false })
          .limit(50),
        supabase
          .from("market_snapshots")
          .select("fear_greed, vix")
          .order("created_at", { ascending: false })
          .limit(1),
      ]);

      if (snapshotRes.data?.[0]) {
        fearGreed = snapshotRes.data[0].fear_greed || 50;
        vix = snapshotRes.data[0].vix || 20;
        vixRegime = vix > 25 ? "High" : vix > 15 ? "Normal" : "Low";
      }

      signals = (signalsRes.data || []).map((s) => {
        const reasons =
          typeof s.reasons === "string"
            ? JSON.parse(s.reasons || "[]")
            : s.reasons || [];
        // Map Supabase score (-100 to +100) to signal_score (-1 to +1)
        const normalizedScore = (s.score || 0) / 100;
        return {
          symbol: s.symbol,
          price: 0,
          signal_score: normalizedScore,
          action: s.action || "HOLD",
          position_size: 0,
          reasons,
          indicators: {
            bollinger: { position: 0.5, width: 0 },
            rsi: { value: 50, is_oversold: false, is_overbought: false },
            volume: { relative_volume: 1, is_confirming: false },
            trend: { trend: "neutral", strength: 0 },
          },
        };
      });
    }

    const fearGreedLabel =
      fearGreed >= 75
        ? "Extreme Greed"
        : fearGreed >= 55
          ? "Greed"
          : fearGreed >= 45
            ? "Neutral"
            : fearGreed >= 25
              ? "Fear"
              : "Extreme Fear";

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      market_context: {
        fear_greed: fearGreed,
        fear_greed_label: fearGreedLabel,
        vix,
        vix_regime: vixRegime,
      },
      signals,
      source: "supabase",
    });
  } catch (error) {
    console.error("Signals API error:", error);
    return NextResponse.json({
      timestamp: new Date().toISOString(),
      market_context: {
        fear_greed: 50,
        fear_greed_label: "Neutral",
        vix: 20,
        vix_regime: "Normal",
      },
      signals: [],
    });
  }
}
