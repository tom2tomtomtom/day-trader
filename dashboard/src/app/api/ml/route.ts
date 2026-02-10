import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    if (!isSupabaseConfigured()) {
      return NextResponse.json({
        error: "Supabase not configured",
        model: null,
        history: [],
      });
    }

    const [activeModel, modelHistory, recentTrades] = await Promise.all([
      supabase
        .from("ml_models")
        .select("*")
        .eq("is_active", true)
        .eq("model_name", "signal_quality")
        .limit(1),
      supabase
        .from("ml_models")
        .select("*")
        .eq("model_name", "signal_quality")
        .order("trained_at", { ascending: false })
        .limit(10),
      supabase
        .from("trades")
        .select("pnl_pct, entry_features, exit_reason, symbol")
        .eq("is_backtest", false)
        .order("exit_date", { ascending: false })
        .limit(50),
    ]);

    const model = activeModel.data?.[0] || null;
    const history = modelHistory.data || [];

    // Compute prediction vs actual for recent trades
    const predictions = (recentTrades.data || []).map((t) => ({
      symbol: t.symbol,
      actual_pnl_pct: t.pnl_pct,
      profitable: t.pnl_pct > 0,
      exit_reason: t.exit_reason,
    }));

    return NextResponse.json({
      model: model
        ? {
            version: model.version,
            accuracy: model.accuracy,
            precision: model.precision_score,
            recall: model.recall,
            f1: model.f1,
            training_samples: model.training_samples,
            feature_importance: model.feature_importance,
            trained_at: model.trained_at,
          }
        : null,
      history: history.map((h) => ({
        version: h.version,
        accuracy: h.accuracy,
        f1: h.f1,
        training_samples: h.training_samples,
        trained_at: h.trained_at,
      })),
      recent_predictions: predictions,
      source: "supabase",
    });
  } catch (error) {
    console.error("ML API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch ML data" },
      { status: 500 }
    );
  }
}
