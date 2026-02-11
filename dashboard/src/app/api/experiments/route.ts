import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    if (!isSupabaseConfigured()) {
      return NextResponse.json({
        error: "Supabase not configured",
        experiments: [],
        hypotheses: [],
        drift: [],
        cycles: [],
      });
    }

    const [experiments, hypotheses, drift, cycles] = await Promise.all([
      supabase
        .from("experiments")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(20),
      supabase
        .from("hypotheses")
        .select("*")
        .order("priority_score", { ascending: false })
        .limit(20),
      supabase
        .from("feature_drift_log")
        .select("*")
        .eq("is_significant", true)
        .order("created_at", { ascending: false })
        .limit(20),
      supabase
        .from("learning_actions")
        .select("*")
        .eq("action_type", "cycle_complete")
        .order("created_at", { ascending: false })
        .limit(10),
    ]);

    return NextResponse.json({
      experiments: (experiments.data || []).map((e) => ({
        experiment_id: e.experiment_id,
        name: e.name,
        experiment_type: e.experiment_type,
        status: e.status,
        effect_size: e.effect_size,
        p_value: e.p_value,
        is_significant: e.is_significant,
        best_model_type: e.best_model_type,
        narrative: e.narrative,
        runtime_seconds: e.runtime_seconds,
        created_at: e.created_at,
      })),
      hypotheses: (hypotheses.data || []).map((h) => ({
        hypothesis_id: h.hypothesis_id,
        category: h.category,
        statement: h.statement,
        priority_score: h.priority_score,
        status: h.status,
        effect_size: h.effect_size,
        sample_size: h.sample_size,
        confidence_level: h.confidence_level,
        created_at: h.created_at,
      })),
      drift: (drift.data || []).map((d) => ({
        feature_name: d.feature_name,
        drift_magnitude: d.drift_magnitude,
        current_mean: d.current_mean,
        training_mean: d.training_mean,
        created_at: d.created_at,
      })),
      cycles: (cycles.data || []).map((c) => ({
        description: c.description,
        after_state: c.after_state,
        created_at: c.created_at,
      })),
      source: "supabase",
    });
  } catch (error) {
    console.error("Experiments API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch experiment data" },
      { status: 500 }
    );
  }
}
