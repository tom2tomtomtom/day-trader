import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

interface StrategyParameters {
  [key: string]: number;
}

interface StrategyConfig {
  name: string;
  enabled: boolean;
  parameters: StrategyParameters;
}

const DEFAULT_STRATEGIES: StrategyConfig[] = [
  {
    name: "momentum",
    enabled: true,
    parameters: {
      fast_ma: 10,
      slow_ma: 30,
      stop_loss_pct: 0.05,
      take_profit_pct: 0.12,
    },
  },
  {
    name: "mean_reversion",
    enabled: true,
    parameters: {
      bb_period: 20,
      bb_std: 2.0,
      stop_loss_pct: 0.04,
      take_profit_pct: 0.06,
    },
  },
  {
    name: "breakout",
    enabled: true,
    parameters: {
      lookback: 20,
      stop_loss_pct: 0.04,
      take_profit_pct: 0.1,
    },
  },
];

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const { data, error } = await supabase
        .from("strategy_configs")
        .select("*")
        .order("name", { ascending: true });

      if (!error && data && data.length > 0) {
        const strategies: StrategyConfig[] = data.map((row) => ({
          name: row.name,
          enabled: row.enabled ?? true,
          parameters:
            typeof row.parameters === "string"
              ? JSON.parse(row.parameters)
              : row.parameters || {},
        }));
        return NextResponse.json({ strategies, source: "supabase" });
      }
    }

    // Return defaults if Supabase is not configured or table doesn't exist
    return NextResponse.json({
      strategies: DEFAULT_STRATEGIES,
      source: "defaults",
    });
  } catch (error) {
    console.error("Strategies API GET error:", error);
    return NextResponse.json({
      strategies: DEFAULT_STRATEGIES,
      source: "defaults",
    });
  }
}

export async function POST(request: Request) {
  try {
    const body: StrategyConfig = await request.json();

    if (!body.name || !body.parameters) {
      return NextResponse.json(
        { error: "Missing required fields: name, parameters" },
        { status: 400 }
      );
    }

    if (!isSupabaseConfigured()) {
      return NextResponse.json(
        { error: "Supabase not configured" },
        { status: 503 }
      );
    }

    const { data, error } = await supabase
      .from("strategy_configs")
      .upsert(
        {
          name: body.name,
          enabled: body.enabled ?? true,
          parameters: body.parameters,
          updated_at: new Date().toISOString(),
        },
        { onConflict: "name" }
      )
      .select()
      .single();

    if (error) {
      console.error("Supabase upsert error:", error);
      return NextResponse.json(
        { error: `Failed to save strategy: ${error.message}` },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true, strategy: data });
  } catch (error) {
    console.error("Strategies API POST error:", error);
    return NextResponse.json(
      { error: "Failed to save strategy config" },
      { status: 500 }
    );
  }
}
