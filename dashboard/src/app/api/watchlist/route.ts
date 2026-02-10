import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const { data, error } = await supabase
        .from("watchlist")
        .select("*")
        .eq("active", true);

      if (!error) {
        return NextResponse.json({
          timestamp: new Date().toISOString(),
          scanned: (data || []).length,
          watchlist: (data || []).map((w) => w.symbol),
          source: "supabase",
        });
      }
    }

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      scanned: 0,
      watchlist: [],
      source: "none",
    });
  } catch {
    return NextResponse.json({
      timestamp: new Date().toISOString(),
      scanned: 0,
      watchlist: [],
    });
  }
}
