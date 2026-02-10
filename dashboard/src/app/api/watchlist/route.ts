import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";
import { promises as fs } from "fs";
import path from "path";
import { DATA_DIR } from "@/lib/data-dir";

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const { data, error } = await supabase
        .from("watchlist")
        .select("*")
        .eq("active", true);

      if (!error && data?.length) {
        return NextResponse.json({
          timestamp: new Date().toISOString(),
          scanned: data.length,
          watchlist: data.map((w) => w.symbol),
          source: "supabase",
        });
      }
    }

    // Fallback
    const watchlistPath = path.join(DATA_DIR, "watchlist.json");
    const data = await fs.readFile(watchlistPath, "utf-8");
    return NextResponse.json(JSON.parse(data));
  } catch {
    return NextResponse.json({
      timestamp: new Date().toISOString(),
      scanned: 0,
      watchlist: [],
    });
  }
}
