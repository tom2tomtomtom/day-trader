import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "..");

export async function GET() {
  try {
    const watchlistPath = path.join(DATA_DIR, "watchlist.json");
    
    try {
      const data = await fs.readFile(watchlistPath, "utf-8");
      const watchlist = JSON.parse(data);
      return NextResponse.json(watchlist);
    } catch {
      return NextResponse.json({
        timestamp: new Date().toISOString(),
        scanned: 0,
        setups: {},
        watchlist: [],
      });
    }
  } catch (error) {
    console.error("Watchlist API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch watchlist" },
      { status: 500 }
    );
  }
}
