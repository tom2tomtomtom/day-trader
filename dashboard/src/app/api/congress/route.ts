import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "..");

export async function GET() {
  try {
    // Try to load congressional trades data
    const congressPath = path.join(DATA_DIR, "congressional_trades.json");
    const congressData = JSON.parse(await fs.readFile(congressPath, "utf-8"));

    // Also try to load intelligence report for signal data
    let signals = null;
    try {
      const intelPath = path.join(DATA_DIR, "intelligence_report.json");
      const intelData = JSON.parse(await fs.readFile(intelPath, "utf-8"));
      signals = intelData.congress;
    } catch {
      // Intel report may not exist yet
    }

    return NextResponse.json({
      timestamp: congressData.timestamp,
      trades: congressData.trades || [],
      signals: signals,
    });
  } catch (error) {
    console.error("Congress API error:", error);
    return NextResponse.json(
      { error: "Congressional data not available. Run: python3 -m core.orchestrator --intel" },
      { status: 500 }
    );
  }
}
