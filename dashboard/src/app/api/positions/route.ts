import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "..");

export async function GET() {
  try {
    const positionsPath = path.join(DATA_DIR, "day_positions.json");
    
    try {
      const data = await fs.readFile(positionsPath, "utf-8");
      const positions = JSON.parse(data);
      return NextResponse.json(positions);
    } catch {
      return NextResponse.json({
        positions: {},
        closed_trades: [],
        total_trades: 0,
        winners: 0,
        losers: 0,
        gross_pnl: 0,
      });
    }
  } catch (error) {
    console.error("Positions API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch positions" },
      { status: 500 }
    );
  }
}
