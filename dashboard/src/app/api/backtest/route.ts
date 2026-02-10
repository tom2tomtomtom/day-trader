import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { DATA_DIR } from "@/lib/data-dir";

export async function GET() {
  try {
    const btPath = path.join(DATA_DIR, "backtest_results.json");
    const data = await fs.readFile(btPath, "utf-8");
    return NextResponse.json(JSON.parse(data));
  } catch (error) {
    console.error("Backtest API error:", error);
    return NextResponse.json(
      { error: "No backtest results. Run: python3 -m core.backtester --symbol SPY" },
      { status: 500 }
    );
  }
}
