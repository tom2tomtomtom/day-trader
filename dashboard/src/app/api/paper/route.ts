import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "..");

export async function GET() {
  try {
    const paperPath = path.join(DATA_DIR, "paper_trades.json");
    const data = await fs.readFile(paperPath, "utf-8");
    return NextResponse.json(JSON.parse(data));
  } catch (error) {
    console.error("Paper trader API error:", error);
    return NextResponse.json(
      { error: "No paper trading data. Run: python3 -m core.paper_trader" },
      { status: 500 }
    );
  }
}
