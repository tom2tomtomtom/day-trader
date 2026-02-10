import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "..");

export async function GET() {
  try {
    const intelPath = path.join(DATA_DIR, "intelligence_report.json");
    const data = await fs.readFile(intelPath, "utf-8");
    const intel = JSON.parse(data);
    return NextResponse.json(intel);
  } catch (error) {
    console.error("Intel API error:", error);
    return NextResponse.json(
      { error: "Intelligence report not available. Run: python3 -m core.orchestrator --intel" },
      { status: 500 }
    );
  }
}
