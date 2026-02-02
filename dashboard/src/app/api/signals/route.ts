import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "..");

export async function GET() {
  try {
    const signalsPath = path.join(DATA_DIR, "combined_signals.json");
    const data = await fs.readFile(signalsPath, "utf-8");
    const signals = JSON.parse(data);
    return NextResponse.json(signals);
  } catch (error) {
    console.error("Signals API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch signals" },
      { status: 500 }
    );
  }
}
