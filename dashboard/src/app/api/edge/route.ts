import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "..");

export async function GET() {
  try {
    const edgePath = path.join(DATA_DIR, "edge_signals.json");
    const data = await fs.readFile(edgePath, "utf-8");
    const edge = JSON.parse(data);
    return NextResponse.json(edge);
  } catch (error) {
    console.error("Edge API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch edge signals. Run edge_scanner.py first." },
      { status: 500 }
    );
  }
}
