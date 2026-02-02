import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "..");

export async function GET() {
  try {
    // Read both regime files
    const [regimeState, currentState] = await Promise.all([
      fs.readFile(path.join(DATA_DIR, "regime_state.json"), "utf-8").catch(() => "{}"),
      fs.readFile(path.join(DATA_DIR, "current_state.json"), "utf-8").catch(() => "{}"),
    ]);

    const regime = JSON.parse(regimeState);
    const current = JSON.parse(currentState);

    return NextResponse.json({
      ...regime,
      current_state: current,
    });
  } catch (error) {
    console.error("Regime API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch regime" },
      { status: 500 }
    );
  }
}
