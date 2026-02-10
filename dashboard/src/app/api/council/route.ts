import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "..");

export async function GET() {
  try {
    const intelPath = path.join(DATA_DIR, "intelligence_report.json");
    const data = await fs.readFile(intelPath, "utf-8");
    const intel = JSON.parse(data);

    // Extract council data from opportunities
    const councilData = {
      timestamp: intel.timestamp,
      opportunities: (intel.opportunities || []).map((opp: Record<string, unknown>) => ({
        symbol: opp.symbol,
        action: opp.action,
        council_action: opp.council_action,
        council_score: opp.council_score,
        council_conviction: opp.council_conviction,
        council_bulls: opp.council_bulls,
        council_bears: opp.council_bears,
        persona_verdicts: opp.persona_verdicts,
        opportunity_score: opp.opportunity_score,
        headline: opp.headline,
        thesis: opp.thesis,
        key_drivers: opp.key_drivers,
        key_risks: opp.key_risks,
      })),
    };

    return NextResponse.json(councilData);
  } catch (error) {
    console.error("Council API error:", error);
    return NextResponse.json(
      { error: "Council data not available. Run: python3 -m core.orchestrator --intel" },
      { status: 500 }
    );
  }
}
