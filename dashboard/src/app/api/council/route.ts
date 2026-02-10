import { NextResponse } from "next/server";

export async function GET() {
  // Council data will be populated when intelligence briefings run
  return NextResponse.json({
    timestamp: new Date().toISOString(),
    opportunities: [],
    message: "Phantom Council data will appear after the first intelligence briefing runs.",
  });
}
