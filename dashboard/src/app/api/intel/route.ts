import { NextResponse } from "next/server";

export async function GET() {
  // Intelligence reports will be populated when briefings run (9 AM + 4:30 PM ET)
  return NextResponse.json({
    timestamp: new Date().toISOString(),
    opportunities: [],
    macro: null,
    congress: null,
    message: "Intelligence reports will appear after the first scheduled briefing.",
  });
}
