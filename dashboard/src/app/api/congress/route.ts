import { NextResponse } from "next/server";

export async function GET() {
  // Congressional data populated during intelligence briefings
  return NextResponse.json({
    timestamp: new Date().toISOString(),
    trades: [],
    signals: null,
    message: "Congressional trading data will appear after the first intelligence briefing.",
  });
}
