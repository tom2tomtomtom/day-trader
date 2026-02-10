import { NextResponse } from "next/server";

export async function GET() {
  // Intelligence reports will be populated when briefings run (9 AM + 4:30 PM ET)
  return NextResponse.json({
    timestamp: new Date().toISOString(),
    market: {
      regime: "UNKNOWN",
      regime_score: 50,
      fear_greed: 50,
      vix: 0,
      risk_score: 50,
      opportunity_score: 50,
    },
    triggers: [],
    critical_triggers: 0,
    digest: {
      headline: "Awaiting First Intelligence Briefing",
      mood: "The system is warming up. Intelligence briefings run at 9:00 AM and 4:30 PM ET.",
      regime_narrative: "No regime data available yet.",
      risk_warnings: [],
      smart_money: "No smart money signals yet.",
      closing_thought: "The worker is scanning markets. Data will populate as trades and signals accumulate.",
    },
    opportunities: [],
    congress: {
      total_trades: 0,
      signals: 0,
      cluster_buys: 0,
      hot_symbols: [],
      notable_activity: [],
    },
    stats: {
      symbols_analyzed: 0,
      actionable_signals: 0,
    },
  });
}
