import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const symbol = searchParams.get("symbol") || "SPY";
  const period = searchParams.get("period") || "1mo";
  const interval = searchParams.get("interval") || "1d";

  try {
    // Fetch from Yahoo Finance API (public endpoint)
    const endDate = Math.floor(Date.now() / 1000);
    const startDate = endDate - getPeriodSeconds(period);
    
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${startDate}&period2=${endDate}&interval=${interval}`;
    
    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0",
      },
    });

    if (!response.ok) {
      throw new Error(`Yahoo Finance returned ${response.status}`);
    }

    const data = await response.json();
    const result = data.chart?.result?.[0];
    
    if (!result) {
      return NextResponse.json({ error: "No data available" }, { status: 404 });
    }

    const timestamps = result.timestamp || [];
    const quotes = result.indicators?.quote?.[0] || {};
    
    const chartData = timestamps.map((timestamp: number, i: number) => ({
      time: new Date(timestamp * 1000).toISOString().split("T")[0],
      open: quotes.open?.[i] ?? 0,
      high: quotes.high?.[i] ?? 0,
      low: quotes.low?.[i] ?? 0,
      close: quotes.close?.[i] ?? 0,
      volume: quotes.volume?.[i] ?? 0,
    })).filter((d: { open: number }) => d.open > 0);

    return NextResponse.json(chartData);
  } catch (error) {
    console.error("Chart API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch chart data" },
      { status: 500 }
    );
  }
}

function getPeriodSeconds(period: string): number {
  const periods: Record<string, number> = {
    "1d": 86400,
    "5d": 5 * 86400,
    "1mo": 30 * 86400,
    "3mo": 90 * 86400,
    "6mo": 180 * 86400,
    "1y": 365 * 86400,
  };
  return periods[period] || periods["1mo"];
}
