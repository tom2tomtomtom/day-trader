import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "..");

export async function GET() {
  try {
    // Read day positions
    const positionsPath = path.join(DATA_DIR, "day_positions.json");
    let positions = {
      date: new Date().toISOString().split("T")[0],
      capital: 100000,
      cash: 100000,
      positions: {},
      closed_trades: [],
      total_trades: 0,
      winners: 0,
      losers: 0,
      gross_pnl: 0,
    };

    try {
      const data = await fs.readFile(positionsPath, "utf-8");
      positions = JSON.parse(data);
    } catch {
      // File doesn't exist yet, use defaults
    }

    // Calculate current portfolio value
    let portfolioValue = positions.cash;
    const positionsWithPnl: Record<string, unknown> = {};

    for (const [symbol, pos] of Object.entries(positions.positions as Record<string, { shares: number; entry_price: number; direction: string }>)) {
      // In production, we'd fetch live prices here
      const currentPrice = pos.entry_price; // Placeholder
      const value = pos.shares * currentPrice;
      portfolioValue += value;

      const pnl = pos.direction === "LONG"
        ? (currentPrice - pos.entry_price) * pos.shares
        : (pos.entry_price - currentPrice) * pos.shares;
      
      positionsWithPnl[symbol] = {
        ...pos,
        current_price: currentPrice,
        pnl: pnl,
        pnl_pct: (pnl / (pos.entry_price * pos.shares)) * 100,
      };
    }

    const response = {
      portfolio_value: portfolioValue,
      cash: positions.cash,
      day_pnl: portfolioValue - 100000,
      day_pnl_pct: ((portfolioValue - 100000) / 100000) * 100,
      total_trades: positions.total_trades,
      winners: positions.winners,
      losers: positions.losers,
      open_positions: Object.keys(positions.positions).length,
      positions: positionsWithPnl,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("Status API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch status" },
      { status: 500 }
    );
  }
}
