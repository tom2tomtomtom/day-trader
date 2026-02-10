import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { spawn } from "child_process";
import { DATA_DIR } from "@/lib/data-dir";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { action, symbol, direction, shares, price } = body;

    if (action === "open") {
      // Execute a trade via Python script
      const result = await runPythonTrade(symbol, direction, shares, price);
      return NextResponse.json({ success: true, result });
    }

    if (action === "close") {
      // Close a position
      const result = await closePythonPosition(symbol);
      return NextResponse.json({ success: true, result });
    }

    return NextResponse.json(
      { error: "Invalid action" },
      { status: 400 }
    );
  } catch (error) {
    console.error("Trade API error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Trade failed" },
      { status: 500 }
    );
  }
}

async function runPythonTrade(
  symbol: string,
  direction: string,
  shares: number,
  price: number
): Promise<string> {
  return new Promise((resolve, reject) => {
    // For now, directly update the positions file
    // In production, this would call the Python trading script
    updatePositionsFile(symbol, direction, shares, price)
      .then(() => resolve(`Opened ${direction} position: ${shares} shares of ${symbol} @ $${price}`))
      .catch(reject);
  });
}

async function closePythonPosition(symbol: string): Promise<string> {
  return new Promise((resolve, reject) => {
    closePositionInFile(symbol)
      .then(() => resolve(`Closed position: ${symbol}`))
      .catch(reject);
  });
}

async function updatePositionsFile(
  symbol: string,
  direction: string,
  shares: number,
  price: number
) {
  const positionsPath = path.join(DATA_DIR, "day_positions.json");

  let data;
  try {
    const content = await fs.readFile(positionsPath, "utf-8");
    data = JSON.parse(content);
  } catch {
    data = {
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
  }

  const cost = shares * price;

  // Check if we have enough cash
  if (cost > data.cash) {
    throw new Error(`Insufficient cash. Need $${cost.toFixed(2)}, have $${data.cash.toFixed(2)}`);
  }

  // Calculate stop and target
  const stopPrice = direction === "LONG"
    ? parseFloat((price * 0.98).toFixed(2))
    : parseFloat((price * 1.02).toFixed(2));
  const targetPrice = direction === "LONG"
    ? parseFloat((price * 1.03).toFixed(2))
    : parseFloat((price * 0.97).toFixed(2));

  // Add position
  data.positions[symbol] = {
    symbol,
    direction,
    shares,
    entry_price: price,
    entry_time: new Date().toISOString(),
    entry_signal: {
      type: "MANUAL",
      direction,
      strength: 1.0,
      reason: "Manual trade from dashboard",
    },
    stop_price: stopPrice,
    target_price: targetPrice,
    highest_price: direction === "LONG" ? price : null,
    lowest_price: direction === "SHORT" ? price : null,
    trailing_stop: null,
    cost_basis: cost,
  };

  data.cash -= cost;
  data.total_trades += 1;

  await fs.writeFile(positionsPath, JSON.stringify(data, null, 2));
}

async function closePositionInFile(symbol: string) {
  const positionsPath = path.join(DATA_DIR, "day_positions.json");

  const content = await fs.readFile(positionsPath, "utf-8");
  const data = JSON.parse(content);

  if (!data.positions[symbol]) {
    throw new Error(`No position found for ${symbol}`);
  }

  const position = data.positions[symbol];

  // For paper trading, simulate exit at entry price (neutral)
  // In reality, this would fetch current price
  const exitPrice = position.entry_price;
  const pnl = position.direction === "LONG"
    ? (exitPrice - position.entry_price) * position.shares
    : (position.entry_price - exitPrice) * position.shares;

  // Move to closed trades
  data.closed_trades.push({
    ...position,
    exit_price: exitPrice,
    exit_time: new Date().toISOString(),
    exit_reason: "MANUAL_CLOSE",
    pnl,
    pnl_pct: (pnl / position.cost_basis) * 100,
  });

  // Return cash
  data.cash += position.cost_basis + pnl;

  // Update win/loss
  if (pnl >= 0) {
    data.winners += 1;
  } else {
    data.losers += 1;
  }
  data.gross_pnl += pnl;

  // Remove position
  delete data.positions[symbol];

  await fs.writeFile(positionsPath, JSON.stringify(data, null, 2));
}
