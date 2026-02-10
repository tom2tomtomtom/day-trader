import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { exec } from "child_process";
import { promisify } from "util";
import { DATA_DIR } from "@/lib/data-dir";

const execAsync = promisify(exec);
const PYTHON = path.join(DATA_DIR, "..", "day-trader", "venv", "bin", "python");

interface AutomationConfig {
  mode: "FULL_AUTO" | "APPROVAL" | "PAUSED" | "STOPPED";
  scan_interval_minutes: number;
  trade_interval_minutes: number;
  auto_close_eod: boolean;
  max_daily_trades: number;
  max_daily_loss_pct: number;
  last_scan: string | null;
  last_trade_cycle: string | null;
  trades_today: number;
}

const DEFAULT_CONFIG: AutomationConfig = {
  mode: "PAUSED",
  scan_interval_minutes: 30,
  trade_interval_minutes: 5,
  auto_close_eod: true,
  max_daily_trades: 20,
  max_daily_loss_pct: 5.0,
  last_scan: null,
  last_trade_cycle: null,
  trades_today: 0,
};

async function loadConfig(): Promise<AutomationConfig> {
  const configPath = path.join(DATA_DIR, "automation_config.json");
  try {
    const data = await fs.readFile(configPath, "utf-8");
    return { ...DEFAULT_CONFIG, ...JSON.parse(data) };
  } catch {
    return DEFAULT_CONFIG;
  }
}

async function saveConfig(config: AutomationConfig): Promise<void> {
  const configPath = path.join(DATA_DIR, "automation_config.json");
  await fs.writeFile(configPath, JSON.stringify(config, null, 2));
}

interface PendingTrade {
  id: string;
  symbol: string;
  action: string;
  price: number;
  reason?: string;
  queued_at: string;
}

async function loadPending(): Promise<{ trades: PendingTrade[] }> {
  const pendingPath = path.join(DATA_DIR, "pending_trades.json");
  try {
    const data = await fs.readFile(pendingPath, "utf-8");
    return JSON.parse(data);
  } catch {
    return { trades: [] };
  }
}

async function loadPositions(): Promise<{ positions: Record<string, unknown>; total_trades: number }> {
  const posPath = path.join(DATA_DIR, "day_positions.json");
  try {
    const data = await fs.readFile(posPath, "utf-8");
    return JSON.parse(data);
  } catch {
    return { positions: {}, total_trades: 0 };
  }
}

// GET - Get automation status
export async function GET() {
  try {
    const config = await loadConfig();
    const pending = await loadPending();
    const positions = await loadPositions();

    // Check market status
    const now = new Date();
    const hour = now.getUTCHours();
    const weekday = now.getUTCDay();
    
    const markets = {
      US: { open: weekday > 0 && weekday < 6 && hour >= 14 && hour < 21 },
      Europe: { open: weekday > 0 && weekday < 6 && hour >= 8 && hour < 16 },
      Japan: { open: weekday > 0 && weekday < 6 && (hour < 6 || hour >= 0) },
      HongKong: { open: weekday > 0 && weekday < 6 && hour >= 1 && hour < 8 },
      Australia: { open: weekday > 0 && weekday < 6 && (hour >= 23 || hour < 5) },
    };

    const activeMarkets = Object.entries(markets)
      .filter(([, v]) => v.open)
      .map(([k]) => k);

    return NextResponse.json({
      mode: config.mode,
      config,
      active_markets: activeMarkets,
      pending_trades: pending.trades,
      pending_count: pending.trades.length,
      open_positions: Object.keys(positions.positions).length,
      trades_today: positions.total_trades,
    });
  } catch (error) {
    console.error("Automation GET error:", error);
    return NextResponse.json({ error: "Failed to get status" }, { status: 500 });
  }
}

// POST - Control automation
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { action, ...params } = body;

    const config = await loadConfig();

    switch (action) {
      case "set_mode": {
        const { mode } = params;
        if (!["FULL_AUTO", "APPROVAL", "PAUSED", "STOPPED"].includes(mode)) {
          return NextResponse.json({ error: "Invalid mode" }, { status: 400 });
        }
        config.mode = mode;
        await saveConfig(config);
        return NextResponse.json({ success: true, mode });
      }

      case "update_config": {
        const updates = params.config || {};
        const newConfig = { ...config, ...updates };
        await saveConfig(newConfig);
        return NextResponse.json({ success: true, config: newConfig });
      }

      case "run_scan": {
        try {
          const { stdout } = await execAsync(
            `cd ${DATA_DIR} && ${PYTHON} scanner.py scan --json`,
            { timeout: 120000 }
          );
          const result = JSON.parse(stdout);
          config.last_scan = new Date().toISOString();
          await saveConfig(config);
          return NextResponse.json({ success: true, result });
        } catch (err) {
          return NextResponse.json({ error: "Scan failed", details: String(err) }, { status: 500 });
        }
      }

      case "run_trade_cycle": {
        try {
          const { stdout } = await execAsync(
            `cd ${DATA_DIR} && ${PYTHON} day_trader.py run --json`,
            { timeout: 60000 }
          );
          const result = JSON.parse(stdout);
          config.last_trade_cycle = new Date().toISOString();
          await saveConfig(config);
          return NextResponse.json({ success: true, result });
        } catch (err) {
          return NextResponse.json({ error: "Trade cycle failed", details: String(err) }, { status: 500 });
        }
      }

      case "approve_trade": {
        const { tradeId } = params;
        try {
          const { stdout } = await execAsync(
            `cd ${DATA_DIR} && ${PYTHON} automation.py approve --id "${tradeId}" --json`,
            { timeout: 30000 }
          );
          return NextResponse.json(JSON.parse(stdout));
        } catch (err) {
          return NextResponse.json({ error: "Approve failed", details: String(err) }, { status: 500 });
        }
      }

      case "reject_trade": {
        const { tradeId } = params;
        const pending = await loadPending();
        pending.trades = pending.trades.filter((t) => t.id !== tradeId);
        const pendingPath = path.join(DATA_DIR, "pending_trades.json");
        await fs.writeFile(pendingPath, JSON.stringify(pending, null, 2));
        return NextResponse.json({ success: true });
      }

      case "manual_trade": {
        const { symbol, tradeAction, reason } = params;
        try {
          const { stdout } = await execAsync(
            `cd ${DATA_DIR} && ${PYTHON} automation.py manual --symbol "${symbol}" --action "${tradeAction}" --json`,
            { timeout: 30000 }
          );
          return NextResponse.json(JSON.parse(stdout));
        } catch (err) {
          return NextResponse.json({ error: "Manual trade failed", details: String(err) }, { status: 500 });
        }
      }

      case "close_all": {
        try {
          const { stdout } = await execAsync(
            `cd ${DATA_DIR} && ${PYTHON} day_trader.py close --json`,
            { timeout: 30000 }
          );
          return NextResponse.json({ success: true, result: stdout });
        } catch (err) {
          return NextResponse.json({ error: "Close all failed", details: String(err) }, { status: 500 });
        }
      }

      default:
        return NextResponse.json({ error: "Unknown action" }, { status: 400 });
    }
  } catch (error) {
    console.error("Automation POST error:", error);
    return NextResponse.json({ error: "Failed to process action" }, { status: 500 });
  }
}
