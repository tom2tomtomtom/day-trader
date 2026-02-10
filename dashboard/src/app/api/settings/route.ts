import { NextRequest, NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

// ── Service definitions ──────────────────────────────────────────────
const SERVICE_DEFS = [
  {
    id: "supabase",
    label: "Supabase (Database)",
    envCheck: () =>
      Boolean(
        (process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL) &&
          (process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ||
            process.env.SUPABASE_ANON_KEY)
      ),
  },
  {
    id: "anthropic",
    label: "Anthropic (AI Analysis)",
    envCheck: () => Boolean(process.env.ANTHROPIC_API_KEY),
  },
  {
    id: "finnhub",
    label: "Finnhub (Market Data)",
    envCheck: () => Boolean(process.env.FINNHUB_API_KEY),
  },
  {
    id: "perplexity",
    label: "Perplexity (Research)",
    envCheck: () => Boolean(process.env.PERPLEXITY_API_KEY),
  },
] as const;

// ── Feature flag definitions ─────────────────────────────────────────
const FEATURE_FLAGS = [
  {
    id: "FEATURE_ML_GATE",
    label: "ML Trade Gate",
    description: "Block low-quality trades using ML predictions",
  },
  {
    id: "FEATURE_PRE_TRADE_RISK",
    label: "Pre-Trade Risk",
    description: "Enforce position limits and correlation checks",
  },
  {
    id: "FEATURE_RL_AGENT",
    label: "RL Agent",
    description: "Include reinforcement learning signals",
  },
  {
    id: "FEATURE_TELEGRAM",
    label: "Telegram Alerts",
    description: "Send trade alerts to Telegram",
  },
  {
    id: "FEATURE_ADVANCED_ORDERS",
    label: "Advanced Orders",
    description: "Enable limit and bracket orders",
  },
];

// ── Default watchlists ───────────────────────────────────────────────
const DEFAULT_STOCKS = [
  "SPY",
  "QQQ",
  "IWM",
  "AAPL",
  "MSFT",
  "GOOGL",
  "AMZN",
  "NVDA",
  "TSLA",
  "META",
  "AMD",
  "MSTR",
  "COIN",
];

const DEFAULT_CRYPTO = [
  "BTC-USD",
  "ETH-USD",
  "SOL-USD",
  "DOGE-USD",
  "SHIB-USD",
  "AVAX-USD",
];

// ── GET — return all settings data ───────────────────────────────────
export async function GET() {
  // Services
  const services = SERVICE_DEFS.map((s) => ({
    id: s.id,
    label: s.label,
    configured: s.envCheck(),
  }));

  // Feature flags
  const featureFlags = FEATURE_FLAGS.map((f) => ({
    ...f,
    enabled: process.env[f.id] === "true" || process.env[f.id] === "1",
  }));

  // Trading universe — try Supabase, fall back to defaults
  let stocks = DEFAULT_STOCKS;
  let crypto = DEFAULT_CRYPTO;

  if (isSupabaseConfigured()) {
    try {
      const { data } = await supabase
        .from("watchlist")
        .select("symbol, asset_type")
        .order("symbol");

      if (data && data.length > 0) {
        const dbStocks = data
          .filter(
            (r: { asset_type?: string }) =>
              !r.asset_type || r.asset_type === "stock"
          )
          .map((r: { symbol: string }) => r.symbol);
        const dbCrypto = data
          .filter(
            (r: { asset_type?: string }) => r.asset_type === "crypto"
          )
          .map((r: { symbol: string }) => r.symbol);

        if (dbStocks.length > 0) stocks = dbStocks;
        if (dbCrypto.length > 0) crypto = dbCrypto;
      }
    } catch {
      // Supabase query failed — use defaults
    }
  }

  // System info
  const systemInfo = {
    environment: process.env.RAILWAY_ENVIRONMENT_NAME || "local",
    project: process.env.RAILWAY_PROJECT_NAME || "dev",
    nodeVersion: process.version,
    uptime: Math.floor(process.uptime()),
    platform: process.platform,
    memoryUsageMB: Math.round(process.memoryUsage().rss / 1024 / 1024),
  };

  return NextResponse.json({
    services,
    featureFlags,
    tradingUniverse: { stocks, crypto },
    systemInfo,
  });
}

// ── POST — test individual service connections ───────────────────────
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { action, service } = body as {
      action: string;
      service: string;
    };

    if (action !== "test") {
      return NextResponse.json({ error: "Unknown action" }, { status: 400 });
    }

    switch (service) {
      case "supabase":
        return await testSupabase();
      case "anthropic":
        return testEnvVar("ANTHROPIC_API_KEY", "Anthropic");
      case "finnhub":
        return await testFinnhub();
      case "perplexity":
        return testEnvVar("PERPLEXITY_API_KEY", "Perplexity");
      default:
        return NextResponse.json(
          { error: `Unknown service: ${service}` },
          { status: 400 }
        );
    }
  } catch {
    return NextResponse.json(
      { ok: false, message: "Invalid request body" },
      { status: 400 }
    );
  }
}

// ── Test helpers ─────────────────────────────────────────────────────

async function testSupabase() {
  if (!isSupabaseConfigured()) {
    return NextResponse.json({
      ok: false,
      message: "Supabase not configured — missing env vars",
    });
  }
  try {
    const { count, error } = await supabase
      .from("trades")
      .select("*", { count: "exact", head: true });

    if (error) {
      return NextResponse.json({
        ok: false,
        message: `Query error: ${error.message}`,
      });
    }

    return NextResponse.json({
      ok: true,
      message: `Connected — ${count ?? 0} trades in database`,
      detail: { tradeCount: count ?? 0 },
      testedAt: new Date().toISOString(),
    });
  } catch (e) {
    return NextResponse.json({
      ok: false,
      message: `Connection failed: ${e instanceof Error ? e.message : "unknown"}`,
    });
  }
}

function testEnvVar(envKey: string, label: string) {
  const value = process.env[envKey];
  if (!value) {
    return NextResponse.json({
      ok: false,
      message: `${label} API key not set`,
    });
  }
  const masked = value.slice(0, 8) + "..." + value.slice(-4);
  return NextResponse.json({
    ok: true,
    message: `${label} key configured (${masked})`,
    testedAt: new Date().toISOString(),
  });
}

async function testFinnhub() {
  const key = process.env.FINNHUB_API_KEY;
  if (!key) {
    return NextResponse.json({
      ok: false,
      message: "Finnhub API key not set",
    });
  }
  try {
    const res = await fetch(
      `https://finnhub.io/api/v1/quote?symbol=SPY&token=${key}`,
      { signal: AbortSignal.timeout(8000) }
    );
    if (!res.ok) {
      return NextResponse.json({
        ok: false,
        message: `Finnhub returned HTTP ${res.status}`,
      });
    }
    const data = await res.json();
    if (data.c && data.c > 0) {
      return NextResponse.json({
        ok: true,
        message: `Connected — SPY @ $${data.c.toFixed(2)}`,
        detail: { spyPrice: data.c },
        testedAt: new Date().toISOString(),
      });
    }
    return NextResponse.json({
      ok: false,
      message: "Finnhub returned empty data — check API key",
    });
  } catch (e) {
    return NextResponse.json({
      ok: false,
      message: `Finnhub request failed: ${e instanceof Error ? e.message : "timeout"}`,
    });
  }
}
