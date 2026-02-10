"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Check,
  X,
  Loader2,
  Server,
  Database,
  Brain,
  BarChart3,
  Zap,
  Clock,
  Cpu,
  ToggleLeft,
  TrendingUp,
  Bitcoin,
  Info,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Circle,
} from "lucide-react";

// ── Types ────────────────────────────────────────────────────────────

interface Service {
  id: string;
  label: string;
  configured: boolean;
}

interface FeatureFlag {
  id: string;
  label: string;
  description: string;
  enabled: boolean;
}

interface TradingUniverse {
  stocks: string[];
  crypto: string[];
}

interface SystemInfo {
  environment: string;
  project: string;
  nodeVersion: string;
  uptime: number;
  platform: string;
  memoryUsageMB: number;
}

interface SettingsData {
  services: Service[];
  featureFlags: FeatureFlag[];
  tradingUniverse: TradingUniverse;
  systemInfo: SystemInfo;
}

interface TestResult {
  ok: boolean;
  message: string;
  testedAt?: string;
}

// ── Helpers ──────────────────────────────────────────────────────────

const serviceIcons: Record<string, React.ReactNode> = {
  supabase: <Database className="w-5 h-5" />,
  anthropic: <Brain className="w-5 h-5" />,
  finnhub: <BarChart3 className="w-5 h-5" />,
  perplexity: <Server className="w-5 h-5" />,
};

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const parts: string[] = [];
  if (d > 0) parts.push(`${d}d`);
  if (h > 0) parts.push(`${h}h`);
  parts.push(`${m}m`);
  return parts.join(" ");
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString();
}

// ── Component ────────────────────────────────────────────────────────

export default function SettingsPage() {
  const [data, setData] = useState<SettingsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [testingService, setTestingService] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, TestResult>>(
    {}
  );

  const fetchSettings = useCallback(async () => {
    try {
      const res = await fetch("/api/settings");
      if (res.ok) {
        setData(await res.json());
      }
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  const testConnection = async (serviceId: string) => {
    setTestingService(serviceId);
    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "test", service: serviceId }),
      });
      const result: TestResult = await res.json();
      setTestResults((prev) => ({ ...prev, [serviceId]: result }));
    } catch {
      setTestResults((prev) => ({
        ...prev,
        [serviceId]: { ok: false, message: "Request failed" },
      }));
    } finally {
      setTestingService(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 gap-3 text-zinc-400">
        <Loader2 className="w-5 h-5 animate-spin" />
        Loading system status...
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center py-20 gap-3 text-red-400">
        <XCircle className="w-5 h-5" />
        Failed to load settings
      </div>
    );
  }

  return (
    <div className="max-w-3xl space-y-8 pb-12">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Settings</h1>
          <p className="text-zinc-400 mt-1">
            System configuration and service status
          </p>
        </div>
        <button
          onClick={() => {
            setLoading(true);
            fetchSettings();
          }}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* ── 1. Service Connections ───────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <Zap className="w-5 h-5 text-emerald-400" />
          <h2 className="text-lg font-semibold">Service Connections</h2>
        </div>
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 divide-y divide-zinc-800">
          {data.services.map((svc) => {
            const result = testResults[svc.id];
            const isTesting = testingService === svc.id;

            return (
              <div key={svc.id} className="p-5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-zinc-400">
                      {serviceIcons[svc.id] || (
                        <Server className="w-5 h-5" />
                      )}
                    </span>
                    <span className="font-medium">{svc.label}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    {svc.configured ? (
                      <span className="flex items-center gap-1.5 text-emerald-400 text-sm">
                        <Check className="w-4 h-4" />
                        Connected
                      </span>
                    ) : (
                      <span className="flex items-center gap-1.5 text-red-400 text-sm">
                        <X className="w-4 h-4" />
                        Not configured
                      </span>
                    )}
                    <button
                      disabled={!svc.configured || isTesting}
                      onClick={() => testConnection(svc.id)}
                      className="flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed bg-zinc-800 hover:bg-zinc-700 text-zinc-300"
                    >
                      {isTesting ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <RefreshCw className="w-3 h-3" />
                      )}
                      Test
                    </button>
                  </div>
                </div>

                {/* Test result */}
                {result && (
                  <div
                    className={`mt-3 flex items-start gap-2 text-sm rounded-lg px-3 py-2 ${
                      result.ok
                        ? "bg-emerald-950/40 text-emerald-300 border border-emerald-900/50"
                        : "bg-red-950/40 text-red-300 border border-red-900/50"
                    }`}
                  >
                    {result.ok ? (
                      <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />
                    ) : (
                      <XCircle className="w-4 h-4 mt-0.5 shrink-0" />
                    )}
                    <div>
                      <span>{result.message}</span>
                      {result.testedAt && (
                        <span className="ml-2 text-xs opacity-60">
                          at {formatTime(result.testedAt)}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
        <p className="mt-2 text-xs text-zinc-500">
          API keys are managed via Railway environment variables. The worker
          service reads them automatically.
        </p>
      </section>

      {/* ── 2. Feature Flags ────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <ToggleLeft className="w-5 h-5 text-amber-400" />
          <h2 className="text-lg font-semibold">Feature Flags</h2>
        </div>
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 divide-y divide-zinc-800">
          {data.featureFlags.map((flag) => (
            <div key={flag.id} className="flex items-center justify-between p-5">
              <div className="flex items-start gap-3">
                <Circle
                  className={`w-3 h-3 mt-1.5 shrink-0 ${
                    flag.enabled
                      ? "fill-emerald-400 text-emerald-400"
                      : "fill-zinc-600 text-zinc-600"
                  }`}
                />
                <div>
                  <div className="font-medium">{flag.label}</div>
                  <div className="text-sm text-zinc-400">
                    {flag.description}
                  </div>
                </div>
              </div>
              <span
                className={`text-xs font-mono px-2 py-1 rounded ${
                  flag.enabled
                    ? "bg-emerald-950/60 text-emerald-400 border border-emerald-900/50"
                    : "bg-zinc-800 text-zinc-500"
                }`}
              >
                {flag.id}
              </span>
            </div>
          ))}
        </div>
        <div className="mt-2 flex items-start gap-1.5 text-xs text-zinc-500">
          <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          <span>
            Feature flags are configured via Railway environment variables.
            Set to &quot;true&quot; or &quot;1&quot; to enable.
          </span>
        </div>
      </section>

      {/* ── 3. Trading Universe ─────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-5 h-5 text-blue-400" />
          <h2 className="text-lg font-semibold">Trading Universe</h2>
        </div>
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5 space-y-5">
          {/* Stocks */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <BarChart3 className="w-4 h-4 text-zinc-400" />
              <span className="text-sm font-medium text-zinc-300">
                Stocks
              </span>
              <span className="text-xs text-zinc-500">
                ({data.tradingUniverse.stocks.length})
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {data.tradingUniverse.stocks.map((sym) => (
                <span
                  key={sym}
                  className="px-2.5 py-1 rounded-md bg-zinc-800 text-zinc-200 text-sm font-mono border border-zinc-700/50 hover:border-zinc-600 transition-colors"
                >
                  {sym}
                </span>
              ))}
            </div>
          </div>

          <div className="border-t border-zinc-800" />

          {/* Crypto */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Bitcoin className="w-4 h-4 text-amber-400" />
              <span className="text-sm font-medium text-zinc-300">
                Crypto
              </span>
              <span className="text-xs text-zinc-500">
                ({data.tradingUniverse.crypto.length})
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {data.tradingUniverse.crypto.map((sym) => (
                <span
                  key={sym}
                  className="px-2.5 py-1 rounded-md bg-amber-950/30 text-amber-200 text-sm font-mono border border-amber-900/30 hover:border-amber-700/50 transition-colors"
                >
                  {sym}
                </span>
              ))}
            </div>
          </div>
        </div>
        <p className="mt-2 text-xs text-zinc-500">
          Sourced from the watchlist table in Supabase. Falls back to defaults
          if unavailable.
        </p>
      </section>

      {/* ── 4. System Info ──────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="w-5 h-5 text-purple-400" />
          <h2 className="text-lg font-semibold">System Info</h2>
        </div>
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 divide-y divide-zinc-800">
          <InfoRow
            icon={<Server className="w-4 h-4" />}
            label="Railway Project"
            value={data.systemInfo.project}
          />
          <InfoRow
            icon={<Zap className="w-4 h-4" />}
            label="Environment"
            value={data.systemInfo.environment}
          />
          <InfoRow
            icon={<Cpu className="w-4 h-4" />}
            label="Node.js Version"
            value={data.systemInfo.nodeVersion}
          />
          <InfoRow
            icon={<Clock className="w-4 h-4" />}
            label="Process Uptime"
            value={formatUptime(data.systemInfo.uptime)}
          />
          <InfoRow
            icon={<Server className="w-4 h-4" />}
            label="Platform"
            value={data.systemInfo.platform}
          />
          <InfoRow
            icon={<BarChart3 className="w-4 h-4" />}
            label="Memory Usage"
            value={`${data.systemInfo.memoryUsageMB} MB`}
          />
        </div>
      </section>
    </div>
  );
}

// ── Sub-component ────────────────────────────────────────────────────

function InfoRow({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between px-5 py-3.5">
      <div className="flex items-center gap-3 text-zinc-400">
        {icon}
        <span className="text-sm">{label}</span>
      </div>
      <span className="font-mono text-sm text-zinc-200">{value}</span>
    </div>
  );
}
