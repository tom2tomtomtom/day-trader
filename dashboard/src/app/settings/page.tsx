"use client";

import { useEffect, useState } from "react";
import { Check, X, Loader2, Server, Database, Brain, BarChart3 } from "lucide-react";

interface Service {
  id: string;
  label: string;
  configured: boolean;
}

interface SystemStatus {
  services: Service[];
  environment: string;
  project: string;
}

const serviceIcons: Record<string, React.ReactNode> = {
  supabase: <Database className="w-5 h-5" />,
  anthropic: <Brain className="w-5 h-5" />,
  finnhub: <BarChart3 className="w-5 h-5" />,
  perplexity: <Server className="w-5 h-5" />,
};

export default function SettingsPage() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/settings")
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatus(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-3 text-zinc-400">
        <Loader2 className="w-5 h-5 animate-spin" />
        Loading system status...
      </div>
    );
  }

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">System Status</h1>
        <p className="text-zinc-400 mt-1">
          API keys and services configured via environment variables
        </p>
      </div>

      {status && (
        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
          <div className="flex items-center gap-3 text-sm text-zinc-400 mb-4">
            <span className="px-2 py-1 bg-zinc-800 rounded font-mono">
              {status.project}
            </span>
            <span className="px-2 py-1 bg-zinc-800 rounded font-mono">
              {status.environment}
            </span>
          </div>
        </div>
      )}

      <div className="bg-zinc-900 rounded-xl border border-zinc-800 divide-y divide-zinc-800">
        {status?.services.map((svc) => (
          <div key={svc.id} className="flex items-center justify-between p-5">
            <div className="flex items-center gap-3">
              <span className="text-zinc-400">
                {serviceIcons[svc.id] || <Server className="w-5 h-5" />}
              </span>
              <span className="font-medium">{svc.label}</span>
            </div>
            {svc.configured ? (
              <span className="flex items-center gap-2 text-emerald-400 text-sm">
                <Check className="w-4 h-4" />
                Connected
              </span>
            ) : (
              <span className="flex items-center gap-2 text-red-400 text-sm">
                <X className="w-4 h-4" />
                Not configured
              </span>
            )}
          </div>
        ))}
      </div>

      <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-5 text-sm text-zinc-500">
        API keys are managed via Railway environment variables.
        The worker service reads them automatically — no manual entry needed.
      </div>
    </div>
  );
}
