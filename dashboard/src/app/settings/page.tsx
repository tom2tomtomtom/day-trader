"use client";

import { useEffect, useState } from "react";
import { Save, Eye, EyeOff, Check, AlertCircle } from "lucide-react";

interface ApiKeys {
  finnhub?: string;
  polygon?: string;
  alpha_vantage?: string;
  openai?: string;
  news_api?: string;
}

interface SaveStatus {
  success: boolean;
  message: string;
}

export default function SettingsPage() {
  const [keys, setKeys] = useState<ApiKeys>({});
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<SaveStatus | null>(null);

  const apiKeyFields = [
    {
      id: "finnhub",
      label: "Finnhub API Key",
      description: "Real-time market data, news, and fundamentals",
      link: "https://finnhub.io/",
    },
    {
      id: "polygon",
      label: "Polygon.io API Key",
      description: "Stock, options, and crypto data",
      link: "https://polygon.io/",
    },
    {
      id: "alpha_vantage",
      label: "Alpha Vantage API Key",
      description: "Stock data, forex, and crypto",
      link: "https://www.alphavantage.co/",
    },
    {
      id: "openai",
      label: "OpenAI API Key",
      description: "For AI-powered analysis and sentiment",
      link: "https://platform.openai.com/",
    },
    {
      id: "news_api",
      label: "News API Key",
      description: "News headlines for sentiment analysis",
      link: "https://newsapi.org/",
    },
  ];

  useEffect(() => {
    fetchKeys();
  }, []);

  const fetchKeys = async () => {
    try {
      const res = await fetch("/api/settings");
      if (res.ok) {
        const data = await res.json();
        setKeys(data.keys || {});
      }
    } catch (error) {
      console.error("Failed to fetch settings:", error);
    }
  };

  const saveKeys = async () => {
    setSaving(true);
    setStatus(null);
    
    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keys }),
      });
      
      if (res.ok) {
        setStatus({ success: true, message: "API keys saved successfully!" });
      } else {
        throw new Error("Failed to save");
      }
    } catch (error) {
      setStatus({ success: false, message: "Failed to save API keys" });
    } finally {
      setSaving(false);
    }
  };

  const updateKey = (keyId: string, value: string) => {
    setKeys((prev) => ({ ...prev, [keyId]: value }));
  };

  const toggleShowKey = (keyId: string) => {
    setShowKeys((prev) => ({ ...prev, [keyId]: !prev[keyId] }));
  };

  const maskKey = (key: string | undefined) => {
    if (!key) return "";
    if (key.length <= 8) return "••••••••";
    return key.slice(0, 4) + "••••••••" + key.slice(-4);
  };

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-zinc-400 mt-1">
          Configure API keys for market data providers
        </p>
      </div>

      {status && (
        <div
          className={`flex items-center gap-2 p-4 rounded-lg ${
            status.success
              ? "bg-emerald-600/20 text-emerald-400"
              : "bg-red-600/20 text-red-400"
          }`}
        >
          {status.success ? (
            <Check className="w-5 h-5" />
          ) : (
            <AlertCircle className="w-5 h-5" />
          )}
          {status.message}
        </div>
      )}

      <div className="bg-zinc-900 rounded-xl border border-zinc-800 divide-y divide-zinc-800">
        {apiKeyFields.map((field) => (
          <div key={field.id} className="p-6">
            <div className="flex items-center justify-between mb-2">
              <label htmlFor={field.id} className="font-medium">
                {field.label}
              </label>
              <a
                href={field.link}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-emerald-500 hover:text-emerald-400"
              >
                Get API Key →
              </a>
            </div>
            <p className="text-sm text-zinc-400 mb-3">{field.description}</p>
            <div className="relative">
              <input
                id={field.id}
                type={showKeys[field.id] ? "text" : "password"}
                value={
                  showKeys[field.id]
                    ? keys[field.id as keyof ApiKeys] || ""
                    : keys[field.id as keyof ApiKeys]
                    ? maskKey(keys[field.id as keyof ApiKeys])
                    : ""
                }
                onChange={(e) => updateKey(field.id, e.target.value)}
                placeholder="Enter API key..."
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 pr-12 focus:outline-none focus:border-emerald-500"
              />
              <button
                type="button"
                onClick={() => toggleShowKey(field.id)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-white"
              >
                {showKeys[field.id] ? (
                  <EyeOff className="w-5 h-5" />
                ) : (
                  <Eye className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="flex justify-end">
        <button
          onClick={saveKeys}
          disabled={saving}
          className="flex items-center gap-2 px-6 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-zinc-700 rounded-lg transition-colors"
        >
          <Save className="w-4 h-4" />
          {saving ? "Saving..." : "Save Settings"}
        </button>
      </div>

      <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
        <h2 className="font-semibold mb-4">Trading Parameters</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-zinc-400 mb-1">
              Starting Capital
            </label>
            <input
              type="number"
              defaultValue={100000}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1">
              Max Positions
            </label>
            <input
              type="number"
              defaultValue={5}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1">
              Stop Loss %
            </label>
            <input
              type="number"
              defaultValue={2}
              step={0.5}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1">
              Profit Target %
            </label>
            <input
              type="number"
              defaultValue={3}
              step={0.5}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 focus:outline-none focus:border-emerald-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
