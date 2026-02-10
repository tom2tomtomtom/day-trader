import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { DATA_DIR } from "@/lib/data-dir";

const SETTINGS_PATH = path.join(DATA_DIR, "settings.json");

interface Settings {
  keys: {
    finnhub?: string;
    polygon?: string;
    alpha_vantage?: string;
    openai?: string;
    news_api?: string;
  };
  trading: {
    starting_capital: number;
    max_positions: number;
    stop_loss_pct: number;
    profit_target_pct: number;
  };
}

const defaultSettings: Settings = {
  keys: {},
  trading: {
    starting_capital: 100000,
    max_positions: 5,
    stop_loss_pct: 2,
    profit_target_pct: 3,
  },
};

async function loadSettings(): Promise<Settings> {
  try {
    const data = await fs.readFile(SETTINGS_PATH, "utf-8");
    return { ...defaultSettings, ...JSON.parse(data) };
  } catch {
    return defaultSettings;
  }
}

async function saveSettings(settings: Settings): Promise<void> {
  await fs.writeFile(SETTINGS_PATH, JSON.stringify(settings, null, 2));
}

export async function GET() {
  try {
    const settings = await loadSettings();
    // Mask API keys for display
    const maskedKeys: Record<string, string> = {};
    for (const [key, value] of Object.entries(settings.keys)) {
      if (value) {
        maskedKeys[key] = value; // In production, you'd mask these
      }
    }
    
    return NextResponse.json({
      keys: maskedKeys,
      trading: settings.trading,
    });
  } catch (error) {
    console.error("Settings GET error:", error);
    return NextResponse.json(
      { error: "Failed to load settings" },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const currentSettings = await loadSettings();
    
    // Update keys
    if (body.keys) {
      for (const [key, value] of Object.entries(body.keys)) {
        if (value && typeof value === "string" && value.trim()) {
          currentSettings.keys[key as keyof typeof currentSettings.keys] = value.trim();
        }
      }
    }
    
    // Update trading params
    if (body.trading) {
      currentSettings.trading = { ...currentSettings.trading, ...body.trading };
    }
    
    await saveSettings(currentSettings);
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Settings POST error:", error);
    return NextResponse.json(
      { error: "Failed to save settings" },
      { status: 500 }
    );
  }
}
