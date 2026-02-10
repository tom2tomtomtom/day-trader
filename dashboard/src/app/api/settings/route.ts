import { NextResponse } from "next/server";

export async function GET() {
  // Report which services are configured via env vars
  const services = [
    {
      id: "supabase",
      label: "Supabase (Database)",
      configured: Boolean(
        process.env.NEXT_PUBLIC_SUPABASE_URL &&
          process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
      ),
    },
    {
      id: "anthropic",
      label: "Anthropic (AI Analysis)",
      configured: Boolean(process.env.ANTHROPIC_API_KEY),
    },
    {
      id: "finnhub",
      label: "Finnhub (Market Data)",
      configured: Boolean(process.env.FINNHUB_API_KEY),
    },
    {
      id: "perplexity",
      label: "Perplexity (Research)",
      configured: Boolean(process.env.PERPLEXITY_API_KEY),
    },
  ];

  return NextResponse.json({
    services,
    environment: process.env.RAILWAY_ENVIRONMENT_NAME || "local",
    project: process.env.RAILWAY_PROJECT_NAME || "dev",
  });
}
