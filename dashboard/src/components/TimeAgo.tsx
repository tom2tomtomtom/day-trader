"use client";

import { useEffect, useState } from "react";

interface TimeAgoProps {
  timestamp: string | null | undefined;
  staleAfterMs?: number; // default 600000 (10 minutes)
  className?: string;
  prefix?: string; // default "Updated"
}

function formatRelativeTime(timestamp: string): string {
  const now = Date.now();
  const then = new Date(timestamp).getTime();
  const diffMs = now - then;

  if (diffMs < 0 || isNaN(diffMs)) return "just now";

  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 30) return "just now";
  if (seconds < 60) return `${seconds}s ago`;

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  if (days === 1) return "1d ago";
  if (days < 7) return `${days}d ago`;

  return new Date(timestamp).toLocaleDateString();
}

function isStale(timestamp: string, staleAfterMs: number): boolean {
  const diffMs = Date.now() - new Date(timestamp).getTime();
  return diffMs > staleAfterMs;
}

export function TimeAgo({
  timestamp,
  staleAfterMs = 600000,
  className = "",
  prefix = "Updated",
}: TimeAgoProps) {
  const [, setTick] = useState(0);

  // Re-render every 30 seconds to keep relative time current
  useEffect(() => {
    const interval = setInterval(() => setTick((t) => t + 1), 30000);
    return () => clearInterval(interval);
  }, []);

  if (!timestamp) return null;

  const stale = isStale(timestamp, staleAfterMs);
  const relativeTime = formatRelativeTime(timestamp);

  return (
    <span
      className={`text-xs ${stale ? "text-amber-500" : "text-zinc-500"} ${className}`}
      title={new Date(timestamp).toLocaleString()}
    >
      {prefix} {relativeTime}
    </span>
  );
}
