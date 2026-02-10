"use client";

import { useEffect, useRef } from "react";
import { isSupabaseConfigured, supabase } from "@/lib/supabase";

type PostgresEvent = "INSERT" | "UPDATE" | "DELETE" | "*";

interface SubscriptionConfig {
  /** Supabase table name to listen on */
  table: string;
  /** Which events to listen for (default: all) */
  events?: PostgresEvent[];
  /** Callback when a change is detected — typically a refetch */
  onchange: () => void;
  /** Disable the subscription (e.g. while loading) */
  enabled?: boolean;
}

/**
 * Subscribe to Supabase Realtime changes on one or more tables.
 * When a change is detected, calls `onchange` (typically your refetch function).
 * Falls back to no-op when Supabase is not configured — pages keep polling.
 */
export function useRealtimeSubscription(configs: SubscriptionConfig[]) {
  const onchangeRefs = useRef<Map<string, () => void>>(new Map());

  // Keep callback refs current without re-subscribing
  for (const cfg of configs) {
    onchangeRefs.current.set(cfg.table, cfg.onchange);
  }

  useEffect(() => {
    if (!isSupabaseConfigured()) return;

    const activeConfigs = configs.filter((c) => c.enabled !== false);
    if (activeConfigs.length === 0) return;

    const channelName = `realtime:${activeConfigs.map((c) => c.table).join("+")}`;
    let channel = supabase.channel(channelName);

    for (const cfg of activeConfigs) {
      const events = cfg.events ?? ["*"];
      for (const event of events) {
        channel = channel.on(
          "postgres_changes" as "postgres_changes",
          { event, schema: "public", table: cfg.table },
          () => {
            const cb = onchangeRefs.current.get(cfg.table);
            if (cb) cb();
          }
        );
      }
    }

    channel.subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
    // Re-subscribe only when table names or enabled state changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    configs
      .map((c) => `${c.table}:${c.enabled !== false}`)
      .join(","),
  ]);
}

/**
 * Convenience: subscribe to a single table.
 */
export function useTableSubscription(
  table: string,
  onchange: () => void,
  events?: PostgresEvent[]
) {
  useRealtimeSubscription([{ table, events: events ?? ["*"], onchange }]);
}
