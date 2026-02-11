"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, IChartApi, CandlestickSeries } from "lightweight-charts";

interface ChartData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export function PriceChart({ symbol }: { symbol: string }) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "rgba(255, 255, 255, 0.6)",
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.02)" },
        horzLines: { color: "rgba(255, 255, 255, 0.02)" },
      },
      width: chartContainerRef.current.clientWidth,
      height: 300,
      timeScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
      },
      crosshair: {
        vertLine: { color: "rgba(255, 255, 255, 0.4)", labelBackgroundColor: "#0a0a0a" },
        horzLine: { color: "rgba(255, 255, 255, 0.4)", labelBackgroundColor: "#0a0a0a" },
      },
    });

    chartRef.current = chart;

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#ff6b00",
      downColor: "#ff2e2e",
      borderDownColor: "#ff2e2e",
      borderUpColor: "#ff6b00",
      wickDownColor: "#ff2e2e",
      wickUpColor: "#ff6b00",
    });

    // Fetch data
    const fetchData = async () => {
      try {
        const res = await fetch(`/api/chart?symbol=${symbol}`);
        if (!res.ok) throw new Error("Failed to fetch chart data");

        const data: ChartData[] = await res.json();

        if (data && data.length > 0) {
          candlestickSeries.setData(
            data.map((d) => ({
              time: d.time,
              open: d.open,
              high: d.high,
              low: d.low,
              close: d.close,
            }))
          );
          chart.timeScale().fitContent();
        }
        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load chart");
        setLoading(false);
      }
    };

    fetchData();

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [symbol]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-red-hot">
        {error}
      </div>
    );
  }

  return (
    <div className="relative h-full">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black-card/50">
          <div className="animate-spin h-8 w-8 border-b-2 border-red-hot"></div>
        </div>
      )}
      <div ref={chartContainerRef} className="h-full" />
    </div>
  );
}
