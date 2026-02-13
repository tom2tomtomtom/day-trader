import { NextResponse } from "next/server";
import { supabase, isSupabaseConfigured } from "@/lib/supabase";

function daysBetween(a: string, b: string): number {
  const ms = new Date(b).getTime() - new Date(a).getTime();
  return Math.max(0, Math.round(ms / 86400000));
}

export async function GET() {
  try {
    if (isSupabaseConfigured()) {
      const { data } = await supabase
        .from("trades")
        .select("*")
        .eq("is_backtest", true)
        .order("exit_date", { ascending: true })
        .limit(500);

      const trades = data || [];

      if (trades.length === 0) {
        return NextResponse.json(
          { error: "No backtest data yet. Run: python3 -m core.backtester --symbol SPY" },
          { status: 404 }
        );
      }

      const initialCapital = 100000;

      // Compute hold days for each trade
      const tradesWithHold = trades.map((t) => ({
        ...t,
        hold_days: daysBetween(t.entry_date, t.exit_date),
      }));

      // Total return from actual dollar P&L
      const totalPnlDollars = trades.reduce((s, t) => s + (t.pnl_dollars || 0), 0);
      const totalReturnPct = (totalPnlDollars / initialCapital) * 100;
      const finalValue = initialCapital + totalPnlDollars;

      // Win/loss stats
      const wins = trades.filter((t) => (t.pnl_dollars || 0) > 0);
      const losses = trades.filter((t) => (t.pnl_dollars || 0) <= 0);
      const winRate = trades.length > 0 ? wins.length / trades.length : 0;

      const avgWinPct = wins.length > 0
        ? wins.reduce((s, t) => s + (t.pnl_pct || 0), 0) / wins.length : 0;
      const avgLossPct = losses.length > 0
        ? losses.reduce((s, t) => s + (t.pnl_pct || 0), 0) / losses.length : 0;
      const largestWinPct = wins.length > 0
        ? Math.max(...wins.map((t) => t.pnl_pct || 0)) : 0;
      const largestLossPct = losses.length > 0
        ? Math.min(...losses.map((t) => t.pnl_pct || 0)) : 0;

      // Profit factor
      const grossProfits = wins.reduce((s, t) => s + (t.pnl_dollars || 0), 0);
      const grossLosses = Math.abs(losses.reduce((s, t) => s + (t.pnl_dollars || 0), 0));
      const profitFactor = grossLosses > 0 ? grossProfits / grossLosses : 0;

      // Expectancy
      const expectancy = trades.length > 0 ? totalPnlDollars / trades.length : 0;

      // Hold days
      const avgHoldDays = tradesWithHold.length > 0
        ? tradesWithHold.reduce((s, t) => s + t.hold_days, 0) / tradesWithHold.length : 0;

      // Period for annualization
      const startDate = trades[0]?.entry_date || "";
      const endDate = trades[trades.length - 1]?.exit_date || "";
      const totalDays = daysBetween(startDate, endDate);
      const years = totalDays / 365;

      // Annualized return
      const annualizedReturn = years > 0
        ? (Math.pow(finalValue / initialCapital, 1 / years) - 1) * 100 : 0;

      // Build equity curve from sequential trades
      let cumValue = initialCapital;
      let peak = initialCapital;
      let maxDrawdownPct = 0;
      let ddStartDay = 0;
      let maxDdDuration = 0;
      let currentDdStart = 0;
      const dailyReturns: number[] = [];
      const equityCurve = trades.map((t, i) => {
        const prevValue = cumValue;
        cumValue += (t.pnl_dollars || 0);
        const dailyReturn = prevValue > 0 ? (cumValue - prevValue) / prevValue : 0;
        dailyReturns.push(dailyReturn);

        if (cumValue > peak) {
          peak = cumValue;
          if (currentDdStart > 0) {
            maxDdDuration = Math.max(maxDdDuration, i - currentDdStart);
          }
          currentDdStart = i;
        }
        const dd = peak > 0 ? ((peak - cumValue) / peak) * 100 : 0;
        if (dd > maxDrawdownPct) maxDrawdownPct = dd;

        return {
          date: t.exit_date,
          portfolio_value: Math.round(cumValue * 100) / 100,
          drawdown: Math.round(dd * 100) / 100,
          cumulative_return: Math.round(((cumValue - initialCapital) / initialCapital) * 10000) / 100,
        };
      });

      // Volatility & risk-adjusted metrics
      const meanReturn = dailyReturns.length > 0
        ? dailyReturns.reduce((s, r) => s + r, 0) / dailyReturns.length : 0;
      const variance = dailyReturns.length > 1
        ? dailyReturns.reduce((s, r) => s + (r - meanReturn) ** 2, 0) / (dailyReturns.length - 1) : 0;
      const stdDev = Math.sqrt(variance);
      // Annualize using trades per year estimate
      const tradesPerYear = years > 0 ? trades.length / years : trades.length;
      const annualizedVol = stdDev * Math.sqrt(tradesPerYear) * 100;

      const sharpe = annualizedVol > 0 ? annualizedReturn / annualizedVol : 0;

      // Sortino (downside deviation)
      const negReturns = dailyReturns.filter((r) => r < 0);
      const downsideVar = negReturns.length > 0
        ? negReturns.reduce((s, r) => s + r ** 2, 0) / negReturns.length : 0;
      const downsideDev = Math.sqrt(downsideVar) * Math.sqrt(tradesPerYear) * 100;
      const sortino = downsideDev > 0 ? annualizedReturn / downsideDev : 0;

      // Calmar
      const calmar = maxDrawdownPct > 0 ? annualizedReturn / maxDrawdownPct : 0;

      return NextResponse.json({
        timestamp: trades[trades.length - 1]?.exit_date || new Date().toISOString(),
        strategy: "Ensemble ML",
        symbols: [...new Set(trades.map((t) => t.symbol))],
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        final_value: Math.round(finalValue * 100) / 100,
        metrics: {
          total_return_pct: Math.round(totalReturnPct * 100) / 100,
          annualized_return_pct: Math.round(annualizedReturn * 100) / 100,
          max_drawdown_pct: Math.round(maxDrawdownPct * 100) / 100,
          max_drawdown_duration_days: maxDdDuration,
          volatility_annualized: Math.round(annualizedVol * 100) / 100,
          sharpe_ratio: Math.round(sharpe * 100) / 100,
          sortino_ratio: Math.round(sortino * 100) / 100,
          calmar_ratio: Math.round(calmar * 100) / 100,
          total_trades: trades.length,
          winning_trades: wins.length,
          losing_trades: losses.length,
          win_rate: Math.round(winRate * 10000) / 10000,
          avg_win_pct: Math.round(avgWinPct * 100) / 100,
          avg_loss_pct: Math.round(avgLossPct * 100) / 100,
          largest_win_pct: Math.round(largestWinPct * 100) / 100,
          largest_loss_pct: Math.round(largestLossPct * 100) / 100,
          avg_hold_days: Math.round(avgHoldDays * 10) / 10,
          profit_factor: Math.round(profitFactor * 100) / 100,
          expectancy: Math.round(expectancy * 100) / 100,
        },
        trades: tradesWithHold.map((t) => ({
          symbol: t.symbol,
          direction: t.direction,
          entry_date: t.entry_date,
          entry_price: t.entry_price,
          exit_date: t.exit_date,
          exit_price: t.exit_price,
          shares: t.shares,
          pnl_dollars: t.pnl_dollars,
          return_pct: t.pnl_pct,
          hold_days: t.hold_days,
          exit_reason: t.exit_reason,
          entry_score: t.entry_score || 0,
        })),
        equity_curve: equityCurve,
        parameters: {},
        source: "supabase",
      });
    }

    return NextResponse.json(
      { error: "No backtest data available" },
      { status: 404 }
    );
  } catch (error) {
    console.error("Backtest API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch backtest data" },
      { status: 500 }
    );
  }
}
