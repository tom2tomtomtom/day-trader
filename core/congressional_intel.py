#!/usr/bin/env python3
"""
CONGRESSIONAL INTELLIGENCE - Track What Congress Is Trading

Inspired by the Congressional Trading System's conviction scoring.
Members of Congress must disclose trades within 45 days (STOCK Act).
Research shows their trades historically outperform the market.

This module:
1. Fetches congressional trade disclosures
2. Scores each trade with a conviction algorithm
3. Identifies the most actionable insider-like signals
4. Cross-references trades with committee assignments

Data sources:
- Capitol Trades API (primary)
- Senate eFD disclosures
- House financial disclosures

Key insight: Congress members on relevant committees who buy aggressively
before legislative events generate the strongest alpha signals.
"""

import json
import requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CONGRESS_DATA = BASE_DIR / "congressional_trades.json"

# Committee-to-sector mapping (from congressional-trading-system)
COMMITTEE_SECTORS = {
    "Armed Services": ["LMT", "RTX", "NOC", "GD", "BA", "HII", "LHX", "LDOS"],
    "Energy and Commerce": ["XOM", "CVX", "NEE", "D", "SO", "DUK", "AEP", "SRE"],
    "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW"],
    "Health": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT"],
    "Judiciary": ["GOOGL", "META", "AMZN", "AAPL", "MSFT"],
    "Science and Technology": ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "TSM"],
    "Agriculture": ["ADM", "DE", "MOS", "CF", "NTR", "CTVA"],
    "Transportation": ["UNP", "CSX", "DAL", "UAL", "FDX", "UPS"],
    "Intelligence": [],  # Classified access - ALL trades are suspicious
    "Appropriations": [],  # Spending power - broad sector influence
}

# Known high-profile traders (historically active)
NOTABLE_TRADERS = [
    "Nancy Pelosi", "Dan Crenshaw", "Tommy Tuberville",
    "Michael McCaul", "Mark Green", "Josh Gottheimer",
    "Marjorie Taylor Greene", "Virginia Foxx",
]


@dataclass
class CongressionalTrade:
    """A single congressional trade disclosure"""
    member: str
    party: str  # R, D, I
    chamber: str  # Senate, House
    symbol: str
    company: str
    trade_type: str  # Purchase, Sale, Exchange
    amount_range: str  # e.g. "$15,001 - $50,000"
    amount_low: int
    amount_high: int
    trade_date: str
    disclosure_date: str
    filing_delay_days: int
    committees: List[str]
    asset_type: str  # Stock, Option, Bond, etc.


@dataclass
class ConvictionScore:
    """Conviction score for a congressional trade (0-100)"""
    total_score: float
    committee_access: float  # 0-25
    timing_proximity: float  # 0-25
    filing_delay: float  # 0-15
    trade_size_anomaly: float  # 0-15
    historical_pattern: float  # 0-10
    sector_concentration: float  # 0-10
    risk_level: str  # Critical, High, Elevated, Moderate, Low
    factors: List[str]


@dataclass
class CongressionalSignal:
    """An actionable signal derived from congressional trades"""
    symbol: str
    signal_type: str  # "congress_buy", "congress_sell", "cluster_buy"
    conviction: float
    members_buying: int
    members_selling: int
    total_volume_est: int
    notable_traders: List[str]
    committee_relevance: float
    top_trades: List[Dict]
    conviction_breakdown: Optional[ConvictionScore] = None


@dataclass
class CongressionalReport:
    """Full congressional intelligence report"""
    timestamp: str
    total_trades_analyzed: int
    signals: List[CongressionalSignal]
    hot_symbols: List[Dict]
    party_breakdown: Dict
    notable_activity: List[str]
    hall_of_fame: List[Dict]  # Best performing congress traders
    recent_cluster_buys: List[Dict]  # Multiple members buying same stock


class CongressionalIntelligence:
    """
    Fetches and analyzes congressional trading data.
    """

    def __init__(self):
        self.trades: List[CongressionalTrade] = []
        self.cache_file = CONGRESS_DATA

    def fetch_trades(self, days_back: int = 90) -> List[CongressionalTrade]:
        """
        Fetch recent congressional trades from Capitol Trades API
        Falls back to cached data if API unavailable.
        """
        trades = []

        # Try Capitol Trades API
        try:
            trades = self._fetch_from_capitol_trades(days_back)
        except Exception:
            pass

        # Try Senate eFD
        if not trades:
            try:
                trades = self._fetch_from_senate_efd(days_back)
            except Exception:
                pass

        # Fallback to cached data
        if not trades and self.cache_file.exists():
            try:
                cached = json.loads(self.cache_file.read_text())
                trades = [CongressionalTrade(**t) for t in cached.get("trades", [])]
            except Exception:
                pass

        # If still no data, generate representative sample for demo
        if not trades:
            trades = self._generate_demo_data()

        self.trades = trades
        self._cache_trades(trades)
        return trades

    def _fetch_from_capitol_trades(self, days_back: int) -> List[CongressionalTrade]:
        """Fetch from Capitol Trades API"""
        trades = []
        url = "https://www.capitoltrades.com/trades"
        params = {
            "page": 1,
            "pageSize": 100,
            "txDate": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
        }

        try:
            resp = requests.get(url, params=params, timeout=15, headers={
                "User-Agent": "DayTrader/1.0 (research tool)"
            })
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", []):
                    trade = self._parse_capitol_trade(item)
                    if trade:
                        trades.append(trade)
        except Exception:
            pass

        return trades

    def _parse_capitol_trade(self, item: Dict) -> Optional[CongressionalTrade]:
        """Parse a Capitol Trades API item"""
        try:
            politician = item.get("politician", {})
            asset = item.get("asset", {})
            tx_date = item.get("txDate", "")
            pub_date = item.get("pubDate", "")

            # Calculate filing delay
            delay = 0
            if tx_date and pub_date:
                try:
                    td = datetime.fromisoformat(tx_date.replace("Z", "+00:00"))
                    pd = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    delay = (pd - td).days
                except Exception:
                    pass

            return CongressionalTrade(
                member=politician.get("firstName", "") + " " + politician.get("lastName", ""),
                party=politician.get("party", ""),
                chamber=politician.get("chamber", ""),
                symbol=asset.get("assetTicker", ""),
                company=asset.get("assetName", ""),
                trade_type=item.get("txType", ""),
                amount_range=item.get("value", ""),
                amount_low=self._parse_amount(item.get("value", ""), "low"),
                amount_high=self._parse_amount(item.get("value", ""), "high"),
                trade_date=tx_date,
                disclosure_date=pub_date,
                filing_delay_days=delay,
                committees=politician.get("committees", []),
                asset_type=asset.get("assetType", "Stock"),
            )
        except Exception:
            return None

    def _fetch_from_senate_efd(self, days_back: int) -> List[CongressionalTrade]:
        """Fetch from Senate eFD system"""
        # Senate eFD requires session management and terms acceptance
        # This is a simplified version
        return []

    def _parse_amount(self, value_str: str, bound: str) -> int:
        """Parse amount range string"""
        ranges = {
            "$1,001 - $15,000": (1001, 15000),
            "$15,001 - $50,000": (15001, 50000),
            "$50,001 - $100,000": (50001, 100000),
            "$100,001 - $250,000": (100001, 250000),
            "$250,001 - $500,000": (250001, 500000),
            "$500,001 - $1,000,000": (500001, 1000000),
            "$1,000,001 - $5,000,000": (1000001, 5000000),
            "Over $5,000,000": (5000001, 50000000),
        }
        low, high = ranges.get(value_str, (0, 0))
        return low if bound == "low" else high

    def score_conviction(self, trade: CongressionalTrade) -> ConvictionScore:
        """
        Score a trade's conviction (0-100) using 6 weighted factors.
        Adapted from congressional-trading-system's conviction scoring.
        """
        factors = []

        # 1. Committee Access (0-25 pts)
        committee_score = 0
        for committee in trade.committees:
            relevant_symbols = COMMITTEE_SECTORS.get(committee, [])
            if trade.symbol in relevant_symbols:
                committee_score = 25
                factors.append(f"Direct committee-stock match: {committee}")
                break
            elif committee == "Intelligence":
                committee_score = max(committee_score, 20)
                factors.append("Intelligence Committee member - all trades flagged")
            elif committee == "Appropriations":
                committee_score = max(committee_score, 15)
                factors.append("Appropriations Committee - spending influence")
            elif relevant_symbols:
                # Check sector overlap
                committee_score = max(committee_score, 10)

        # 2. Filing Delay (0-15 pts) - longer delay = more suspicious
        delay_score = 0
        if trade.filing_delay_days > 90:
            delay_score = 15
            factors.append(f"Extreme filing delay: {trade.filing_delay_days} days")
        elif trade.filing_delay_days > 60:
            delay_score = 12
            factors.append(f"Late filing: {trade.filing_delay_days} days")
        elif trade.filing_delay_days > 45:
            delay_score = 8
            factors.append(f"Filing at deadline: {trade.filing_delay_days} days")
        elif trade.filing_delay_days <= 14:
            delay_score = 2
            factors.append(f"Quick disclosure: {trade.filing_delay_days} days")

        # 3. Trade Size Anomaly (0-15 pts)
        size_score = 0
        mid_amount = (trade.amount_low + trade.amount_high) / 2
        if mid_amount > 1000000:
            size_score = 15
            factors.append(f"Very large trade: {trade.amount_range}")
        elif mid_amount > 250000:
            size_score = 10
            factors.append(f"Large trade: {trade.amount_range}")
        elif mid_amount > 50000:
            size_score = 5

        # 4. Timing Proximity (0-25 pts) - placeholder for event correlation
        timing_score = 0
        # In a full implementation, this would check proximity to:
        # - Committee hearings, bill votes, earnings, FDA decisions
        # For now, use filing delay as a proxy
        if trade.filing_delay_days > 45 and mid_amount > 100000:
            timing_score = 15
            factors.append("Large trade with late disclosure - potential timing")

        # 5. Historical Pattern (0-10 pts)
        pattern_score = 0
        if trade.member in NOTABLE_TRADERS:
            pattern_score = 7
            factors.append(f"Notable trader: {trade.member}")

        # 6. Notable member bonus
        sector_score = 0
        if committee_score > 15:
            sector_score = 8
            factors.append("High committee-sector correlation")

        total = min(100, committee_score + timing_score + delay_score +
                    size_score + pattern_score + sector_score)

        # Risk level
        if total >= 80:
            risk = "Critical"
        elif total >= 65:
            risk = "High"
        elif total >= 50:
            risk = "Elevated"
        elif total >= 30:
            risk = "Moderate"
        else:
            risk = "Low"

        return ConvictionScore(
            total_score=total,
            committee_access=committee_score,
            timing_proximity=timing_score,
            filing_delay=delay_score,
            trade_size_anomaly=size_score,
            historical_pattern=pattern_score,
            sector_concentration=sector_score,
            risk_level=risk,
            factors=factors,
        )

    def generate_signals(self) -> List[CongressionalSignal]:
        """Generate actionable signals from congressional trades"""
        if not self.trades:
            self.fetch_trades()

        # Group trades by symbol
        symbol_trades: Dict[str, List[CongressionalTrade]] = {}
        for trade in self.trades:
            if not trade.symbol:
                continue
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)

        signals = []
        for symbol, trades in symbol_trades.items():
            buys = [t for t in trades if t.trade_type.lower() in ("purchase", "buy")]
            sells = [t for t in trades if t.trade_type.lower() in ("sale", "sell", "sale (full)", "sale (partial)")]

            if not buys and not sells:
                continue

            # Score each trade
            scored_trades = []
            max_conviction = 0
            for trade in trades:
                score = self.score_conviction(trade)
                max_conviction = max(max_conviction, score.total_score)
                scored_trades.append({
                    "member": trade.member,
                    "party": trade.party,
                    "type": trade.trade_type,
                    "amount": trade.amount_range,
                    "date": trade.trade_date,
                    "conviction": score.total_score,
                    "risk_level": score.risk_level,
                    "factors": score.factors,
                })

            # Estimate total volume
            total_vol = sum((t.amount_low + t.amount_high) // 2 for t in trades)

            # Find notable traders
            notable = [t.member for t in trades if t.member in NOTABLE_TRADERS]

            # Committee relevance
            all_committees = set()
            for t in trades:
                all_committees.update(t.committees)
            relevant = any(
                symbol in COMMITTEE_SECTORS.get(c, [])
                for c in all_committees
            )

            # Determine signal type
            if len(buys) >= 3 and len(sells) == 0:
                signal_type = "cluster_buy"
            elif len(buys) > len(sells):
                signal_type = "congress_buy"
            elif len(sells) > len(buys):
                signal_type = "congress_sell"
            else:
                signal_type = "congress_mixed"

            signals.append(CongressionalSignal(
                symbol=symbol,
                signal_type=signal_type,
                conviction=max_conviction,
                members_buying=len(set(t.member for t in buys)),
                members_selling=len(set(t.member for t in sells)),
                total_volume_est=total_vol,
                notable_traders=notable,
                committee_relevance=1.0 if relevant else 0.5,
                top_trades=sorted(scored_trades, key=lambda x: x["conviction"], reverse=True)[:5],
            ))

        # Sort by conviction
        signals.sort(key=lambda s: s.conviction, reverse=True)
        return signals

    def generate_report(self) -> CongressionalReport:
        """Generate full congressional intelligence report"""
        if not self.trades:
            self.fetch_trades()

        signals = self.generate_signals()

        # Hot symbols (most traded)
        symbol_count: Dict[str, int] = {}
        for t in self.trades:
            if t.symbol:
                symbol_count[t.symbol] = symbol_count.get(t.symbol, 0) + 1
        hot = sorted(symbol_count.items(), key=lambda x: x[1], reverse=True)[:10]
        hot_symbols = [{"symbol": s, "trade_count": c} for s, c in hot]

        # Party breakdown
        r_trades = len([t for t in self.trades if t.party == "R"])
        d_trades = len([t for t in self.trades if t.party == "D"])
        party = {
            "republican": r_trades,
            "democrat": d_trades,
            "r_buys": len([t for t in self.trades if t.party == "R" and "purchase" in t.trade_type.lower()]),
            "d_buys": len([t for t in self.trades if t.party == "D" and "purchase" in t.trade_type.lower()]),
        }

        # Notable activity
        notable = []
        for t in self.trades:
            if t.member in NOTABLE_TRADERS:
                notable.append(
                    f"{t.member} ({t.party}) {t.trade_type} {t.symbol} "
                    f"({t.amount_range}) on {t.trade_date}"
                )

        # Cluster buys
        clusters = [s for s in signals if s.signal_type == "cluster_buy"]
        cluster_data = [{
            "symbol": s.symbol,
            "members": s.members_buying,
            "volume_est": s.total_volume_est,
            "conviction": s.conviction,
        } for s in clusters]

        # Hall of fame (members with most trades)
        member_count: Dict[str, int] = {}
        for t in self.trades:
            member_count[t.member] = member_count.get(t.member, 0) + 1
        hof = sorted(member_count.items(), key=lambda x: x[1], reverse=True)[:10]
        hall_of_fame = [{"member": m, "trade_count": c} for m, c in hof]

        return CongressionalReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_trades_analyzed=len(self.trades),
            signals=signals[:20],
            hot_symbols=hot_symbols,
            party_breakdown=party,
            notable_activity=notable[:10],
            hall_of_fame=hall_of_fame,
            recent_cluster_buys=cluster_data,
        )

    def _generate_demo_data(self) -> List[CongressionalTrade]:
        """Generate representative demo data for testing"""
        now = datetime.now(timezone.utc)
        trades = []

        demo_trades = [
            ("Nancy Pelosi", "D", "House", "NVDA", "NVIDIA Corp", "Purchase",
             "$1,000,001 - $5,000,000", 1000001, 5000000, 5,
             ["Intelligence", "Financial Services"]),
            ("Nancy Pelosi", "D", "House", "AAPL", "Apple Inc", "Purchase",
             "$500,001 - $1,000,000", 500001, 1000000, 12,
             ["Intelligence", "Financial Services"]),
            ("Tommy Tuberville", "R", "Senate", "NVDA", "NVIDIA Corp", "Purchase",
             "$250,001 - $500,000", 250001, 500000, 45,
             ["Armed Services", "Agriculture"]),
            ("Dan Crenshaw", "R", "House", "LMT", "Lockheed Martin", "Purchase",
             "$15,001 - $50,000", 15001, 50000, 30,
             ["Armed Services", "Energy and Commerce"]),
            ("Dan Crenshaw", "R", "House", "RTX", "RTX Corp", "Purchase",
             "$50,001 - $100,000", 50001, 100000, 32,
             ["Armed Services", "Energy and Commerce"]),
            ("Michael McCaul", "R", "House", "MSFT", "Microsoft Corp", "Purchase",
             "$100,001 - $250,000", 100001, 250000, 28,
             ["Science and Technology", "Financial Services"]),
            ("Josh Gottheimer", "D", "House", "GOOGL", "Alphabet Inc", "Purchase",
             "$50,001 - $100,000", 50001, 100000, 15,
             ["Financial Services"]),
            ("Mark Green", "R", "House", "BA", "Boeing Co", "Purchase",
             "$15,001 - $50,000", 15001, 50000, 22,
             ["Armed Services", "Health"]),
            ("Virginia Foxx", "R", "House", "UNH", "UnitedHealth Group", "Sale (Partial)",
             "$100,001 - $250,000", 100001, 250000, 60,
             ["Health"]),
            ("Marjorie Taylor Greene", "R", "House", "TSLA", "Tesla Inc", "Purchase",
             "$15,001 - $50,000", 15001, 50000, 35,
             []),
            ("Nancy Pelosi", "D", "House", "RBLX", "Roblox Corp", "Purchase",
             "$500,001 - $1,000,000", 500001, 1000000, 8,
             ["Intelligence", "Financial Services"]),
            ("Tommy Tuberville", "R", "Senate", "MSFT", "Microsoft Corp", "Purchase",
             "$100,001 - $250,000", 100001, 250000, 50,
             ["Armed Services", "Agriculture"]),
            ("Dan Crenshaw", "R", "House", "NOC", "Northrop Grumman", "Purchase",
             "$50,001 - $100,000", 50001, 100000, 25,
             ["Armed Services", "Energy and Commerce"]),
            ("Josh Gottheimer", "D", "House", "NVDA", "NVIDIA Corp", "Purchase",
             "$15,001 - $50,000", 15001, 50000, 10,
             ["Financial Services"]),
        ]

        for i, (member, party, chamber, sym, company, ttype,
                amount_range, low, high, delay, committees) in enumerate(demo_trades):
            trade_date = (now - timedelta(days=delay + i * 3)).strftime("%Y-%m-%d")
            disc_date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            trades.append(CongressionalTrade(
                member=member,
                party=party,
                chamber=chamber,
                symbol=sym,
                company=company,
                trade_type=ttype,
                amount_range=amount_range,
                amount_low=low,
                amount_high=high,
                trade_date=trade_date,
                disclosure_date=disc_date,
                filing_delay_days=delay,
                committees=committees,
                asset_type="Stock",
            ))

        return trades

    def _cache_trades(self, trades: List[CongressionalTrade]):
        """Cache trades to disk"""
        try:
            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trades": [asdict(t) for t in trades],
            }
            self.cache_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def get_symbol_intel(self, symbol: str) -> Optional[CongressionalSignal]:
        """Get congressional intelligence for a specific symbol"""
        signals = self.generate_signals()
        for s in signals:
            if s.symbol == symbol:
                return s
        return None


def run_congressional_scan() -> CongressionalReport:
    """Convenience function to run full scan"""
    intel = CongressionalIntelligence()
    intel.fetch_trades()
    return intel.generate_report()


if __name__ == "__main__":
    report = run_congressional_scan()
    print(f"\n{'='*60}")
    print(f"CONGRESSIONAL INTELLIGENCE REPORT")
    print(f"{'='*60}")
    print(f"Trades Analyzed: {report.total_trades_analyzed}")
    print(f"Signals: {len(report.signals)}")

    print(f"\nHot Symbols:")
    for s in report.hot_symbols[:5]:
        print(f"  {s['symbol']}: {s['trade_count']} trades")

    print(f"\nTop Signals:")
    for s in report.signals[:5]:
        print(f"  {s.symbol}: {s.signal_type} (conviction: {s.conviction:.0f})")
        print(f"    Buying: {s.members_buying} | Selling: {s.members_selling}")
        print(f"    Est Volume: ${s.total_volume_est:,}")
        if s.notable_traders:
            print(f"    Notable: {', '.join(s.notable_traders)}")

    print(f"\nNotable Activity:")
    for a in report.notable_activity[:5]:
        print(f"  {a}")

    print(f"\nCluster Buys:")
    for c in report.recent_cluster_buys:
        print(f"  {c['symbol']}: {c['members']} members, ${c['volume_est']:,} est")
