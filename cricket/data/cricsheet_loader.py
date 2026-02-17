"""
Cricsheet.org data loader.

Loads historical ball-by-ball match data from Cricsheet CSV files
for model training and backtesting.

Data source: https://cricsheet.org/downloads/
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from cricket.data.ball_event import BallEvent, ExtrasType, MatchInfo, WicketType

logger = logging.getLogger(__name__)

# Cricsheet CSV column mappings
EXTRAS_MAP = {
    "wides": ExtrasType.WIDE,
    "noballs": ExtrasType.NO_BALL,
    "byes": ExtrasType.BYE,
    "legbyes": ExtrasType.LEG_BYE,
    "penalty": ExtrasType.PENALTY,
}

WICKET_MAP = {
    "bowled": WicketType.BOWLED,
    "caught": WicketType.CAUGHT,
    "caught and bowled": WicketType.CAUGHT,
    "lbw": WicketType.LBW,
    "run out": WicketType.RUN_OUT,
    "stumped": WicketType.STUMPED,
    "hit wicket": WicketType.HIT_WICKET,
    "retired hurt": WicketType.RETIRED_HURT,
    "retired out": WicketType.RETIRED_OUT,
    "obstructing the field": WicketType.OBSTRUCTING,
}


def load_match_from_csv(csv_path: Path) -> tuple[MatchInfo, list[BallEvent]]:
    """Load a single match from a Cricsheet CSV file.

    Cricsheet CSV format has columns:
    match_id, season, start_date, venue, innings, ball, batting_team,
    bowling_team, striker, non_striker, bowler, runs_off_bat, extras,
    wides, noballs, byes, legbyes, penalty, wicket_type, player_dismissed

    Returns:
        Tuple of (MatchInfo, list of BallEvents in order)
    """
    events: list[BallEvent] = []
    match_info: Optional[MatchInfo] = None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV file: {csv_path}")

    # Extract match info from first row
    first = rows[0]
    match_id = first.get("match_id", csv_path.stem)
    teams = set()
    for row in rows:
        teams.add(row["batting_team"])
        if len(teams) == 2:
            break
    team_list = sorted(teams)

    match_info = MatchInfo(
        match_id=str(match_id),
        format=_infer_format(rows),
        team_a=team_list[0] if len(team_list) > 0 else "",
        team_b=team_list[1] if len(team_list) > 1 else "",
        venue=first.get("venue", ""),
        date=first.get("start_date", ""),
        season=first.get("season", ""),
    )

    # Track cumulative state per innings
    innings_score: dict[int, int] = {}
    innings_wickets: dict[int, int] = {}

    for row in rows:
        innings = int(row["innings"])
        ball_str = row["ball"]  # e.g. "5.3"

        if "." in ball_str:
            parts = ball_str.split(".")
            over = int(parts[0])
            ball_num = int(parts[1])
        else:
            over = int(float(ball_str))
            ball_num = 0

        runs_off_bat = int(row.get("runs_off_bat", 0))
        extras = int(row.get("extras", 0))
        total_runs = runs_off_bat + extras

        # Determine extras type
        extras_type = None
        for col, etype in EXTRAS_MAP.items():
            val = int(row.get(col, 0))
            if val > 0:
                extras_type = etype
                break

        # Determine wicket
        wicket_type_str = row.get("wicket_type", "").strip()
        is_wicket = bool(wicket_type_str)
        wicket_type = WICKET_MAP.get(wicket_type_str.lower()) if is_wicket else None
        player_dismissed = row.get("player_dismissed", "").strip() or None

        # Update cumulative state
        innings_score.setdefault(innings, 0)
        innings_wickets.setdefault(innings, 0)
        innings_score[innings] += total_runs
        if is_wicket and wicket_type != WicketType.RETIRED_HURT:
            innings_wickets[innings] += 1

        # Calculate cumulative overs (legal deliveries only)
        is_legal = extras_type not in (ExtrasType.WIDE, ExtrasType.NO_BALL)
        if is_legal:
            cumulative_overs = over + ball_num / 10.0
        else:
            # Non-legal delivery: use same over.ball as previous
            cumulative_overs = over + max(0, ball_num - 1) / 10.0

        event = BallEvent(
            match_id=str(match_id),
            innings=innings,
            over=over,
            ball=ball_num,
            batting_team=row["batting_team"],
            bowling_team=row["bowling_team"],
            striker=row.get("striker", ""),
            non_striker=row.get("non_striker", ""),
            bowler=row.get("bowler", ""),
            runs_off_bat=runs_off_bat,
            extras=extras,
            extras_type=extras_type,
            total_runs=total_runs,
            is_wicket=is_wicket,
            wicket_type=wicket_type,
            player_dismissed=player_dismissed,
            is_boundary_four=(runs_off_bat == 4),
            is_boundary_six=(runs_off_bat == 6),
            cumulative_score=innings_score[innings],
            cumulative_wickets=innings_wickets[innings],
            cumulative_overs=cumulative_overs,
        )
        events.append(event)

    logger.info(
        "Loaded match %s: %s vs %s, %d deliveries",
        match_id, match_info.team_a, match_info.team_b, len(events),
    )
    return match_info, events


def load_matches_from_directory(
    directory: Path,
    match_format: Optional[str] = None,
    max_matches: Optional[int] = None,
) -> list[tuple[MatchInfo, list[BallEvent]]]:
    """Load all matches from a directory of Cricsheet CSV files.

    Args:
        directory: Path to directory containing CSV files
        match_format: Filter by format (t20, odi, test)
        max_matches: Maximum number of matches to load

    Returns:
        List of (MatchInfo, BallEvents) tuples
    """
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", directory)
        return []

    matches = []
    for csv_file in csv_files:
        if max_matches and len(matches) >= max_matches:
            break
        try:
            info, events = load_match_from_csv(csv_file)
            if match_format and info.format != match_format:
                continue
            matches.append((info, events))
        except Exception as e:
            logger.warning("Failed to load %s: %s", csv_file.name, e)
            continue

    logger.info("Loaded %d matches from %s", len(matches), directory)
    return matches


def _infer_format(rows: list[dict]) -> str:
    """Infer match format from ball-by-ball data."""
    max_over = 0
    innings_set = set()
    for row in rows:
        ball_str = row.get("ball", "0")
        if "." in ball_str:
            over = int(ball_str.split(".")[0])
        else:
            over = int(float(ball_str))
        max_over = max(max_over, over)
        innings_set.add(int(row.get("innings", 1)))

    if len(innings_set) > 2:
        return "test"
    if max_over > 20:
        return "odi"
    return "t20"
