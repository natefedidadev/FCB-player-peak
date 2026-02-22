# generate_llm_insights.py
from __future__ import annotations

import argparse
import json
import pandas as pd
import math
from pathlib import Path
from typing import Optional

from data_loader import list_matches, load_events
from risk_engine import compute_risk_score
from danger_detector import detect_danger_moments
from pattern_analyzer import (
    build_all_matches_dangers_for_patterns,
    find_patterns,
    format_patterns_for_llm,
)
from explainer import explain_moment, explain_pattern
from tracking_features import summarize_window, load_team_map


PARSED_DIR = Path("parsed")


def _nan_to_none(obj):
    """
    Recursively replace float('nan') with None so JSON is clean.
    """
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj

def load_tracking_frames(match_name: str):
    """
    Load parsed tracking CSVs + team_map.json for a match, if available.
    Returns (player_df, ball_df, team_map) or (None, None, {}).
    """
    mdir = PARSED_DIR / match_name
    if not mdir.exists():
        return None, None, {}

    player_csv = mdir / "player_positions.csv"
    ball_csv = mdir / "ball_positions.csv"
    team_map = load_team_map(mdir)

    if not player_csv.exists():
        return None, None, team_map

    player_df = pd.read_csv(player_csv)

    # Ensure expected columns exist
    for col in ("time_s", "x", "y", "team_id"):
        if col not in player_df.columns:
            return None, None, team_map

    ball_df = None
    if ball_csv.exists():
        ball_df = pd.read_csv(ball_csv)

    # --- Normalize tracking times to seconds (robust heuristics) ---
    def _to_numeric_time(df, col="time_s"):
        if df is None or df.empty or col not in df.columns:
            return df
        df = df.copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    player_df = _to_numeric_time(player_df, "time_s")
    ball_df = _to_numeric_time(ball_df, "time_s") if ball_df is not None else None

    player_max = float(player_df["time_s"].max()) if "time_s" in player_df.columns else float("nan")
    ball_max = float(ball_df["time_s"].max()) if (ball_df is not None and "time_s" in ball_df.columns) else float("nan")

    # 1) Players: if huge, treat as milliseconds -> seconds
    if pd.notna(player_max) and player_max > 10000:
        player_df["time_s"] = player_df["time_s"] / 1000.0
        player_max = float(player_df["time_s"].max())

    # 2) Ball: if it looks like minutes (0–120-ish), convert to seconds
    if ball_df is not None and pd.notna(ball_max):
        # If ball clock is small but players (in seconds) extend much further, likely minutes
        if ball_max <= 200 and player_max >= 60 and (player_max / max(ball_max, 1e-6)) > 20:
            ball_df["time_s"] = ball_df["time_s"] * 60.0

        # Clip tiny negatives from sync noise
        ball_df["time_s"] = ball_df["time_s"].clip(lower=0.0)

    # ---- DEBUG (safe) ----
    print("\n=== TRACKING LOAD DEBUG ===")
    print("MATCH:", match_name)
    print("PLAYER PATH:", str(player_csv))
    print("PLAYER ROWS:", player_df.shape)
    print("PLAYER time_s range:", (player_df["time_s"].min(), player_df["time_s"].max()) if "time_s" in player_df.columns else None)

    print("BALL PATH:", str(ball_csv))
    if ball_df is None:
        print("BALL: missing")
    else:
        print("BALL ROWS:", ball_df.shape)
        if "match" in ball_df.columns:
            print("BALL MATCHES:", ball_df["match"].nunique())
        else:
            print("BALL MATCHES: no match col")

        # ensure numeric
        ball_df["time_s"] = pd.to_numeric(ball_df["time_s"], errors="coerce")
        print("BALL time_s non-null:", int(ball_df["time_s"].notna().sum()), "of", len(ball_df))
        print("BALL time_s max:", ball_df["time_s"].max())
        print("BALL time_s range:", (ball_df["time_s"].min(), ball_df["time_s"].max()))

        cols = [c for c in ["match", "time_s", "frame"] if c in ball_df.columns]
        if cols:
            print("BALL top 5 time_s rows:")
            print(ball_df.loc[ball_df["time_s"].notna(), cols].sort_values("time_s").tail(5).to_string(index=False))
    print("============================\n")

    return player_df, ball_df, team_map
OUT_DIR = Path("outputs/llm_insights")


def infer_opponent(match_name: str, barca_tokens: tuple[str, ...] = ("Barça", "FC Barcelona")) -> str:
    """
    Robust opponent inference from a match folder name like:
      "AC Milan - Barça (0-1) Partit ..."
      "Barça - AC Milan (2-2_ 3-4) Partit ..."
    Returns the side that is NOT Barça (cleaned).
    """
    if " - " not in match_name:
        return "Opponent"

    left, right = match_name.split(" - ", 1)

    def is_barca(s: str) -> bool:
        s2 = s.lower()
        return any(tok.lower() in s2 for tok in barca_tokens)

    def clean_team(s: str) -> str:
        # remove score-ish parentheticals e.g. "(0-1)" "(2-2_ 3-4)"
        # keep team name only
        s = s.strip()
        s = s.split("(")[0].strip()
        return s or "Opponent"

    if is_barca(left) and not is_barca(right):
        return clean_team(right)
    if is_barca(right) and not is_barca(left):
        return clean_team(left)

    # fallback: if ambiguous, take left as opponent label
    return clean_team(left)


def main(limit_matches: Optional[int], limit_dangers: Optional[int], top_n_patterns: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matches = list_matches()
    if limit_matches is not None:
        matches = matches[:limit_matches]

    # -------------------------
    # A) Per-match danger moments -> LLM explanations
    # -------------------------
    all_moment_outputs = []

    for match_name in matches:
        events_df = load_events(match_name)
        risk_df = compute_risk_score(events_df)

        opponent = infer_opponent(match_name)
        player_df, ball_df, team_map = load_tracking_frames(match_name)

        dangers = detect_danger_moments(
            risk_df,
            events_df,
            debug=False,
            match_name=match_name,
        )

        if limit_dangers is not None:
            dangers = dangers[:limit_dangers]

        for d in dangers:
            tracking_summary = None
            if player_df is not None:
                # window bounds for spatial summary
                w = d.get("danger_window", {})
                start_s = float(w.get("start_s", 0))
                end_s = float(w.get("end_s", 0))

                barca_id = (team_map or {}).get("barca_team_id")
                opp_id   = (team_map or {}).get("opponent_team_id")

                if barca_id is None or opp_id is None:
                    print("[TEAM MAP ERROR]", match_name, "team_map keys:", list((team_map or {}).keys()))
                    print("[TEAM MAP ERROR] team_map:", team_map)

                tracking_summary = summarize_window(
                    player_df,
                    ball_df,
                    start_s,
                    end_s,
                    preferred_time_s=float((d.get("peak") or {}).get("time_s")) if (d.get("peak") or {}).get("time_s") is not None else None,
                    defending_team_id=str((team_map or {}).get("barca_team_id")) if (team_map or {}).get("barca_team_id") is not None else None,
                    attacking_team_id=str((team_map or {}).get("opponent_team_id")) if (team_map or {}).get("opponent_team_id") is not None else None,
                )

            text = explain_moment(d, match_name, opponent, tracking_summary=tracking_summary)
            all_moment_outputs.append(
                {
                    "match_name": match_name,
                    "opponent": opponent,
                    "danger_moment": d,
                    "tracking_summary": tracking_summary,
                    "llm_response": text,
                }
            )

    clean_moments = _nan_to_none(all_moment_outputs)

    (OUT_DIR / "danger_moment_explanations.json").write_text(
        json.dumps(clean_moments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # -------------------------
    # B) Cross-match patterns -> LLM explanations
    # -------------------------
    baseline = build_all_matches_dangers_for_patterns(matches, mode="all")
    goal_dangers = build_all_matches_dangers_for_patterns(matches, mode="goals")

    patterns = find_patterns(goal_dangers, baseline_dangers=baseline)
    patterns_for_llm = format_patterns_for_llm(patterns, top_n=top_n_patterns)

    pattern_outputs = []
    for p in patterns_for_llm:
        text = explain_pattern(p)
        pattern_outputs.append({"pattern": p, "llm_response": text})

    clean_patterns = _nan_to_none(pattern_outputs)

    (OUT_DIR / "pattern_explanations.json").write_text(
        json.dumps(clean_patterns, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {OUT_DIR / 'danger_moment_explanations.json'}")
    print(f"Saved: {OUT_DIR / 'pattern_explanations.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_matches", type=int, default=None, help="Process only first N matches")
    parser.add_argument("--limit_dangers", type=int, default=None, help="Explain only first N dangers per match")
    parser.add_argument("--top_n_patterns", type=int, default=10, help="Top N patterns to explain")
    args = parser.parse_args()

    main(args.limit_matches, args.limit_dangers, args.top_n_patterns)