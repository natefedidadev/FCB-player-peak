from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from data_loader import list_matches, load_events
from risk_engine import compute_risk_score
from danger_detector import detect_danger_moments, BARCA_TEAM_NAME


def goals_against_barca(events_df: pd.DataFrame) -> int:
    if "code" not in events_df.columns:
        return 0
    goals = events_df[events_df["code"] == "GOALS"]
    if len(goals) == 0:
        return 0
    if "Team" in goals.columns:
        goals = goals[goals["Team"] != BARCA_TEAM_NAME]
    return int(len(goals))


def summarize_match(dangers: List[dict]) -> Dict[str, float]:
    if not dangers:
        return {
            "count": 0,
            "critical": 0,
            "high": 0,
            "moderate": 0,
            "avg_window_len": 0.0,
            "median_window_len": 0.0,
            "avg_peak": 0.0,
        }

    lens = [d["window_end"] - d["window_start"] for d in dangers]
    peaks = [d["peak_score"] for d in dangers]

    sev_counts = {"critical": 0, "high": 0, "moderate": 0}
    for d in dangers:
        sev_counts[d["severity"]] = sev_counts.get(d["severity"], 0) + 1

    return {
        "count": len(dangers),
        "critical": sev_counts.get("critical", 0),
        "high": sev_counts.get("high", 0),
        "moderate": sev_counts.get("moderate", 0),
        "avg_window_len": float(np.mean(lens)),
        "median_window_len": float(np.median(lens)),
        "avg_peak": float(np.mean(peaks)),
    }


def run_config(config: dict) -> pd.DataFrame:
    rows = []
    matches = list_matches()

    for i, name in enumerate(matches):
        events_df = load_events(i)
        risk_df = compute_risk_score(events_df)

        dangers = detect_danger_moments(
            risk_df,
            events_df,
            peak_percentile=config["peak_percentile"],
            threshold_floor=config["threshold_floor"],
            min_distance=config["min_distance"],
            prominence=config["prominence"],
            goal_lookback=config.get("goal_lookback", 90),
            context_window=config.get("context_window", 60),
        )

        g_against = goals_against_barca(events_df)
        goal_anchored = sum(1 for d in dangers if d.get("resulted_in_goal") is True)

        s = summarize_match(dangers)
        rows.append({
            "match": name,
            "goals_against": g_against,
            "goal_anchored_moments": goal_anchored,
            **s
        })

    df = pd.DataFrame(rows)

    # Add “global” summary at bottom
    summary = {
        "match": "ALL_MATCHES",
        "goals_against": int(df["goals_against"].sum()),
        "goal_anchored_moments": int(df["goal_anchored_moments"].sum()),
        "count": float(df["count"].mean()),
        "critical": float(df["critical"].mean()),
        "high": float(df["high"].mean()),
        "moderate": float(df["moderate"].mean()),
        "avg_window_len": float(df["avg_window_len"].mean()),
        "median_window_len": float(df["median_window_len"].mean()),
        "avg_peak": float(df["avg_peak"].mean()),
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    return df


def main():
    # Start with a reasonable baseline
    config = {
        "peak_percentile": 70,
        "threshold_floor": 40.0,
        "prominence": 10.0,
        "min_distance": 35
    }

    df = run_config(config)
    print("\nConfig:", config)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
