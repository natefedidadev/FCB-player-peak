# risk_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Event weights (these names are the "source of truth" now)
# -------------------------------------------------------------------

# Weights for opponent-driven danger (higher = more dangerous)
OPPONENT_EVENT_WEIGHTS: Dict[str, float] = {
    "ATTACKING TRANSITION": 1.25,
    "DEFENSIVE TRANSITION": 1.35,
    "BALL IN FINAL THIRD": 1.10,
    "BALL IN THE BOX": 1.55,
    "PLAYERS IN THE BOX": 1.15,
    "PLAYERS IN FINAL THIRD": 1.05,
    "COUNTER ATTACK": 1.35,
    "FAST BREAK": 1.30,
    "SET PIECES": 0.70,
}

# Weights for Barça being in control / reducing danger (higher = more control)
BARCA_EVENT_WEIGHTS: Dict[str, float] = {
    "POSSESSION": -0.35,
    "PROGRESSION": -0.20,
    "BUILD UP": -0.25,
    "FINAL THIRD": -0.15,
    "SUSTAINED ATTACK": -0.30,
}

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------


@dataclass(frozen=True)
class RiskConfig:
    dt_s: float = 0.25          # time step for the grid
    smooth_window_s: float = 3  # smoothing window for score
    clamp_min: float = 0.0
    clamp_max: float = 100.0


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _ensure_events_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the events dataframe has the columns we need:
      - Code, Team, Half
      - timestamp (timedelta)
      - duration (seconds)
      - Start, End (seconds)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Code", "Team", "Half", "timestamp", "duration", "Start", "End"])

    df = df.copy()

    # Normalize common column variants
    rename_map = {}
    if "code" in df.columns and "Code" not in df.columns:
        rename_map["code"] = "Code"
    if "team" in df.columns and "Team" not in df.columns:
        rename_map["team"] = "Team"
    if "half" in df.columns and "Half" not in df.columns:
        rename_map["half"] = "Half"
    if "start_s" in df.columns and "Start" not in df.columns:
        rename_map["start_s"] = "Start"
    if "end_s" in df.columns and "End" not in df.columns:
        rename_map["end_s"] = "End"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Required columns with defaults
    if "Code" not in df.columns:
        df["Code"] = "UNKNOWN"
    if "Team" not in df.columns:
        df["Team"] = "N/A"
    if "Half" not in df.columns:
        df["Half"] = "N/A"

    # Make sure we can derive Start/End consistently
    if "timestamp" not in df.columns:
        # If Start exists, build timestamp from it; else fallback to 0
        if "Start" in df.columns:
            df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
            df["timestamp"] = pd.to_timedelta(df["Start"].fillna(0.0), unit="s")
        else:
            df["timestamp"] = pd.to_timedelta(0.0, unit="s")

    # duration
    if "duration" not in df.columns:
        if "Start" in df.columns and "End" in df.columns:
            df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
            df["End"] = pd.to_numeric(df["End"], errors="coerce")
            df["duration"] = (df["End"] - df["Start"]).clip(lower=0.0)
        else:
            df["duration"] = 0.0

    # Start/End
    if "Start" not in df.columns:
        df["Start"] = df["timestamp"].dt.total_seconds()
    else:
        df["Start"] = pd.to_numeric(df["Start"], errors="coerce")

    if "End" not in df.columns:
        df["End"] = df["Start"] + pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    else:
        df["End"] = pd.to_numeric(df["End"], errors="coerce")

    # IMPORTANT: rescale Start/End into seconds if they look like ns/ms
    df = _rescale_time_columns(df)

    return df


def _rescale_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort fix for time units.

    Some event exports encode time in milliseconds or nanoseconds (e.g., 5.5e11).
    This function rescales Start/End so they're in *seconds*.

    Heuristic:
      - if max(Start, End) > 1e9  -> assume nanoseconds
      - elif > 1e6               -> assume milliseconds
      - else                     -> assume seconds
    """
    if df is None or df.empty:
        return df

    for c in ("Start", "End"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if not {"Start", "End"}.issubset(df.columns):
        return df

    tmax = float(np.nanmax(df[["Start", "End"]].to_numpy()))
    if not np.isfinite(tmax):
        return df

    scale = 1.0
    if tmax > 1e9:
        scale = 1e-9   # ns -> s
    elif tmax > 1e6:
        scale = 1e-3   # ms -> s

    if scale != 1.0:
        df["Start"] = df["Start"] * scale
        df["End"] = df["End"] * scale

    # Recompute derived columns consistently
    df["timestamp"] = pd.to_timedelta(df["Start"], unit="s")
    df["duration"] = (df["End"] - df["Start"]).clip(lower=0.0)

    return df


def _clean_events(events_df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_events_schema(events_df)

    # Drop totally broken times
    df = df[np.isfinite(df["Start"]) & np.isfinite(df["End"])].copy()

    # Ensure End >= Start
    df.loc[df["End"] < df["Start"], "End"] = df.loc[df["End"] < df["Start"], "Start"]

    # Minimal string cleanup
    df["Code"] = df["Code"].astype(str).str.strip()
    df["Team"] = df["Team"].astype(str).str.strip()
    df["Half"] = df["Half"].astype(str).str.strip()

    return df


def _build_time_grid(events_cleaned: pd.DataFrame, dt_s: float) -> np.ndarray:
    if events_cleaned is None or events_cleaned.empty:
        return np.arange(0, 1, dt_s)

    t_min = float(np.nanmin(events_cleaned["Start"].to_numpy()))
    t_max = float(np.nanmax(events_cleaned["End"].to_numpy()))

    # Clamp negative
    t_min = max(0.0, t_min)

    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return np.arange(0, 1, dt_s)

    # Guard against unit issues creating absurdly large grids
    span = t_max - t_min
    if span > 20000:  # > ~5.5 hours is not realistic for a single football match
        # Cap to keep the system usable (and avoid OOM). You can tune this if needed.
        t_max = t_min + 20000

    return np.arange(t_min, t_max + dt_s, dt_s)


def _compute_raw_scores(
    time_grid: np.ndarray,
    events_cleaned: pd.DataFrame,
    opponent_weights: Dict[str, float],
    barca_weights: Dict[str, float],
) -> Tuple[np.ndarray, List[List[str]]]:
    raw = np.zeros_like(time_grid, dtype=float)
    active: List[List[str]] = [[] for _ in range(len(time_grid))]

    if events_cleaned is None or events_cleaned.empty:
        return raw, active

    # Pre-extract arrays for speed
    starts = events_cleaned["Start"].to_numpy(dtype=float)
    ends = events_cleaned["End"].to_numpy(dtype=float)
    codes = events_cleaned["Code"].astype(str).to_numpy()
    teams = events_cleaned["Team"].astype(str).to_numpy()

    def weight_for(code: str, team: str) -> float:
        # "N/A" team means neutral (e.g. kickoff), treat as small/no effect
        if team == "N/A":
            return 0.0
        if team.lower() in ("fc barcelona", "barça", "barca"):
            return float(barca_weights.get(code, 0.0))
        return float(opponent_weights.get(code, 0.0))

    # For each event, add its weight over its active interval
    for s, e, c, tm in zip(starts, ends, codes, teams):
        if not np.isfinite(s) or not np.isfinite(e) or e <= s:
            continue
        w = weight_for(c, tm)
        if w == 0.0:
            continue

        mask = (time_grid >= s) & (time_grid <= e)
        raw[mask] += w

        # Track active event codes for debugging/explanations
        idxs = np.where(mask)[0]
        for i in idxs:
            active[i].append(c)

    return raw, active


def _smooth(x: np.ndarray, window_n: int) -> np.ndarray:
    if window_n <= 1 or len(x) == 0:
        return x
    w = np.ones(window_n, dtype=float) / float(window_n)
    return np.convolve(x, w, mode="same")


def compute_risk_score(events_df: pd.DataFrame, config: RiskConfig = RiskConfig()) -> pd.DataFrame:
    """
    Build a continuous risk score over time from pattern events.
    Returns dataframe with:
      - time_s
      - raw_score
      - risk_score (0-100 scaled, ABSOLUTE scale)
      - active_event_codes (list)

    Fixes:
    - Removes per-match min/max normalization (which caused random 100/100 spikes).
    - Forces OPPONENT GOAL events (goals conceded) to spike to 100/100.
    - Clamps implausible event times that can push peaks past the match end.
    """
    events_cleaned = _clean_events(events_df)

    # -----------------------------
    # 1) Clamp event times (guards)
    # -----------------------------
    CLAMP_MATCH_MAX_S = 95 * 60  # 95:00 (tune if you want 100/105)
    if events_cleaned is not None and not events_cleaned.empty:
        events_cleaned = events_cleaned.copy()
        events_cleaned["Start"] = pd.to_numeric(events_cleaned["Start"], errors="coerce")
        events_cleaned["End"] = pd.to_numeric(events_cleaned["End"], errors="coerce")
        events_cleaned = events_cleaned[np.isfinite(events_cleaned["Start"]) & np.isfinite(events_cleaned["End"])].copy()

        events_cleaned["Start"] = events_cleaned["Start"].clip(lower=0.0, upper=float(CLAMP_MATCH_MAX_S))
        events_cleaned["End"] = events_cleaned["End"].clip(lower=0.0, upper=float(CLAMP_MATCH_MAX_S))

        bad = events_cleaned["End"] < events_cleaned["Start"]
        if bad.any():
            events_cleaned.loc[bad, "End"] = events_cleaned.loc[bad, "Start"]

    time_grid = _build_time_grid(events_cleaned, dt_s=config.dt_s)

    raw_scores, active_events = _compute_raw_scores(
        time_grid,
        events_cleaned,
        opponent_weights=OPPONENT_EVENT_WEIGHTS,
        barca_weights=BARCA_EVENT_WEIGHTS,
    )

    # -----------------------------
    # 2) Smooth raw risk
    # -----------------------------
    window_n = max(1, int(round(config.smooth_window_s / config.dt_s)))
    raw_smooth = _smooth(raw_scores, window_n=window_n)

    # -----------------------------
    # 3) ABSOLUTE scaling (no per-match normalization)
    # -----------------------------
    TOP_K = 6
    opp_top = sorted([float(v) for v in OPPONENT_EVENT_WEIGHTS.values()], reverse=True)[:TOP_K]
    bar_top = sorted([float(v) for v in BARCA_EVENT_WEIGHTS.values()], reverse=True)[:TOP_K]
    abs_max = float(sum(opp_top) + sum(bar_top))
    if not np.isfinite(abs_max) or abs_max <= 1e-9:
        abs_max = 1.0

    raw_pos = np.clip(raw_smooth, 0.0, None)
    scaled = (raw_pos / abs_max) * 100.0
    scaled = np.clip(scaled, config.clamp_min, config.clamp_max)

    # -----------------------------
    # 4) Force OPPONENT GOAL spikes to 100 (goals conceded)
    # -----------------------------
    GOAL_SPIKE_RADIUS_S = 3.0  # +/- seconds around the goal timestamp
    GOAL_CODE_KEYWORDS = ("goal",)  # case-insensitive substring match

    # Names we will treat as Barça if the "Team" field is a string label
    BARCA_NAMES = {"fc barcelona", "barcelona", "barça", "fcb"}

    if events_cleaned is not None and not events_cleaned.empty and len(time_grid):
        codes = events_cleaned["Code"].astype(str).str.lower()
        is_goal_code = codes.apply(lambda s: any(k in s for k in GOAL_CODE_KEYWORDS))

        # Find a usable team column
        team_col = None
        for candidate in ["Team", "team", "team_name", "TeamName"]:
            if candidate in events_cleaned.columns:
                team_col = candidate
                break

        if team_col is not None and is_goal_code.any():
            teams = events_cleaned[team_col].astype(str).str.lower().str.strip()

            is_known_team = (teams != "") & (teams != "n/a") & (teams != "na") & (teams != "none")
            is_barca_team = teams.isin(BARCA_NAMES)

            # Only spike opponent goals (goals conceded by Barça)
            is_opponent_goal = is_goal_code & is_known_team & (~is_barca_team)

            if is_opponent_goal.any():
                g_start = pd.to_numeric(events_cleaned.loc[is_opponent_goal, "Start"], errors="coerce")
                g_end   = pd.to_numeric(events_cleaned.loc[is_opponent_goal, "End"], errors="coerce")

                # Prefer End (often closer to the actual goal moment), else midpoint, else Start
                goal_times = np.where(
                    np.isfinite(g_end.to_numpy()),
                    g_end.to_numpy(),
                    np.where(
                        np.isfinite(((g_start + g_end) / 2).to_numpy()),
                        ((g_start + g_end) / 2).to_numpy(),
                        g_start.to_numpy()
                    )
                )

                for gt in goal_times:
                    if not np.isfinite(gt):
                        continue
                    mask = np.abs(time_grid - float(gt)) <= GOAL_SPIKE_RADIUS_S
                    if not mask.any():
                        continue

                    scaled[mask] = config.clamp_max  # force 100 for conceded goals
                    raw_smooth[mask] = np.maximum(raw_smooth[mask], abs_max)

                    idxs = np.where(mask)[0]
                    for i in idxs:
                        active_events[i].append("GOAL_CONCEDED")

    out = pd.DataFrame(
        {
            "time_s": time_grid.astype(float),
            "raw_score": raw_smooth.astype(float),
            "risk_score": scaled.astype(float),
            "active_event_codes": active_events,
        }
    )
    return out