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
      - risk_score (0-100 scaled)
      - active_event_codes (list)
    """
    events_cleaned = _clean_events(events_df)

    time_grid = _build_time_grid(events_cleaned, dt_s=config.dt_s)

    raw_scores, active_events = _compute_raw_scores(
        time_grid,
        events_cleaned,
        opponent_weights=OPPONENT_EVENT_WEIGHTS,
        barca_weights=BARCA_EVENT_WEIGHTS,
    )

    # Smooth
    window_n = max(1, int(round(config.smooth_window_s / config.dt_s)))
    raw_smooth = _smooth(raw_scores, window_n=window_n)

    # Convert to 0-100 with a simple scaling
    # (You can change this later — it’s just a stable default)
    # Shift so min is 0
    shifted = raw_smooth - np.nanmin(raw_smooth) if len(raw_smooth) else raw_smooth
    maxv = float(np.nanmax(shifted)) if len(shifted) else 1.0
    if not np.isfinite(maxv) or maxv <= 1e-9:
        scaled = np.zeros_like(shifted)
    else:
        scaled = (shifted / maxv) * 100.0

    scaled = np.clip(scaled, config.clamp_min, config.clamp_max)

    out = pd.DataFrame(
        {
            "time_s": time_grid.astype(float),
            "raw_score": raw_smooth.astype(float),
            "risk_score": scaled.astype(float),
            "active_event_codes": active_events,
        }
    )
    return out