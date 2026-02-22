# danger_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# IMPORTANT: risk_engine uses *_EVENT_WEIGHTS now; alias to keep detector stable.
from risk_engine import (
    OPPONENT_EVENT_WEIGHTS as OPPONENT_WEIGHTS,
    BARCA_EVENT_WEIGHTS as BARCA_WEIGHTS,
)


@dataclass(frozen=True)
class DangerMoment:
    start_s: float
    end_s: float
    peak_s: float
    peak_score: float
    severity: str
    active_event_codes: List[str]
    meta: Dict[str, Any]


def _severity_from_score(score: float) -> str:
    if score >= 80:
        return "high"
    if score >= 50:
        return "moderate"
    if score >= 25:
        return "low"
    return "very_low"


def detect_danger_moments(
    risk_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    *,
    match_name: Optional[str] = None,
    debug: bool = False,
    threshold: float = 45.0,
    min_gap_s: float = 12.0,
    min_duration_s: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Find danger windows based on risk_score peaks.
    Expects risk_df columns: time_s, risk_score, active_event_codes.
    Returns list of dicts (JSON safe).
    """
    if risk_df is None or risk_df.empty:
        return []

    df = risk_df.copy()

    # Defensive: ensure columns exist
    if "time_s" not in df.columns or "risk_score" not in df.columns:
        return []

    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df = df[np.isfinite(df["time_s"]) & np.isfinite(df["risk_score"])].sort_values("time_s").reset_index(drop=True)
    if df.empty:
        return []

    times = df["time_s"].to_numpy(dtype=float)
    scores = df["risk_score"].to_numpy(dtype=float)

    # Identify segments above threshold
    above = scores >= threshold
    if not above.any():
        return []

    segments = []
    start_idx = None
    for i, is_above in enumerate(above):
        if is_above and start_idx is None:
            start_idx = i
        if (not is_above) and start_idx is not None:
            segments.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(df) - 1))

    # Convert segments to windows + merge close ones
    windows = []
    for a, b in segments:
        t0, t1 = float(times[a]), float(times[b])
        if (t1 - t0) < min_duration_s:
            continue
        windows.append([t0, t1])

    # Merge windows that are close
    merged = []
    for w in windows:
        if not merged:
            merged.append(w)
            continue
        if w[0] - merged[-1][1] <= min_gap_s:
            merged[-1][1] = max(merged[-1][1], w[1])
        else:
            merged.append(w)

    out: List[Dict[str, Any]] = []
    for t0, t1 in merged:
        mask = (df["time_s"] >= t0) & (df["time_s"] <= t1)
        chunk = df.loc[mask]
        if chunk.empty:
            continue

        peak_idx = int(chunk["risk_score"].idxmax())
        peak_s = float(df.loc[peak_idx, "time_s"])
        peak_score = float(df.loc[peak_idx, "risk_score"])
        severity = _severity_from_score(peak_score)

        # Active event codes at peak, if available
        active_codes: List[str] = []
        if "active_event_codes" in df.columns:
            val = df.loc[peak_idx, "active_event_codes"]
            if isinstance(val, list):
                active_codes = [str(x) for x in val]
            elif pd.notna(val):
                active_codes = [str(val)]

        out.append(
            {
                "match_name": match_name,
                "danger_window": {"start_s": float(t0), "end_s": float(t1)},
                "peak": {"time_s": peak_s, "score": peak_score},
                "severity": severity,
                "active_event_codes": active_codes,
            }
        )

    if debug:
        print(f"[detect_danger_moments] {match_name=} found {len(out)} windows")

    return out