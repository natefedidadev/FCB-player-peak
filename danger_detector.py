from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from risk_engine import OPPONENT_WEIGHTS, BARCA_WEIGHTS

# Update this if your events_df uses a different Barcelona name
BARCA_TEAM_NAME = "FC Barcelona"


# ----------------------------
# Weight helper (for diagnostics)
# ----------------------------

def _build_weight_lookup() -> Dict[str, float]:
    """
    We only have active_codes at the peak (no team attached),
    so for diagnostics we use the MAX weight a code can have
    across opponent + Barca dictionaries.
    """
    codes = set(OPPONENT_WEIGHTS.keys()) | set(BARCA_WEIGHTS.keys())
    return {c: float(max(OPPONENT_WEIGHTS.get(c, 0), BARCA_WEIGHTS.get(c, 0))) for c in codes}


WEIGHT_LOOKUP = _build_weight_lookup()


# ----------------------------
# Formatting / normalization helpers
# ----------------------------

def _format_nexus_timestamp(t_sec: int) -> str:
    """Format seconds -> mm:ss for Nexus timestamp."""
    m = int(t_sec) // 60
    s = int(t_sec) % 60
    return f"{m:02d}:{s:02d}"


def _normalize_active_events_to_codes(active_events: Any) -> List[str]:
    """
    Accepts:
      - list[str]
      - list[dict] where dict may contain 'code'
      - str
      - None / NaN
    Returns list[str] codes.
    """
    if active_events is None:
        return []

    # pandas may store NaN in object column
    try:
        if isinstance(active_events, float) and np.isnan(active_events):
            return []
    except Exception:
        pass

    # list case
    if isinstance(active_events, list):
        if not active_events:
            return []

        # list[str]
        if isinstance(active_events[0], str):
            return [str(x) for x in active_events]

        # list[dict] with {"code": "..."}
        if isinstance(active_events[0], dict):
            out: List[str] = []
            for e in active_events:
                if isinstance(e, dict) and e.get("code"):
                    out.append(str(e["code"]))
            return out

        # fallback
        return [str(x) for x in active_events]

    # single string
    if isinstance(active_events, str):
        return [active_events]

    return []


def _get_goal_timestamps_sec(
    events_df: pd.DataFrame,
    barca_team_name: str = BARCA_TEAM_NAME
) -> List[int]:
    """
    Return goal timestamps (seconds) for goals AGAINST Barcelona.

    Supports events_df['timestamp'] as:
      - timedelta64 (kloppy often uses this)
      - numeric seconds
      - datetime64 (converted relative to match start)
    """
    if events_df is None or len(events_df) == 0:
        return []

    if "code" not in events_df.columns or "timestamp" not in events_df.columns:
        return []

    goals = events_df[events_df["code"] == "GOALS"].copy()
    if len(goals) == 0:
        return []

    # Goals "against" Barca => Team != Barca (based on your convention)
    if "Team" in goals.columns:
        goals = goals[goals["Team"] != barca_team_name]
    if len(goals) == 0:
        return []

    ts = goals["timestamp"]

    # timedelta
    if pd.api.types.is_timedelta64_dtype(ts):
        return ts.dt.total_seconds().round().astype(int).tolist()

    # numeric seconds
    if pd.api.types.is_numeric_dtype(ts):
        return ts.round().astype(int).tolist()

    # datetime-like
    if pd.api.types.is_datetime64_any_dtype(ts):
        t0 = ts.min()
        return ((ts - t0).dt.total_seconds()).round().astype(int).tolist()

    # fallback: try parse
    try:
        parsed = pd.to_datetime(ts, errors="coerce")
        if parsed.notna().any():
            t0 = parsed.min()
            return ((parsed - t0).dt.total_seconds()).round().astype(int).tolist()
    except Exception:
        pass

    return []


# ----------------------------
# Window extraction
# ----------------------------

def _find_window_bounds(
    times: np.ndarray,
    risk: np.ndarray,
    peak_idx: int,
    threshold: float
) -> Tuple[int, int]:
    """
    Find contiguous window around peak where risk > threshold.
    Returns window_start_sec, window_end_sec (inclusive).
    """
    n = len(risk)
    i = int(peak_idx)

    # walk left while still above threshold
    left = i
    while left > 0 and risk[left] > threshold:
        left -= 1
    if risk[left] <= threshold and left < i:
        left += 1  # first index that is above threshold

    # walk right while still above threshold
    right = i
    while right < n - 1 and risk[right] > threshold:
        right += 1
    if risk[right] <= threshold and right > i:
        right -= 1

    return int(times[left]), int(times[right])


# ----------------------------
# Merge close moments
# ----------------------------

def merge_close_danger_moments(
    dangers: List[Dict[str, Any]],
    merge_within_sec: int = 60
) -> List[Dict[str, Any]]:
    """
    Merge danger moments within merge_within_sec seconds.
    Collapses repeated peaks during one sustained pressure spell.

    Merge rule:
      - peak_time/peak_score: keep the higher peak_score
      - window_start: min
      - window_end: max
      - active_codes: union (order preserved)
      - resulted_in_goal: True if any merged moment is True
      - severity: max severity across merged (critical > high > moderate)
    """
    
    if not dangers:
        return []

    dangers_sorted = sorted(dangers, key=lambda d: int(d["peak_time"]))
    sev_rank = {"critical": 3, "high": 2, "moderate": 1}

    def union_codes(a: List[str], b: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in (a or []) + (b or []):
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    merged: List[Dict[str, Any]] = []
    cur = dangers_sorted[0].copy()

    for nxt in dangers_sorted[1:]:
        cur_peak = int(cur["peak_time"])
        nxt_peak = int(nxt["peak_time"])

        if (nxt_peak - cur_peak) <= merge_within_sec:
            # merge windows
            cur["window_start"] = int(min(cur["window_start"], nxt["window_start"]))
            cur["window_end"] = int(max(cur["window_end"], nxt["window_end"]))

            # keep max peak
            if float(nxt["peak_score"]) > float(cur["peak_score"]):
                cur["peak_score"] = float(nxt["peak_score"])
                cur["peak_time"] = int(nxt["peak_time"])
                cur["nexus_timestamp"] = nxt.get("nexus_timestamp", cur.get("nexus_timestamp"))

            # merge codes
            cur["active_codes"] = union_codes(cur.get("active_codes", []), nxt.get("active_codes", []))

            # merge goal flag
            cur["resulted_in_goal"] = bool(cur.get("resulted_in_goal", False) or nxt.get("resulted_in_goal", False))

            # merge severity
            cur_sev = cur.get("severity", "moderate")
            nxt_sev = nxt.get("severity", "moderate")
            cur["severity"] = cur_sev if sev_rank.get(cur_sev, 0) >= sev_rank.get(nxt_sev, 0) else nxt_sev
        else:
            merged.append(cur)
            cur = nxt.copy()

    merged.append(cur)
    return merged


# ----------------------------
# Diagnostics helper
# ----------------------------

def _compute_top_contributors(active_codes: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Return top contributing codes by heuristic weight (diagnostic only).
    Uses WEIGHT_LOOKUP (max across Barca/opponent weights) because active_codes has no team info.
    """
    contrib = [(c, float(WEIGHT_LOOKUP.get(c, 0.0))) for c in (active_codes or [])]
    contrib.sort(key=lambda x: x[1], reverse=True)
    return contrib[:top_k]


# ----------------------------
# Main API
# ----------------------------

def detect_danger_moments(
    risk_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    peak_percentile: float = 70,
    threshold_floor: float = 40.0,
    min_distance: int = 35,         # tuned default
    prominence: float = 10.0,       # tuned default
    goal_lookback: int = 90,
    context_window: int = 60,       # Phase 3 fingerprinting
    merge_within_sec: Optional[int] = 60
) -> List[Dict[str, Any]]:
    """
    Identify danger moments (fault lines) from risk score timeline.

    Returns list of dicts with:
      peak_time, window_start, window_end, peak_score, severity,
      resulted_in_goal, active_codes, nexus_timestamp,
      threshold_used, raw_percentile_threshold, num_active_codes, top_contributors
    """
    if risk_df is None or len(risk_df) == 0:
        return []

    required = {"timestamp_sec", "risk_score"}
    missing = required - set(risk_df.columns)
    if missing:
        raise ValueError(f"risk_df missing required columns: {missing}")

    # Ensure sorted timeline
    risk_df = risk_df.sort_values("timestamp_sec").reset_index(drop=True)

    times = risk_df["timestamp_sec"].astype(int).to_numpy()
    risk_scores = risk_df["risk_score"].astype(float).to_numpy()

    # Threshold for peaks + windows (safe against percentile collapsing to 0)
    raw_percentile_thr = float(np.percentile(risk_scores, peak_percentile))
    threshold_used = max(raw_percentile_thr, float(threshold_floor))

    # Peak detection
    peaks, _props = find_peaks(
        risk_scores,
        height=threshold_used,
        distance=min_distance,
        prominence=prominence
    )

    danger_moments: List[Dict[str, Any]] = []

    # Peaks-based moments
    for peak_idx in peaks:
        peak_idx = int(peak_idx)
        peak_time = int(times[peak_idx])
        peak_score = float(risk_scores[peak_idx])

        window_start, window_end = _find_window_bounds(times, risk_scores, peak_idx, threshold_used)

        active_codes: List[str] = []
        if "active_events" in risk_df.columns:
            active_codes = _normalize_active_events_to_codes(risk_df.loc[peak_idx, "active_events"])

        # Severity assumes risk is normalized 0–100
        if peak_score >= 85:
            severity = "critical"
        elif peak_score >= 70:
            severity = "high"
        else:
            severity = "moderate"

        danger_moments.append({
            "peak_time": peak_time,
            "window_start": window_start,
            "window_end": window_end,
            "peak_score": peak_score,
            "severity": severity,
            "resulted_in_goal": False,
            "active_codes": active_codes,
            "nexus_timestamp": _format_nexus_timestamp(peak_time),

            # diagnostics (Phase 2 “why was this high?”)
            "threshold_used": float(threshold_used),
            "raw_percentile_threshold": float(raw_percentile_thr),
            "num_active_codes": int(len(active_codes)),
            "top_contributors": _compute_top_contributors(active_codes, top_k=3),
        })

    # Goal anchoring: look back and mark max-risk point in that window as critical+goal-related
    goal_times = _get_goal_timestamps_sec(events_df, barca_team_name=BARCA_TEAM_NAME)

    for gt in goal_times:
        start_t = max(int(gt) - int(goal_lookback), int(times[0]))
        end_t = min(int(gt), int(times[-1]))

        start_idx = int(np.searchsorted(times, start_t, side="left"))
        end_idx = int(np.searchsorted(times, end_t, side="right")) - 1
        if start_idx > end_idx:
            continue

        seg = risk_scores[start_idx:end_idx + 1]
        if len(seg) == 0:
            continue

        anchor_idx = start_idx + int(np.argmax(seg))
        peak_time = int(times[anchor_idx])
        peak_score = float(risk_scores[anchor_idx])

        window_start, window_end = _find_window_bounds(times, risk_scores, anchor_idx, threshold_used)

        active_codes: List[str] = []
        if "active_events" in risk_df.columns:
            active_codes = _normalize_active_events_to_codes(risk_df.loc[anchor_idx, "active_events"])

        # If within ±5s of an existing moment, promote it; else add
        merged_into_existing = False
        for m in danger_moments:
            if abs(int(m["peak_time"]) - peak_time) <= 5:
                m["resulted_in_goal"] = True
                m["severity"] = "critical"
                merged_into_existing = True
                break

        if not merged_into_existing:
            danger_moments.append({
                "peak_time": peak_time,
                "window_start": window_start,
                "window_end": window_end,
                "peak_score": peak_score,
                "severity": "critical",
                "resulted_in_goal": True,
                "active_codes": active_codes,
                "nexus_timestamp": _format_nexus_timestamp(peak_time),

                "threshold_used": float(threshold_used),
                "raw_percentile_threshold": float(raw_percentile_thr),
                "num_active_codes": int(len(active_codes)),
                "top_contributors": _compute_top_contributors(active_codes, top_k=3),
            })

    # Merge close moments to collapse pressure spells
    if merge_within_sec is not None and merge_within_sec > 0:
        danger_moments = merge_close_danger_moments(danger_moments, merge_within_sec=merge_within_sec)

    # Recompute diagnostics AFTER merge (merge changes active_codes)
    for m in danger_moments:
        codes = m.get("active_codes", []) or []

        m["num_active_codes"] = len(codes)

        contributions = [(c, float(max(OPPONENT_WEIGHTS.get(c, 0), BARCA_WEIGHTS.get(c, 0)))) for c in codes]
        contributions.sort(key=lambda x: x[1], reverse=True)
        m["top_contributors"] = contributions[:3]

    # Rank by severity then peak score (coach-facing ordering)
    sev_rank = {"critical": 3, "high": 2, "moderate": 1}
    danger_moments.sort(key=lambda m: (sev_rank.get(m["severity"], 0), m["peak_score"]), reverse=True)

    return danger_moments