from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import os
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


# -------------------------
# danger_detector.py (REPLACE your _get_goal_timestamps_sec with this)
# -------------------------

def _to_seconds_series(s: pd.Series) -> pd.Series:
    """Convert a Series of timestamps (timedelta/numeric/datetime/parseable) to float seconds."""
    if pd.api.types.is_timedelta64_dtype(s):
        return s.dt.total_seconds()

    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    if pd.api.types.is_datetime64_any_dtype(s):
        t0 = s.min()
        return (s - t0).dt.total_seconds()

    # fallback parse
    parsed = pd.to_datetime(s, errors="coerce")
    if parsed.notna().any():
        t0 = parsed.min()
        return (parsed - t0).dt.total_seconds()

    return pd.Series([float("nan")] * len(s), index=s.index)


def _get_goal_timestamps_sec(
    events_df: pd.DataFrame,
    barca_team_name: str = BARCA_TEAM_NAME,
    *,
    debug: bool = False,
    # How much to "trim" off the end of long GOALS windows (celebration).
    # If a GOALS event is 25s long, end_timestamp is often celebration end.
    celebration_trim_cap_sec: int = 10,
    # Treat GOALS windows longer than this as "definitely includes celebration".
    long_window_sec: int = 12,
    # Dedupe goal moments within N seconds
    dedupe_within_sec: int = 8,
) -> List[int]:
    """
    Return goal timestamps (seconds) for goals AGAINST Barcelona.

    Fixes common issues:
      - GOALS windows often include celebration; end_timestamp can be late.
      - Team labeling can be inconsistent; we log and keep N/A for debugging.
    """
    if events_df is None or len(events_df) == 0 or "code" not in events_df.columns:
        return []

    goals = events_df[events_df["code"] == "GOALS"].copy()
    if goals.empty:
        return []

    # Filter to goals "against" Barca (opponent goals) when Team is available.
    # NOTE: If Team is N/A or missing, we keep them for debug instead of dropping silently.
    if "Team" in goals.columns:
        # Keep rows where Team != Barca OR Team is null/NA-like
        team = goals["Team"].astype(str)
        is_unknown = goals["Team"].isna() | team.str.strip().isin(["", "N/A", "nan", "None"])
        goals_against = goals[(team != str(barca_team_name)) | is_unknown].copy()
        if debug:
            total = len(goals)
            kept = len(goals_against)
            dropped = total - kept
            print(f"[GOALS] total={total} kept(as-against-or-unknown)={kept} dropped(team==Barca)={dropped}")
        goals = goals_against

    if goals.empty:
        return []

    # Prefer start + end if present
    has_start = "timestamp" in goals.columns
    has_end = "end_timestamp" in goals.columns

    if not has_start and not has_end:
        return []

    start_sec = _to_seconds_series(goals["timestamp"]) if has_start else pd.Series([float("nan")] * len(goals), index=goals.index)
    end_sec = _to_seconds_series(goals["end_timestamp"]) if has_end else start_sec.copy()

    # If end is missing/NaN, fall back to start
    end_sec = end_sec.fillna(start_sec)

    dur = (end_sec - start_sec).fillna(0.0)

    goal_secs: List[int] = []

    for idx in goals.index:
        s0 = float(start_sec.loc[idx]) if pd.notna(start_sec.loc[idx]) else float("nan")
        s1 = float(end_sec.loc[idx]) if pd.notna(end_sec.loc[idx]) else float("nan")
        d = float(dur.loc[idx]) if pd.notna(dur.loc[idx]) else 0.0

        if pd.isna(s1):
            continue

        # Heuristic: actual goal tends to be near the end of the window,
        # but if the window is long, end_timestamp is usually *after* celebration.
        # So trim a few seconds off the end (capped).
        if d >= long_window_sec:
            trim = min(celebration_trim_cap_sec, max(3, int(round(d * 0.25))))
            est = s1 - trim
        else:
            # short window -> end is usually close enough
            est = s1

        goal_secs.append(int(round(est)))

        if debug:
            team_val = goals.loc[idx, "Team"] if "Team" in goals.columns else "?"
            print(f"[GOALS] team={team_val} start={s0:.1f} end={s1:.1f} dur={d:.1f} -> est_goal={int(round(est))}")

    # Sort + dedupe close ones
    goal_secs.sort()
    deduped: List[int] = []
    for t in goal_secs:
        if not deduped or abs(t - deduped[-1]) > dedupe_within_sec:
            deduped.append(t)

    if debug:
        print(f"[GOALS] final goal_times={deduped}")

    return deduped


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
    min_distance: int = 35,
    prominence: float = 10.0,
    goal_lookback: int = 90,
    context_window: int = 60,
    merge_within_sec: Optional[int] = 60,

    # ---- NEW (safe optional debug + goal-anchoring knobs) ----
    match_name: Optional[str] = None,
    debug: bool = False,
    debug_match_name: Optional[str] = None,
    goal_forward_tolerance: int = 10,          # allow slicing slightly AFTER goal time
    goal_tag_tolerance: int = 12,              # tag an existing peak if within ±N seconds of goal time
    prefer_latest_max_anchor: bool = True      # if multiple max values, pick the latest one
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

    # -------- DEBUG HEADER --------
    do_dbg = bool(debug) or (debug_match_name is not None and match_name == debug_match_name)
    if do_dbg:
        print("\n" + "=" * 80)
        print(f"[detect_danger_moments] {match_name or '<unknown match>'}")
        print(f"timeline: t=[{int(times[0])}..{int(times[-1])}] n={len(times)}")
        print(f"threshold: raw_percentile({peak_percentile})={raw_percentile_thr:.2f}  used={threshold_used:.2f}")
        print(f"peaks params: distance={min_distance} prominence={prominence}")
        print(f"goal anchoring: lookback={goal_lookback}s forward_tol={goal_forward_tolerance}s tag_tol=±{goal_tag_tolerance}s")
        print("=" * 80)

    # -------- A) Peaks-based moments --------
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

            "threshold_used": float(threshold_used),
            "raw_percentile_threshold": float(raw_percentile_thr),
            "num_active_codes": int(len(active_codes)),
            "top_contributors": _compute_top_contributors(active_codes, top_k=3),
        })

    if do_dbg:
        print(f"found peaks: {len(peaks)}")
        if len(peaks) > 0:
            first = [(int(times[i]), float(risk_scores[i])) for i in peaks[:10]]
            print(f"first peaks (t,score): {[(t, round(s,2)) for t,s in first]}")

    # -------- B) Goal anchoring --------
    goal_times = _get_goal_timestamps_sec(events_df, barca_team_name=BARCA_TEAM_NAME)

    if do_dbg:
        print(f"goal_times (raw sec): {goal_times}")

    for gt in goal_times:
        gt = int(gt)

        # include a small forward tolerance to protect against rounding / window-end issues
        start_t = max(gt - int(goal_lookback), int(times[0]))
        end_t = min(gt + int(goal_forward_tolerance), int(times[-1]))

        start_idx = int(np.searchsorted(times, start_t, side="left"))
        end_idx = int(np.searchsorted(times, end_t, side="right")) - 1
        if start_idx > end_idx:
            continue

        seg = risk_scores[start_idx:end_idx + 1]
        if len(seg) == 0:
            continue

        # Pick anchor = argmax(seg). If ties, choose the LATEST max (closest to the goal moment)
        max_val = float(np.max(seg))
        max_idxs = np.where(seg == max_val)[0]
        if prefer_latest_max_anchor:
            anchor_local = int(max_idxs[-1])  # latest
        else:
            anchor_local = int(max_idxs[0])   # earliest
        anchor_idx = start_idx + anchor_local

        anchor_peak_time = int(times[anchor_idx])
        anchor_peak_score = float(risk_scores[anchor_idx])

        # Prefer tagging an existing moment close to the goal time OR whose window reaches the goal
        best_idx = None
        best_key = None

        for i, m in enumerate(danger_moments):
            pt = int(m["peak_time"])
            ws = int(m.get("window_start", pt))
            we = int(m.get("window_end", pt))

            # Candidate if:
            # 1) peak is close to gt, OR
            # 2) the moment window overlaps the goal time (with forward tolerance)
            close_to_goal = abs(pt - gt) <= goal_tag_tolerance
            window_hits_goal = (ws <= gt <= (we + goal_forward_tolerance))

            if not (close_to_goal or window_hits_goal):
                continue

            # scoring: prioritize window overlap, then closeness to gt, then later time
            # (later matters because GOALS windows often start well before the finish)
            overlap_bonus = 0 if not window_hits_goal else -1000
            close = abs(pt - gt)
            key = (overlap_bonus, close, -pt)

            if best_key is None or key < best_key:
                best_key = key
                best_idx = i

        if best_idx is not None:
            danger_moments[best_idx]["resulted_in_goal"] = True
            danger_moments[best_idx]["severity"] = "critical"
            if do_dbg:
                m = danger_moments[best_idx]
                print(f"[goal anchor] gt={gt} -> tagged peak={m['peak_time']} (Δ={abs(int(m['peak_time'])-gt)}s, score={m['peak_score']:.2f})")
            continue

        # Otherwise create a new anchored moment at the anchor_peak_time
        window_start, window_end = _find_window_bounds(times, risk_scores, anchor_idx, threshold_used)

        active_codes: List[str] = []
        if "active_events" in risk_df.columns:
            active_codes = _normalize_active_events_to_codes(risk_df.loc[anchor_idx, "active_events"])

        danger_moments.append({
            "peak_time": anchor_peak_time,
            "window_start": window_start,
            "window_end": window_end,
            "peak_score": anchor_peak_score,
            "severity": "critical",
            "resulted_in_goal": True,
            "active_codes": active_codes,
            "nexus_timestamp": _format_nexus_timestamp(anchor_peak_time),

            "threshold_used": float(threshold_used),
            "raw_percentile_threshold": float(raw_percentile_thr),
            "num_active_codes": int(len(active_codes)),
            "top_contributors": _compute_top_contributors(active_codes, top_k=3),
        })

        if do_dbg:
            print(f"[goal anchor fallback] gt={gt} -> created moment at peak={anchor_peak_time} score={anchor_peak_score:.2f}")

    # -------- C) Merge close moments to collapse pressure spells --------
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

    if do_dbg:
        print(f"final danger_moments: {len(danger_moments)}")
        for m in danger_moments[:10]:
            print(
                f"  - peak={m['peak_time']} score={m['peak_score']:.1f} "
                f"sev={m['severity']} goal={m['resulted_in_goal']} "
                f"window={m['window_start']}..{m['window_end']}"
            )

    return danger_moments