from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from data_loader import list_matches, load_events
from risk_engine import compute_risk_score, OPPONENT_WEIGHTS, BARCA_WEIGHTS
from danger_detector import detect_danger_moments


# ----------------------------
# Config
# ----------------------------

DEFAULT_FINGERPRINT_WINDOW_SEC = 60

# Stopwords for cross-match PATTERNING (not Phase 2 diagnostics)
DEFAULT_STOPWORDS = {
    "BUILD UP",
    "GOALS",
    "SET PIECES",
    "PLAYERS IN THE BOX",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
}

# Keep fingerprints short + interpretable
DEFAULT_TOP_K_CODES = 4

# Pattern grouping
DEFAULT_MIN_SUBSEQ_SIMILARITY = 0.85
DEFAULT_MIN_MATCH_FREQUENCY = 2

# NEW: avoid useless "BALL IN FINAL THIRD -> BALL IN THE BOX" patterns
MIN_PATTERN_LEN = 4

# NEW: require at least one "cause" code so patterns are actionable
CAUSE_CODES = {
    "ATTACKING TRANSITION",
    "DEFENSIVE TRANSITION",
    "PROGRESSION",
    "CREATING CHANCES",
}
REQUIRE_CAUSE_CODE = True

# Optional: if you ONLY want goal-related patterns (coach-facing version)
REPORT_GOAL_ONLY = False


# ----------------------------
# Helpers
# ----------------------------

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
    try:
        if isinstance(active_events, float) and np.isnan(active_events):
            return []
    except Exception:
        pass

    if isinstance(active_events, list):
        if not active_events:
            return []
        if isinstance(active_events[0], str):
            return [str(x) for x in active_events]
        if isinstance(active_events[0], dict):
            out: List[str] = []
            for e in active_events:
                if isinstance(e, dict) and e.get("code"):
                    out.append(str(e["code"]))
            return out
        return [str(x) for x in active_events]

    if isinstance(active_events, str):
        return [active_events]

    return []


def _dedupe_preserve_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _code_weight(code: str) -> float:
    """Heuristic importance weight used only for fingerprint compression."""
    return float(max(OPPONENT_WEIGHTS.get(code, 0), BARCA_WEIGHTS.get(code, 0)))


def _compress_sequence_keep_order(seq: List[str], top_k: int = DEFAULT_TOP_K_CODES) -> List[str]:
    """
    Keep only the top_k highest-weight codes that appear in the sequence,
    but preserve original order.
    """
    if not seq:
        return []

    unique = _dedupe_preserve_order(seq)
    scored = [(c, _code_weight(c)) for c in unique]
    scored.sort(key=lambda x: x[1], reverse=True)

    keep = set([c for c, _w in scored[:top_k]])
    return [c for c in unique if c in keep]


def _has_cause_code(seq: List[str], cause_codes: set[str] = CAUSE_CODES) -> bool:
    return any(c in cause_codes for c in (seq or []))


def build_fingerprint_sequence(
    risk_df: pd.DataFrame,
    peak_time_sec: int,
    window_sec: int = DEFAULT_FINGERPRINT_WINDOW_SEC,
    *,
    stopwords: Optional[set[str]] = None,
    top_k: int = DEFAULT_TOP_K_CODES,
) -> List[str]:
    """
    Fingerprint = codes that ENTER the active set over the last `window_sec`,
    filtered by stopwords, then compressed to top_k by weight.

    Note:
      - This is intentionally NOT ML. It’s a stable heuristic fingerprint.
      - We dedupe + keep order to preserve the tactical story.
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    if risk_df is None or len(risk_df) == 0:
        return []
    if "timestamp_sec" not in risk_df.columns or "active_events" not in risk_df.columns:
        return []

    t0 = max(int(peak_time_sec) - int(window_sec), int(risk_df["timestamp_sec"].min()))
    t1 = int(peak_time_sec)

    window = risk_df[(risk_df["timestamp_sec"] >= t0) & (risk_df["timestamp_sec"] <= t1)].copy()

    seq: List[str] = []
    prev_set: set[str] = set()

    for _, row in window.iterrows():
        codes = _normalize_active_events_to_codes(row.get("active_events"))
        cur_set = {c for c in codes if c and c not in stopwords}

        # add only codes that newly appear at this second
        entered = [c for c in codes if (c in cur_set and c not in prev_set)]
        seq.extend(entered)

        prev_set = cur_set

    seq = _dedupe_preserve_order(seq)
    seq = _compress_sequence_keep_order(seq, top_k=top_k)
    return seq


def _is_subsequence(shorter: List[str], longer: List[str]) -> bool:
    """True if `shorter` is a subsequence of `longer` (order preserved, not necessarily contiguous)."""
    if not shorter:
        return True
    it = iter(longer)
    return all(any(x == y for y in it) for x in shorter)


def subseq_similarity(a: List[str], b: List[str]) -> float:
    """
    A simple similarity for sequences based on subsequence overlap.
    Returns value in [0,1].
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    if len(a) <= len(b):
        overlap = len(a) if _is_subsequence(a, b) else 0
        denom = max(len(a), len(b))
        return overlap / denom
    else:
        overlap = len(b) if _is_subsequence(b, a) else 0
        denom = max(len(a), len(b))
        return overlap / denom


# ----------------------------
# Build danger moments for patterning
# ----------------------------

def build_all_matches_dangers_for_patterns(
    matches: List[str],
    *,
    mode: str = "all",  # "all" | "goals" | "critical"
    peak_percentile: float = 70,
    threshold_floor: float = 40.0,
    min_distance: int = 35,
    prominence: float = 10.0,
    goal_lookback: int = 90,
    merge_within_sec: int = 60,
) -> List[Dict[str, Any]]:
    """
    For each match:
      events_df -> risk_df -> danger moments -> (optional filter) -> fingerprint
    Returns a flat list of danger dicts, each augmented with match_name + fingerprint.
    """
    all_dangers: List[Dict[str, Any]] = []

    mode = (mode or "all").lower().strip()
    if mode not in {"all", "goals", "critical"}:
        raise ValueError("mode must be one of: 'all', 'goals', 'critical'")

    for match_name in matches:
        events_df = load_events(match_name)
        risk_df = compute_risk_score(events_df)

        dangers = detect_danger_moments(
            risk_df,
            events_df,
            peak_percentile=peak_percentile,
            threshold_floor=threshold_floor,
            min_distance=min_distance,
            prominence=prominence,
            goal_lookback=goal_lookback,
            merge_within_sec=merge_within_sec,
        )

        # ✅ FILTER HERE (before fingerprinting)
        if mode == "goals":
            dangers = [d for d in dangers if bool(d.get("resulted_in_goal", False))]
        elif mode == "critical":
            dangers = [d for d in dangers if str(d.get("severity", "")).lower() == "critical"]

        for d in dangers:
            peak_time = int(d["peak_time"])
            seq = build_fingerprint_sequence(
                risk_df,
                peak_time_sec=peak_time,
                window_sec=DEFAULT_FINGERPRINT_WINDOW_SEC,
                stopwords=DEFAULT_STOPWORDS,
                top_k=DEFAULT_TOP_K_CODES,
            )

            dd = dict(d)
            dd["match_name"] = match_name
            dd["fingerprint_seq"] = seq
            all_dangers.append(dd)

    return all_dangers


# ----------------------------
# Pattern grouping
# ----------------------------

@dataclass
class Pattern:
    sequence: List[str]
    matches: set[str]
    examples: List[Dict[str, Any]]
    goals_in_pattern: int


def find_patterns(
    all_dangers: List[Dict[str, Any]],
    *,
    baseline_dangers: Optional[List[Dict[str, Any]]] = None,
    min_subseq_similarity: float = DEFAULT_MIN_SUBSEQ_SIMILARITY,
    min_match_frequency: int = DEFAULT_MIN_MATCH_FREQUENCY,
    min_occurrences: int = 3,
    min_lift: float = 1.25,
) -> List[Dict[str, Any]]:
    """
    Group similar sequences and return summary patterns.

    IMPORTANT:
    - If you pass goal-only `all_dangers`, you MUST pass `baseline_dangers`
      (typically mode="all") to compute a meaningful baseline goal rate.
    """

    # ---------- helpers ----------
    def _passes_sequence_filters(seq: List[str]) -> bool:
        if not seq:
            return False
        if len(seq) < MIN_PATTERN_LEN:
            return False
        if REQUIRE_CAUSE_CODE and not _has_cause_code(seq, CAUSE_CODES):
            return False
        return True

    # ---------- choose baseline ----------
    baseline_src = baseline_dangers if baseline_dangers is not None else all_dangers

    baseline_occ = 0
    baseline_goals = 0
    for d in baseline_src:
        seq = d.get("fingerprint_seq", []) or []
        if not _passes_sequence_filters(seq):
            continue
        baseline_occ += 1
        if bool(d.get("resulted_in_goal", False)):
            baseline_goals += 1

    baseline_goal_rate = (baseline_goals / baseline_occ) if baseline_occ else 0.0

    # ---------- cluster goal (or selected) dangers ----------
    clusters: List[Pattern] = []
    total_matches = len(set(d.get("match_name", "UNKNOWN") for d in all_dangers))

    for d in all_dangers:
        seq: List[str] = d.get("fingerprint_seq", []) or []
        if not _passes_sequence_filters(seq):
            continue

        match_name = str(d.get("match_name", "UNKNOWN"))
        resulted_in_goal = bool(d.get("resulted_in_goal", False))

        placed = False
        for c in clusters:
            sim = subseq_similarity(seq, c.sequence)
            if sim >= min_subseq_similarity:
                c.matches.add(match_name)
                if len(c.examples) < 5:
                    c.examples.append(d)
                if resulted_in_goal:
                    c.goals_in_pattern += 1

                # keep representative short
                if len(seq) < len(c.sequence):
                    c.sequence = seq
                placed = True
                break

        if not placed:
            clusters.append(
                Pattern(
                    sequence=seq,
                    matches={match_name},
                    examples=[d],
                    goals_in_pattern=1 if resulted_in_goal else 0,
                )
            )

    # ---------- compute occurrences + lift using *baseline_src* ----------
    out: List[Dict[str, Any]] = []

    for c in clusters:
        match_count = len(c.matches)
        if match_count < min_match_frequency:
            continue

        occurrences = 0
        goals_in_cluster = 0

        for d in baseline_src:
            seq = d.get("fingerprint_seq", []) or []
            if not _passes_sequence_filters(seq):
                continue
            if subseq_similarity(seq, c.sequence) >= min_subseq_similarity:
                occurrences += 1
                if bool(d.get("resulted_in_goal", False)):
                    goals_in_cluster += 1

        if occurrences < min_occurrences:
            continue

        goal_rate = (goals_in_cluster / occurrences) if occurrences else 0.0
        lift = (goal_rate / baseline_goal_rate) if baseline_goal_rate > 0 else 0.0

        if lift < min_lift:
            continue

        out.append(
            {
                "sequence": c.sequence,
                "match_count": match_count,
                "frequency": f"{match_count}/{total_matches} matches",
                "occurrences": occurrences,
                "goals_in_pattern": int(c.goals_in_pattern),  # from the passed-in (often goal-only) list
                "goal_rate": round(goal_rate, 4),
                "baseline_goal_rate": round(baseline_goal_rate, 4),
                "lift": round(lift, 3),
                "example_matches": sorted(list(c.matches))[:5],
                "examples": [
                    {
                        "match_name": ex.get("match_name"),
                        "peak_time": ex.get("peak_time"),
                        "peak_score": ex.get("peak_score"),
                        "severity": ex.get("severity"),
                        "resulted_in_goal": ex.get("resulted_in_goal"),
                        "nexus_timestamp": ex.get("nexus_timestamp"),
                    }
                    for ex in c.examples
                ],
            }
        )

    out.sort(key=lambda p: (p["match_count"], p["lift"], p["occurrences"]), reverse=True)
    return out


# ----------------------------
# Quick test runner
# ----------------------------

def _pretty_seq(seq: List[str]) -> str:
    return " → ".join(seq)


def main():
    matches = list_matches()
    all_dangers = build_all_matches_dangers_for_patterns(matches, mode="goals")

    print(f"Total danger moments for patterning: {len(all_dangers)}")

    # patterns = find_patterns(
    #     all_dangers,
    #     min_subseq_similarity=DEFAULT_MIN_SUBSEQ_SIMILARITY,
    #     min_match_frequency=DEFAULT_MIN_MATCH_FREQUENCY,
    # )

    # print(f"Patterns found: {len(patterns)}\n")

    # for p in patterns[:12]:
    #     print(f"{p['frequency']} | goals_in_pattern={p['goals_in_pattern']} | seq: {_pretty_seq(p['sequence'])}")
    #     print(f"examples: {p['example_matches']}\n")


if __name__ == "__main__":
    main()
