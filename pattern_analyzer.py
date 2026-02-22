from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from data_loader import load_events
from danger_detector import detect_danger_moments
from risk_engine import BARCA_EVENT_WEIGHTS, OPPONENT_EVENT_WEIGHTS, compute_risk_score

# Backwards-friendly aliases (older code referenced these names)
OPPONENT_WEIGHTS = OPPONENT_EVENT_WEIGHTS
BARCA_WEIGHTS = BARCA_EVENT_WEIGHTS


@dataclass
class Pattern:
    code_combo: tuple[str, ...]
    # number of matches in the *target* set (e.g., goal_dangers) where this combo appears
    count: int
    matches: list[str]

    # baseline stats (optional)
    baseline_count: int = 0  # matches in baseline set where combo appears
    prevalence_goals: float | None = None
    prevalence_all: float | None = None
    lift: float | None = None
    confidence: int | None = None



def build_all_matches_dangers_for_patterns(matches: list[str], mode: str = "all") -> list[dict[str, Any]]:
    """
    Builds a flat list of danger moments across matches.

    mode:
      - "all": include all detected danger moments
      - "goals": only those with outcome == goal (if present), else falls back to all
    """
    all_dangers: list[dict[str, Any]] = []
    for match_name in matches:
        events_df = load_events(match_name)
        risk_df = compute_risk_score(events_df)

        dangers = detect_danger_moments(risk_df, events_df, match_name=match_name, debug=False)

        # If your upstream adds outcomes (goals/shot), filter here.
        if mode == "goals":
            gd = [d for d in dangers if str(d.get("outcome", "")).lower() in ("goal", "scored", "conceded_goal")]
            dangers = gd if gd else dangers

        for d in dangers:
            d2 = dict(d)
            d2["match_name"] = match_name
            all_dangers.append(d2)

    return all_dangers


def _danger_signature(d: dict[str, Any]) -> tuple[str, ...]:
    """
    Signature used for cross-match pattern mining.
    Treat pattern as an unordered combo of the key active event codes.
    """
    codes = d.get("active_event_codes") or []
    codes = [str(c).strip() for c in codes if str(c).strip()]

    # de-dup
    seen = set()
    uniq = []
    for c in codes:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    return tuple(sorted(uniq))


def find_patterns(dangers: list[dict[str, Any]], *, baseline_dangers: Optional[list[dict[str, Any]]] = None) -> list[Pattern]:
    """
    Counts repeated combos of active event codes across matches.

    If baseline_dangers is provided, compute a simple lift/confidence measure comparing
    prevalence in `dangers` vs prevalence in `baseline_dangers`.

    Notes:
    - We measure prevalence as "in how many matches does the combo appear at least once?"
      (not raw window frequency) to avoid overweighting matches with many similar windows.
    """
    if not dangers:
        return []

    # Target set (e.g., goal dangers)
    combo_to_matches: dict[tuple[str, ...], set[str]] = {}
    target_matches: set[str] = set()
    for d in dangers:
        mn = str(d.get("match_name", ""))
        if mn:
            target_matches.add(mn)
        combo = _danger_signature(d)
        if not combo:
            continue
        combo_to_matches.setdefault(combo, set()).add(mn)

    total_target_matches = max(1, len(target_matches))

    # Baseline set (e.g., all dangers)
    baseline_combo_to_matches: dict[tuple[str, ...], set[str]] = {}
    baseline_matches: set[str] = set()
    if baseline_dangers:
        for d in baseline_dangers:
            mn = str(d.get("match_name", ""))
            if mn:
                baseline_matches.add(mn)
            combo = _danger_signature(d)
            if not combo:
                continue
            baseline_combo_to_matches.setdefault(combo, set()).add(mn)

    total_baseline_matches = max(1, len(baseline_matches)) if baseline_matches else None

    patterns: list[Pattern] = []
    for combo, ms in combo_to_matches.items():
        p = Pattern(code_combo=combo, count=len([x for x in ms if x]), matches=sorted([x for x in ms if x]))

        if baseline_combo_to_matches and total_baseline_matches:
            b_ms = baseline_combo_to_matches.get(combo, set())
            p.baseline_count = len([x for x in b_ms if x])

            p.prevalence_goals = p.count / float(total_target_matches)
            p.prevalence_all = p.baseline_count / float(total_baseline_matches)

            # lift: how much more common in target than baseline
            if p.prevalence_all > 0:
                p.lift = p.prevalence_goals / p.prevalence_all
            else:
                p.lift = None

            # heuristic confidence 0-100 based on prevalence + lift (capped)
            lift_term = 0.0 if p.lift is None else min(3.0, float(p.lift)) / 3.0  # 0..1
            prev_term = min(1.0, p.count / float(total_baseline_matches))  # 0..1
            p.confidence = int(round(min(100.0, max(0.0, 60.0 * prev_term + 40.0 * lift_term))))
        else:
            p.confidence = None

        patterns.append(p)

    patterns.sort(key=lambda p: p.count, reverse=True)
    return patterns


def format_patterns_for_llm(patterns: list[Pattern], *, top_n: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in patterns[:top_n]:
        out.append(
            {
                "pattern": " + ".join(p.code_combo),
                "event_codes": list(p.code_combo),
                "match_count": int(p.count),
                "matches": p.matches,
                "baseline_match_count": int(getattr(p, "baseline_count", 0) or 0),
                "prevalence_goals": getattr(p, "prevalence_goals", None),
                "prevalence_all": getattr(p, "prevalence_all", None),
                "lift": getattr(p, "lift", None),
                "confidence": getattr(p, "confidence", None),
            }
        )
    return out