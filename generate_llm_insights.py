import json
from pathlib import Path

from data_loader import list_matches, load_events, get_halftime_offset
from risk_engine import compute_risk_score
from danger_detector import detect_danger_moments
from pattern_analyzer import (
    build_all_matches_dangers_for_patterns,
    find_patterns,
    format_patterns_for_llm,
)
from explainer import explain_moment, explain_pattern

OUT_DIR = Path("outputs/llm_insights")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_int_seconds(td) -> int:
    """Convert a pandas/py Timedelta to int seconds safely."""
    try:
        return int(td.total_seconds())
    except Exception:
        return 0


def main():
    matches = list_matches()
    matches = matches[:3] # REMOVE THIS LATER PELEEASDFJAELIFJAEKJFLE

    # -------------------------
    # A) Per-match danger moments -> LLM explanations
    # -------------------------
    all_moment_outputs = []

    for match_name in matches:
        events_df = load_events(match_name)
        risk_df = compute_risk_score(events_df)

        dangers = detect_danger_moments(
            risk_df,
            events_df,
            debug=True,
            match_name=match_name
        )
        dangers = dangers[:3]   # REMOVE THIS LATER PELEEASDFJAELIFJAEKJFLE

        # Infer opponent name from match string; fallback to "Opponent"
        opponent = match_name.split(" - ")[-1].strip() if " - " in match_name else "Opponent"

        # ---- NEW: halftime correction values so explainer formats true in-game time ----
        halftime_offset = _safe_int_seconds(get_halftime_offset(events_df))

        # First raw timestamp of 2nd half (if present) — used by explainer for formatting.
        h2_start_sec = 0
        try:
            if "Half" in events_df.columns and "timestamp" in events_df.columns:
                h2 = events_df[events_df["Half"] == "2nd Half"]
                if not h2.empty:
                    h2_start_sec = _safe_int_seconds(h2["timestamp"].min())
        except Exception:
            h2_start_sec = 0

        for d in dangers:
            text = explain_moment(
                d,
                match_name,
                opponent,
                halftime_offset=halftime_offset,
                h2_start_sec=h2_start_sec,
            )

            all_moment_outputs.append(
                {
                    "match_name": match_name,
                    "opponent": opponent,
                    "halftime_offset_sec": halftime_offset,
                    "h2_start_sec": h2_start_sec,
                    "danger_moment": d,
                    "llm_response": text,
                }
            )

    (OUT_DIR / "danger_moment_explanations.json").write_text(
        json.dumps(all_moment_outputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # -------------------------
    # B) Cross-match patterns -> LLM explanations
    # -------------------------
    baseline = build_all_matches_dangers_for_patterns(matches, mode="all")
    goal_dangers = build_all_matches_dangers_for_patterns(matches, mode="goals")

    patterns = find_patterns(goal_dangers, baseline_dangers=baseline)
    patterns_for_llm = format_patterns_for_llm(patterns, top_n=10)

    pattern_outputs = []
    for p in patterns_for_llm:
        text = explain_pattern(p)
        pattern_outputs.append({"pattern": p, "llm_response": text})

    (OUT_DIR / "pattern_explanations.json").write_text(
        json.dumps(pattern_outputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved: {OUT_DIR / 'danger_moment_explanations.json'}")
    print(f"Saved: {OUT_DIR / 'pattern_explanations.json'}")


if __name__ == "__main__":
    main()