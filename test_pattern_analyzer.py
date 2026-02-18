from data_loader import list_matches
from pattern_analyzer import build_all_matches_dangers_for_patterns, find_patterns

def main():
    matches = list_matches()

    # baseline = ALL danger moments
    baseline = build_all_matches_dangers_for_patterns(matches, mode="all")
    print("Baseline danger moments:", len(baseline))

    # cluster = GOAL moments only
    goal_dangers = build_all_matches_dangers_for_patterns(matches, mode="goals")
    print("Goal danger moments:", len(goal_dangers))

    patterns = find_patterns(
        goal_dangers,
        baseline_dangers=baseline,
        min_subseq_similarity=0.85,
        min_match_frequency=2,
        min_occurrences=3,
        min_lift=1.25,
    )

    print("Patterns found:", len(patterns))
    for p in patterns[:10]:
        print(
            p["frequency"],
            "| occ=", p["occurrences"],
            "| goal_rate=", p["goal_rate"],
            "| baseline=", p["baseline_goal_rate"],
            "| lift=", p["lift"],
            "| seq:", " → ".join(p["sequence"]),
            "| goals=", p["goals_in_pattern"]
        )
        print("examples:", p["example_matches"])
        print()

if __name__ == "__main__":
    main()