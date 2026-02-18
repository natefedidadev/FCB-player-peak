import numpy as np
import pandas as pd

from tune_danger_detector_all_matches import run_config  # uses the function from above


def score_run(df: pd.DataFrame) -> float:
    # last row is ALL_MATCHES
    all_row = df.iloc[-1]

    mean_count = float(all_row["count"])
    mean_window = float(all_row["median_window_len"])
    goals_against = int(all_row["goals_against"])
    goal_anchored = int(all_row["goal_anchored_moments"])

    # penalty if too many or too few peaks
    count_penalty = abs(mean_count - 16.0) * 2.0

    # window target: median ~20–60s is nice (penalize outside)
    window_penalty = 0.0
    if mean_window < 15:
        window_penalty += (15 - mean_window) * 1.5
    if mean_window > 90:
        window_penalty += (mean_window - 90) * 0.8

    # goal anchoring coverage: if goals exist, we want at least one anchored moment per goal
    goal_penalty = 0.0
    if goals_against > 0:
        coverage = goal_anchored / max(goals_against, 1)
        if coverage < 1.0:
            goal_penalty += (1.0 - coverage) * 50.0

    return count_penalty + window_penalty + goal_penalty


def main():
    configs = []
    for threshold_floor in [30.0, 35.0, 40.0]:
        for prominence in [8.0, 10.0, 12.0]:
            for min_distance in [30, 35, 40]:
                configs.append({
                    "peak_percentile": 70,
                    "threshold_floor": threshold_floor,
                    "prominence": prominence,
                    "min_distance": min_distance
                })

    results = []
    for cfg in configs:
        df = run_config(cfg)
        run_score = score_run(df)
        all_row = df.iloc[-1].to_dict()
        results.append({
            **cfg,
            "score": run_score,
            "mean_count": all_row["count"],
            "mean_median_window": all_row["median_window_len"],
            "goals_against": all_row["goals_against"],
            "goal_anchored_moments": all_row["goal_anchored_moments"],
        })

    out = pd.DataFrame(results).sort_values("score").head(10)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
