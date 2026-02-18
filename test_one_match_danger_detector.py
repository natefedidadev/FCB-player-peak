import matplotlib.pyplot as plt

from data_loader import list_matches, load_events
from risk_engine import compute_risk_score
from danger_detector import detect_danger_moments


def main(match_idx: int = 0):
    matches = list_matches()
    print(f"Found {len(matches)} matches")
    print("Testing match:", matches[match_idx])

    events_df = load_events(match_idx)

    # Quick dtype check (useful for debugging)
    print("\nEvents dtypes:")
    print(events_df[["timestamp", "end_timestamp", "code", "Team"]].dtypes)

    risk_df = compute_risk_score(events_df)
    dangers = detect_danger_moments(
        risk_df,
        events_df,
        peak_percentile=70,
        threshold_floor=40.0,
        min_distance=35,
        prominence=10.0,
        goal_lookback=90,
        context_window=60,
        merge_within_sec=60
    )

    print(f"\nDetected {len(dangers)} danger moments. Top 10:\n")
    for d in dangers[:10]:
        print(d)

    # Plot risk + danger markers
    plt.figure()
    plt.plot(risk_df["timestamp_sec"], risk_df["risk_score"])
    for d in dangers:
        plt.scatter(d["peak_time"], d["peak_score"])
    plt.title(f"Risk score + danger peaks: {matches[match_idx]}")
    plt.xlabel("Seconds")
    plt.ylabel("Risk (0-100)")
    plt.show()


if __name__ == "__main__":
    main(match_idx=0)
