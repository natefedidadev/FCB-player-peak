import pandas as pd
import numpy as np
from danger_detector import detect_danger_moments, BARCA_TEAM_NAME

def make_risk_df():
    t = np.arange(0, 300)  # 5 minutes
    risk = np.zeros_like(t, dtype=float)

    # A normal peak around 100s
    risk[90:110] = np.linspace(20, 80, 20)
    risk[110:130] = np.linspace(80, 10, 20)

    # A goal-related buildup peaking around 200s
    risk[170:200] = np.linspace(10, 92, 30)
    risk[200:220] = np.linspace(92, 40, 20)

    # active_events: list of codes each second
    active = []
    for sec in t:
        codes = []
        if 90 <= sec <= 130:
            codes += ["BALL IN FINAL THIRD"]
        if 100 <= sec <= 115:
            codes += ["CREATING CHANCES"]
        if 170 <= sec <= 220:
            codes += ["DEFENSIVE TRANSITION", "BALL IN FINAL THIRD"]
        if 190 <= sec <= 210:
            codes += ["BALL IN THE BOX"]
        active.append(codes)

    return pd.DataFrame({"timestamp_sec": t, "risk_score": risk, "active_events": active})

def make_events_df():
    # Here we assume timestamp is a timedelta (because your code uses .dt.total_seconds()).
    # If your real data isn't timedelta, you'll want to adjust _get_goal_timestamps.
    return pd.DataFrame({
        "code": ["GOALS"],
        "Team": ["Some Opponent"],
        "timestamp": [pd.to_timedelta(220, unit="s")]
    })

if __name__ == "__main__":
    risk_df = make_risk_df()
    events_df = make_events_df()

    print("min risk:", risk_df["risk_score"].min())
    print("max risk:", risk_df["risk_score"].max())
    thr = np.percentile(risk_df["risk_score"].values, 60)
    print("threshold p60:", thr)
    print("num secs above thr:", (risk_df["risk_score"].values >= thr).sum(), "of", len(risk_df))


    dangers = detect_danger_moments(
        risk_df,
        events_df,
        peak_percentile=60,   # lower so we pick up both
        min_distance=30,
        goal_lookback=90,
        context_window=60
    )

    print("Detected danger moments:")
    for d in dangers:
        print(d)

    assert any(d["resulted_in_goal"] for d in dangers), "Expected at least one goal-anchored danger moment"
    assert any(d["peak_time"] in range(190, 205) for d in dangers), "Expected a peak around ~200s"
    print("✅ basic synthetic test passed")
