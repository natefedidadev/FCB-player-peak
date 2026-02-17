import pandas as pd
import numpy as np

BARCA_TEAM_NAME = "FC Barcelona"
DEFAULT_SMOOTHING_WINDOW = 15

OPPONENT_WEIGHTS = {
    'GOALS': 10,
    'BALL IN THE BOX': 8,
    'CREATING CHANCES': 7,
    'SET PIECES': 6,
    'BALL IN FINAL THIRD': 5,
    'ATTACKING TRANSITION': 4,
    'PROGRESSION': 3
}

BARCA_WEIGHTS = {
    'DEFENDING IN DEFENSIVE THIRD': 4,
    'DEFENSIVE TRANSITION': 3,
    'DEFENDING IN MIDDLE THIRD': 2,
    'DEFENDING IN ATTACKING THIRD': 1
}

# Passing in the raw events dataframe, and we spit out a tuple containing:
# 1. A time grid array
# 2. A copy of the events dataframe with two new columns added

def _build_time_grid(events_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    
    events_cleaned = events_df.dropna(subset=['timestamp', 'end_timestamp']).copy()

    # make a new column in events_cleaned for the start seconds of an event
    # it's format then turns from '0 days 00:00:42.920000' to an integer seconds value we can actually use
    
    events_cleaned['start_sec'] = events_cleaned['timestamp'].dt.total_seconds().astype(int)
    
    # then we do the same for the end timestamp. convert the timestamp to seconds as an int, round value to ceil    
    events_cleaned['end_sec'] = np.ceil(events_cleaned['end_timestamp'].dt.total_seconds()).astype(int)
    
    # lastly we need the time grid array, which will list out second 0 to the last second of the match
    time_grid = np.arange(0, events_cleaned['end_sec'].max() + 1)
    
    return time_grid, events_cleaned


# For every second of the match, we need to figure out which events are active and them sum their weights
# compute_raw_scores returns -> raw_scores (np array of risk values, one per second)
#                               active_events (list of lists, where each inner list has event code strings happening that second)

def _compute_raw_scores(time_grid: np.ndarray, events_cleaned: pd.DataFrame,
    opponent_weights: dict[str, int] = OPPONENT_WEIGHTS,
    barca_weights: dict[str, int] = BARCA_WEIGHTS) -> tuple[np.ndarray, list[list[str]]]:
    
    # For each row/event, if the event happened for barca, return the risk value in the barca weights dictionary for that event
    # else if for opponent, get the opponent's score
    
    def get_weight(row):
        if row['Team'] == BARCA_TEAM_NAME:
            return barca_weights.get(row['code'], 0)
        else:
            return opponent_weights.get(row['code'], 0)

    events_cleaned['weight'] = events_cleaned.apply(get_weight, axis=1)

    raw_scores = np.zeros(len(time_grid)) # np array of zeros. we'll add weights into it
    active_events = [[] for i in range(len(time_grid))] # list of empty lists, we'll append event code strings into each one
    
    # now we loop over events and paint the weights into them
    
    for _, row in events_cleaned.iterrows():
        start = row['start_sec']
        end = row['end_sec']
        raw_scores[start : end] += row['weight']
        for sec in range(start, end):
            active_events[sec].append(row['code'])
    
    return raw_scores, active_events

# with this function we smoothen the scoring, to deal with scores immediately dropping 
# because an event stops at a certain second
def _smooth_scores(raw_scores: np.ndarray, window: int = DEFAULT_SMOOTHING_WINDOW) -> np.ndarray:
    return pd.Series(raw_scores).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

# now we normalize the scores to have a consistent risk value
# a boring game could range from 0 - 30 and a chaotic game could range from 0 - 80
# we need to scale them to 0 - 100 so scores are comparable across every game

def _normalize_scores(smoothed_scores: np.ndarray) -> np.ndarray:
    # min - max normalization
    array_max = np.max(smoothed_scores)
    array_min = np.min(smoothed_scores)
    
    if array_max == array_min:
        return np.zeros_like(smoothed_scores)
    else:
        normalized_scores = (smoothed_scores - array_min) / (array_max - array_min) * 100
    
    return normalized_scores

def compute_risk_score(
    events_df: pd.DataFrame,
    opponent_weights: dict[str, int] = OPPONENT_WEIGHTS,
    barca_weights: dict[str, int] = BARCA_WEIGHTS,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    ) -> pd.DataFrame:
    
    time_grid, events_cleaned = _build_time_grid(events_df)
    raw_scores, active_events = _compute_raw_scores(time_grid, events_cleaned, opponent_weights, barca_weights)
    smoothed = _smooth_scores(raw_scores, smoothing_window)
    normalized = _normalize_scores(smoothed)
    
    return pd.DataFrame({'timestamp_sec': time_grid, 'risk_score': np.round(normalized, 2), 'active_events': active_events})
    
