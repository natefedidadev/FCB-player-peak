from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import os
import glob as glob_module
import pandas as pd

from data_loader import list_matches, load_events, get_halftime_offset
from risk_engine import compute_risk_score
from danger_detector import detect_danger_moments
from explainer import explain_moment, explain_window
from tracking_features import summarize_window, load_team_map
from pathlib import Path

app = FastAPI(title="FCB Defensive Risk Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

# In-memory cache: avoid recomputing risk scores on every request
_cache = {}

# Seconds of pre-kickoff broadcast content per match video (user-measured, 1st half)
VIDEO_PRE_MATCH_OFFSETS = {
    0: 105,   # AC Milan - Barça (0-1)
    1: 280,   # Arsenal - Barça (5-3)
    2: 305,   # Barça - AC Milan (2-2)
    3: 682,   # Barça - AS Mònaco (0-3)
    4: 634,   # Barça - Como 1907 (5-0)
    5: 284,   # Barça - Manchester City (2-2)
    6: 123,   # Barça - Reial Madrid (3-0)
    7: 459,   # Daegu FC - Barça (0-5)
    8: 927,   # FC Seül - Barça (3-7)
    9: 283,   # Reial Madrid - Barça (1-2)
    10: 590,  # Vissel Kobe - Barça (1-3)
}

# Additional seconds to add to 2nd half seeks on top of VIDEO_PRE_MATCH_OFFSETS
# = display_window_start - video_clip_time (user-measured per match)
VIDEO_H2_EXTRA_OFFSETS = {
    0:   -5,  # AC Milan - Barça (0-1)       73:43 -> 73:48
    1:   96,  # Arsenal - Barça (5-3)         54:45 -> 53:09
    2:  507,  # Barça - AC Milan (2-2)        64:18 -> 55:51
    3:  357,  # Barça - AS Mònaco (0-3)       58:52 -> 52:55
    4:  349,  # Barça - Como 1907 (5-0)       62:28 -> 56:39
    5:  283,  # Barça - Manchester City (2-2) 52:37 -> 47:54
    6:  373,  # Barça - Reial Madrid (3-0)    73:22 -> 67:09
    7:  300,  # Daegu FC - Barça (0-5)        49:09 -> 44:09
    8:  403,  # FC Seül - Barça (3-7)         83:56 -> 77:13
    9:  832,  # Reial Madrid - Barça (1-2)    81:17 -> 67:25
    10: 178,  # Vissel Kobe - Barça (1-3)     91:01 -> 88:03
}

def _get_match_data(index: int):
    if index not in _cache:
        matches = list_matches()
        if index < 0 or index >= len(matches):
            raise HTTPException(status_code=404, detail=f"Match index {index} not found")
        events_df = load_events(matches[index])
        risk_df = compute_risk_score(events_df)
        _cache[index] = (events_df, risk_df)
    return _cache[index]

def _get_halftime_info(events_df):
    offset = get_halftime_offset(events_df)
    offset_sec = int(offset.total_seconds())
    h2 = events_df[events_df["Half"] == "2nd Half"]
    h2_start_sec = int(h2["timestamp"].dt.total_seconds().min()) if not h2.empty else 99999
    return offset_sec, h2_start_sec

def _apply_offset(raw_sec, offset_sec, h2_start_sec):
    if raw_sec >= h2_start_sec:
        return raw_sec - offset_sec
    return raw_sec

def _extract_opponent(events_df):
    teams = events_df["Team"].dropna().unique()
    for t in teams:
        if t != "FC Barcelona" and t != "N/A":
            return t
    return "Opponent"

def _load_tracking_data(match_name: str):
    """
    Load tracking data for a match if available.
    Returns (player_df, ball_df, team_map) or (None, None, {}) if unavailable.
    """
    parsed_dir = Path("parsed") / match_name
    if not parsed_dir.exists():
        return None, None, {}

    player_csv = parsed_dir / "player_positions.csv"
    ball_csv = parsed_dir / "ball_positions.csv"

    if not player_csv.exists():
        return None, None, {}

    try:
        player_df = pd.read_csv(player_csv)
        ball_df = pd.read_csv(ball_csv) if ball_csv.exists() else None
        team_map = load_team_map(parsed_dir)
        return player_df, ball_df, team_map
    except Exception as e:
        print(f"Warning: Failed to load tracking data for {match_name}: {e}")
        return None, None, {}

def _get_defending_attacking_ids(team_map: dict, opponent: str):
    """
    Determine team IDs for defending (Barça) and attacking (opponent) teams.
    Returns (defending_team_id, attacking_team_id).

    team_map structure: {
        'barca_team_id': str,
        'opponent_team_id': str,
        'team_id_to_name': dict,
        'barca_name': str,
        'opponent_name': str
    }
    """
    # team_map is already structured with the IDs we need
    barca_id = team_map.get('barca_team_id')
    opponent_id = team_map.get('opponent_team_id')

    return str(barca_id) if barca_id else None, str(opponent_id) if opponent_id else None


@app.get("/api/matches")
def get_matches():
    matches = list_matches()
    return [{"index": i, "name": m} for i, m in enumerate(matches)]


@app.get("/api/matches/{index}/risk")
def get_risk(index: int):
    matches = list_matches()
    if index < 0 or index >= len(matches):
        raise HTTPException(404, "Match not found")

    events_df, risk_df = _get_match_data(index)
    offset_sec, h2_start_sec = _get_halftime_info(events_df)

    # Downsample to every 3rd second for chart performance
    timeline = []
    for i in range(0, len(risk_df), 3):
        row = risk_df.iloc[i]
        raw_sec = int(row["time_s"])
        display_sec = _apply_offset(raw_sec, offset_sec, h2_start_sec)
        timeline.append({
            "time_sec": raw_sec,
            "display_sec": display_sec,
            "match_minute": round(display_sec / 60.0, 2),
            "risk_score": float(row["risk_score"]),
        })

    return {
        "match_name": matches[index],
        "halftime_offset_sec": offset_sec,
        "h2_start_sec": h2_start_sec,
        "timeline": timeline,
    }


@app.get("/api/matches/{index}/dangers")
def get_dangers(index: int):
    matches = list_matches()
    if index < 0 or index >= len(matches):
        raise HTTPException(404, "Match not found")

    events_df, risk_df = _get_match_data(index)
    dangers = detect_danger_moments(risk_df, events_df)
    offset_sec, h2_start_sec = _get_halftime_info(events_df)
    match_name = matches[index]
    opponent = _extract_opponent(events_df)

    # Load tracking data if available
    player_df, ball_df, team_map = _load_tracking_data(match_name)
    defending_id, attacking_id = _get_defending_attacking_ids(team_map, opponent)

    results = []
    for d in dangers:
        peak_s = d["peak"]["time_s"]
        win_start = d["danger_window"]["start_s"]
        win_end   = d["danger_window"]["end_s"]

        peak_display         = _apply_offset(peak_s,    offset_sec, h2_start_sec)
        window_start_display = _apply_offset(win_start, offset_sec, h2_start_sec)
        window_end_display   = _apply_offset(win_end,   offset_sec, h2_start_sec)

        # Generate tracking summary if data available
        tracking_summary = None
        if player_df is not None and not player_df.empty:
            try:
                tracking_summary = summarize_window(
                    player_df,
                    ball_df,
                    win_start,
                    win_end,
                    defending_team_id=defending_id,
                    attacking_team_id=attacking_id,
                    preferred_time_s=peak_s,
                )
            except Exception as e:
                print(f"Warning: Failed to generate tracking summary: {e}")
                tracking_summary = None

        explanation = explain_moment(d, match_name, opponent, tracking_summary=tracking_summary)

        results.append({
            "peak_time": peak_s,
            "window_start": win_start,
            "window_end": win_end,
            "display_peak_sec": peak_display,
            "display_peak_minute": round(peak_display / 60.0, 2),
            "display_window_start": window_start_display,
            "display_window_end": window_end_display,
            "peak_score": d["peak"]["score"],
            "severity": d["severity"],
            "active_codes": d.get("active_event_codes", []),
            "resulted_in_goal": d.get("resulted_in_goal", False),
            "top_contributors": [{"code": c, "weight": w} for c, w in d.get("top_contributors", [])],
            "nexus_timestamp": d.get("nexus_timestamp", ""),
            "explanation": explanation,
        })

    video_offset = VIDEO_PRE_MATCH_OFFSETS.get(index, 0)
    video_h2_extra = VIDEO_H2_EXTRA_OFFSETS.get(index, 0)
    return {
        "match_name": match_name,
        "opponent": opponent,
        "dangers": results,
        "video_pre_match_offset": video_offset,
        "video_h2_extra_offset": video_h2_extra,
    }


@app.get("/api/matches/{index}/video")
async def get_video(index: int):
    matches = list_matches()
    if index < 0 or index >= len(matches):
        raise HTTPException(404, "Match not found")

    match_name = matches[index]
    match_dir = os.path.join("matches", match_name)
    mp4_files = glob_module.glob(os.path.join(match_dir, "*.mp4"))
    if not mp4_files:
        raise HTTPException(404, "No video found for this match")

    return FileResponse(mp4_files[0], media_type="video/mp4")


class WindowRequest(BaseModel):
    start_sec: int
    end_sec: int


@app.post("/api/matches/{index}/analyze-window")
def analyze_window(index: int, req: WindowRequest):
    matches = list_matches()
    if index < 0 or index >= len(matches):
        raise HTTPException(404, "Match not found")

    events_df, risk_df = _get_match_data(index)
    offset_sec, h2_start_sec = _get_halftime_info(events_df)
    match_name = matches[index]
    opponent = _extract_opponent(events_df)

    # Filter events overlapping with the requested window
    events_cleaned = events_df.dropna(subset=["timestamp", "end_timestamp"]).copy()
    events_cleaned["start_sec"] = events_cleaned["timestamp"].dt.total_seconds().astype(int)
    events_cleaned["end_sec"] = np.ceil(events_cleaned["end_timestamp"].dt.total_seconds()).astype(int)

    in_window = events_cleaned[
        (events_cleaned["end_sec"] >= req.start_sec) &
        (events_cleaned["start_sec"] <= req.end_sec)
    ]
    events_list = in_window[["start_sec", "end_sec", "code", "Team"]].to_dict("records")

    # Average risk in window
    window_risk = risk_df[
        (risk_df["timestamp_sec"] >= req.start_sec) &
        (risk_df["timestamp_sec"] <= req.end_sec)
    ]
    avg_risk = float(window_risk["risk_score"].mean()) if len(window_risk) > 0 else 0.0

    explanation = explain_window(
        events_list, req.start_sec, req.end_sec,
        match_name, opponent, avg_risk,
        offset_sec, h2_start_sec
    )

    return {
        "explanation": explanation,
        "avg_risk": round(avg_risk, 1),
        "event_count": len(events_list),
    }


# Serve built React app in production
if os.path.exists("dashboard/dist"):
    app.mount("/assets", StaticFiles(directory="dashboard/dist/assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse("dashboard/dist/index.html")
