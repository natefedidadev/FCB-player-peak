import re
import math
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

XML_PATH = r"/mnt/data/AC Milan - Barça (0-1) Partit Amistós Gira 2023-2024_FifaData.xml"
TXT_PATH = r"/mnt/data/your_tracking.txt"  # <-- change this
OUT_PARQUET = r"/mnt/data/tracking_long.parquet"
OUT_CSV = r"/mnt/data/tracking_long.csv"

def safe_float(s):
    s = s.strip()
    if s.lower() == "nan" or s == "":
        return float("nan")
    return float(s)

def parse_fifa_xml(xml_path: str) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    meta = root.find("Metadata")

    # Teams
    team_map = {}
    for t in meta.find("Teams"):
        tid = t.attrib.get("id")
        name = (t.findtext("Name") or "").strip()
        team_map[tid] = name or tid

    # Players
    rows = []
    for p in meta.find("Players"):
        pid = p.attrib.get("id")
        team_id = p.attrib.get("teamId")
        name = (p.findtext("Name") or "").strip()  # often "Track 1" style
        shirt = (p.findtext("ShirtNumber") or "").strip()

        rows.append({
            "player_id": pid,
            "team_id": team_id,
            "team": team_map.get(team_id, team_id),
            "player_label": name if name else pid,
            "shirt": shirt if shirt else None
        })

    # NOTE: Many exports don’t include real player names; you’ll see "Track 1", etc.
    return pd.DataFrame(rows)

def parse_tracking_txt(txt_path: str, roster: pd.DataFrame) -> pd.DataFrame:
    lines = Path(txt_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    # Build an ordered list of player labels based on roster order.
    # In many FIFA exports, roster order matches Track 1..N.
    roster = roster.reset_index(drop=True)
    roster["track_index"] = roster.index + 1  # Track 1 => 1, etc.

    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # frame_id : players_blob : ball_blob
        parts = line.split(":")
        if len(parts) < 3:
            continue  # skip malformed

        frame_id = int(parts[0])

        # everything between first and last colon = players section (in case extra colons exist)
        players_blob = ":".join(parts[1:-1])
        ball_blob = parts[-1]

        # players are separated by ';'
        player_tokens = [t for t in players_blob.split(";") if t.strip()]

        # parse players
        for i, tok in enumerate(player_tokens, start=1):
            vals = tok.split(",")
            if len(vals) < 3:
                continue
            x, y, spd = map(safe_float, vals[:3])

            # attach roster info if available
            r = roster[roster["track_index"] == i]
            if len(r) == 1:
                team = r.iloc[0]["team"]
                player_label = r.iloc[0]["player_label"]
                shirt = r.iloc[0]["shirt"]
            else:
                team = None
                player_label = f"Track {i}"
                shirt = None

            out.append({
                "frame_id": frame_id,
                "entity_type": "player",
                "track_id": i,
                "team": team,
                "player_label": player_label,
                "shirt": shirt,
                "x": x, "y": y, "speed": spd
            })

        # parse ball (track_id = 0)
        bvals = ball_blob.split(",")
        if len(bvals) >= 3:
            bx, by, bspd = map(safe_float, bvals[:3])
            out.append({
                "frame_id": frame_id,
                "entity_type": "ball",
                "track_id": 0,
                "team": None,
                "player_label": "ball",
                "shirt": None,
                "x": bx, "y": by, "speed": bspd
            })

    return pd.DataFrame(out)

if __name__ == "__main__":
    roster = parse_fifa_xml(XML_PATH)
    df_long = parse_tracking_txt(TXT_PATH, roster)

    # optional: sort nicely
    df_long = df_long.sort_values(["frame_id", "entity_type", "track_id"]).reset_index(drop=True)

    # save
    df_long.to_parquet(OUT_PARQUET, index=False)
    df_long.to_csv(OUT_CSV, index=False)

    print("Wrote:", OUT_PARQUET)
    print("Wrote:", OUT_CSV)
    print(df_long.head(10))