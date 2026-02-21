import os
import re
import csv
import sys
import glob
import math
import time
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, Optional, List

# -----------------------------
# Helpers: file discovery
# -----------------------------

def find_match_files(match_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (xml_path, raw_txt_path) for a match folder, if found.
    """
    xml_candidates = glob.glob(os.path.join(match_dir, "*_FifaData.xml"))
    txt_candidates = glob.glob(os.path.join(match_dir, "*_FifaDataRawData.txt"))

    xml_path = xml_candidates[0] if xml_candidates else None
    txt_path = txt_candidates[0] if txt_candidates else None
    return xml_path, txt_path


# -----------------------------
# XML roster parsing
# -----------------------------

def parse_fifa_xml_roster(xml_path: str) -> Dict[str, dict]:
    """
    Parses FIFA XML to build a roster lookup.
    Returns dict keyed by playerId (string) -> info dict.
    """
    roster = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # FIFA XMLs vary; we try a few likely paths.
    # We'll collect any node that looks like a Player with id + name.
    # You can adjust tags once you confirm your XML structure.

    # Common pattern: search all elements that have attributes like "uID" or "id"
    # and child nodes like FirstName/LastName/ShirtNumber/TeamId.
    for player in root.iter():
        tag = player.tag.lower()

        # Heuristic: identify player nodes
        if "player" in tag:
            attrs = {k.lower(): v for k, v in player.attrib.items()}

            # possible ids
            pid = attrs.get("uid") or attrs.get("id") or attrs.get("u_id") or attrs.get("playerid")
            if not pid:
                continue

            # try to pull fields from children
            first = None
            last = None
            shirt = None
            team = None
            position = None

            for child in list(player):
                ctag = child.tag.lower()
                text = (child.text or "").strip()

                if ctag in ("firstname", "first", "givenname"):
                    first = text
                elif ctag in ("lastname", "last", "surname", "familyname"):
                    last = text
                elif ctag in ("shirtnumber", "shirt", "number", "jersey"):
                    shirt = text
                elif ctag in ("teamid", "team", "side"):
                    team = text
                elif ctag in ("position", "pos"):
                    position = text

            name = " ".join([p for p in [first, last] if p]) or None

            roster[str(pid)] = {
                "player_id": str(pid),
                "name": name,
                "shirt_number": shirt,
                "team_id": team,
                "position_label": position,
            }

    return roster


# -----------------------------
# Raw tracking parsing
# -----------------------------

# This is the key: parse positional lines without needing any parquet libs.
# Since tracking formats vary, we implement a flexible line parser:
# - Extract timestamp/frame
# - Extract repeating (playerId, x, y) tuples
# - Optionally extract ball (x, y)

NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def parse_tracking_line(line: str):
    """
    Attempts to parse one tracking line and return:
    (time_s, frame_idx, players: List[(player_id, x, y)], ball_xy: Optional[(x, y)])
    Returns None if the line doesn't look like a tracking frame.
    """

    # Skip empty/comment lines
    s = line.strip()
    if not s:
        return None

    # Heuristic: look for at least 3 numbers on the line
    nums = re.findall(NUM_RE, s)
    if len(nums) < 6:
        return None

    # Try to interpret first numbers as time/frame
    # Many formats start with: frame, time, ...
    # We'll attempt both and keep the one that seems plausible.
    # If you know the format exactly, we can lock this down.

    # Candidate A: [frame, time, ...]
    frame_a = int(float(nums[0]))
    time_a = float(nums[1])

    # Candidate B: [time, frame, ...]
    time_b = float(nums[0])
    frame_b = int(float(nums[1]))

    def time_plausible(t):
        return 0 <= t <= 3 * 60 * 60  # <= 3 hours

    if time_plausible(time_a):
        frame_idx = frame_a
        time_s = time_a
        start_i = 2
    elif time_plausible(time_b):
        frame_idx = frame_b
        time_s = time_b
        start_i = 2
    else:
        # can't decide; assume frame,time
        frame_idx = frame_a
        time_s = time_a
        start_i = 2

    # Now parse remaining numbers into tuples.
    # Common possibilities:
    #   - repeating groups of 3: (playerId, x, y)
    #   - ball at the end as (x, y) or prefixed in some way
    rest = nums[start_i:]

    players = []
    ball_xy = None

    # Prefer groups of 3 for players
    # We'll take as many full triples as possible
    triple_count = len(rest) // 3

    for i in range(triple_count):
        pid = rest[i*3 + 0]
        x = rest[i*3 + 1]
        y = rest[i*3 + 2]

        # Player IDs are usually integers; if it doesn't look like one, skip
        # (this avoids accidentally treating ball coords as a "player")
        try:
            pid_int = int(float(pid))
        except ValueError:
            continue

        try:
            xf = float(x)
            yf = float(y)
        except ValueError:
            continue

        players.append((str(pid_int), xf, yf))

    # If there are leftover numbers (1-2), interpret last two as ball if plausible
    leftover = rest[triple_count*3:]
    if len(leftover) >= 2:
        try:
            bx = float(leftover[-2])
            by = float(leftover[-1])
            # simple plausibility: pitch coords usually within [-200, 200] range or meters
            if abs(bx) < 5000 and abs(by) < 5000:
                ball_xy = (bx, by)
        except ValueError:
            pass

    if not players and ball_xy is None:
        return None

    return time_s, frame_idx, players, ball_xy


# -----------------------------
# Writing outputs
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_player_positions_csv(out_path: str, rows: List[dict]):
    """
    Writes player position rows to CSV.
    """
    if not rows:
        return

    fieldnames = [
        "match",
        "time_s",
        "frame",
        "player_id",
        "player_name",
        "shirt_number",
        "team_id",
        "x",
        "y",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def write_ball_positions_csv(out_path: str, rows: List[dict]):
    if not rows:
        return

    fieldnames = ["match", "time_s", "frame", "ball_x", "ball_y"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# -----------------------------
# Main batch runner
# -----------------------------

def process_match_folder(match_dir: str, out_root: str, write_ball: bool = True) -> bool:
    match_name = os.path.basename(match_dir.rstrip("\\/"))

    xml_path, txt_path = find_match_files(match_dir)
    if not xml_path or not txt_path:
        print(f"  SKIP: Missing XML or RawData TXT in: {match_name}")
        return False

    print(f"  XML: {os.path.basename(xml_path)}")
    print(f"  TXT: {os.path.basename(txt_path)}")

    roster = parse_fifa_xml_roster(xml_path)

    out_dir = os.path.join(out_root, match_name)
    ensure_dir(out_dir)

    player_rows = []
    ball_rows = []

    # Stream-read the big tracking file (fast + low memory)
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_tracking_line(line)
            if not parsed:
                continue

            time_s, frame_idx, players, ball_xy = parsed

            for pid, x, y in players:
                info = roster.get(pid, {})
                player_rows.append({
                    "match": match_name,
                    "time_s": time_s,
                    "frame": frame_idx,
                    "player_id": pid,
                    "player_name": info.get("name"),
                    "shirt_number": info.get("shirt_number"),
                    "team_id": info.get("team_id"),
                    "x": x,
                    "y": y,
                })

            if write_ball and ball_xy is not None:
                bx, by = ball_xy
                ball_rows.append({
                    "match": match_name,
                    "time_s": time_s,
                    "frame": frame_idx,
                    "ball_x": bx,
                    "ball_y": by,
                })

    player_csv = os.path.join(out_dir, "player_positions.csv")
    write_player_positions_csv(player_csv, player_rows)

    if write_ball:
        ball_csv = os.path.join(out_dir, "ball_positions.csv")
        write_ball_positions_csv(ball_csv, ball_rows)

    print(f"  Wrote: {player_csv} ({len(player_rows)} rows)")
    if write_ball:
        print(f"  Wrote: {os.path.join(out_dir, 'ball_positions.csv')} ({len(ball_rows)} rows)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches_dir", default="matches", help="Root folder containing match subfolders")
    parser.add_argument("--out_dir", default="outputs", help="Where to write parsed CSVs")
    parser.add_argument("--no_ball", action="store_true", help="Do not write ball_positions.csv")
    args = parser.parse_args()

    matches_dir = args.matches_dir
    out_root = args.out_dir
    write_ball = not args.no_ball

    if not os.path.isdir(matches_dir):
        print(f"ERROR: matches_dir not found: {matches_dir}")
        sys.exit(1)

    match_folders = [
        os.path.join(matches_dir, d)
        for d in os.listdir(matches_dir)
        if os.path.isdir(os.path.join(matches_dir, d))
    ]

    print(f"Found {len(match_folders)} match folders.\n")

    ok = 0
    for folder in match_folders:
        print(f"Processing: {os.path.basename(folder)}")
        try:
            if process_match_folder(folder, out_root, write_ball=write_ball):
                ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    print(f"Done. Successfully processed {ok}/{len(match_folders)} folders.")
    print(f"Outputs written under: {out_root}/")


if __name__ == "__main__":
    main()