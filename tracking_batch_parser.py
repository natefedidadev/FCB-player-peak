# tracking_batch_parser.py
import os
import re
import csv
import sys
import glob
import argparse
import xml.etree.ElementTree as ET
import json
import pandas as pd
from typing import Dict, Tuple, Optional, List

NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


# -----------------------------
# Helpers: file discovery
# -----------------------------
def find_match_files(match_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (fifa_xml_path, rawdata_txt_path, pattern_xml_path) for a match directory.
    """
    xml_candidates = glob.glob(os.path.join(match_dir, "*_FifaData.xml"))
    txt_candidates = glob.glob(os.path.join(match_dir, "*_FifaDataRawData.txt"))
    pattern_candidates = glob.glob(os.path.join(match_dir, "*_pattern.xml"))

    xml_path = xml_candidates[0] if xml_candidates else None
    txt_path = txt_candidates[0] if txt_candidates else None
    pattern_path = pattern_candidates[0] if pattern_candidates else None
    return xml_path, txt_path, pattern_path


# -----------------------------
# XML roster parsing (your FIFA XML format)
# -----------------------------
def parse_fifa_xml_roster(xml_path: str) -> Dict[str, dict]:
    """
    Parses FIFA XML to build a roster lookup.
    Returns dict keyed by playerId (string) -> info dict.
    Handles structure like:
      <Player id="24" teamId="...">
         <Name>Track 24</Name>
         <ShirtNumber>24</ShirtNumber>
         ...
      </Player>
    """
    roster: Dict[str, dict] = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for el in root.iter():
        # Match tags that end with "Player" (namespace-safe-ish)
        tag = el.tag.split("}")[-1]
        if tag != "Player":
            continue

        pid = el.attrib.get("id") or el.attrib.get("uID") or el.attrib.get("uid")
        if not pid:
            continue

        team_id = el.attrib.get("teamId") or el.attrib.get("teamID") or el.attrib.get("team_id")

        name = None
        shirt = None
        position = None

        for child in list(el):
            ctag = child.tag.split("}")[-1]
            text = (child.text or "").strip()

            if ctag == "Name":
                name = text or None
            elif ctag == "ShirtNumber":
                shirt = text or None
            elif ctag == "ProviderPlayerParameters":
                # optional: try to extract position_type if present
                for pp in list(child):
                    pptag = pp.tag.split("}")[-1]
                    if pptag != "ProviderParameter":
                        continue
                    n = None
                    v = None
                    for ppchild in list(pp):
                        subtag = ppchild.tag.split("}")[-1]
                        subtext = (ppchild.text or "").strip()
                        if subtag == "Name":
                            n = subtext
                        elif subtag == "Value":
                            v = subtext
                    if n == "position_type":
                        position = v

        roster[str(pid)] = {
            "player_id": str(pid),
            "name": name,
            "shirt_number": shirt,
            "team_id": team_id,
            "position_label": position,
        }

    return roster


# -----------------------------
# MORE HELPERS WITH PARSING
# -----------------------------
def _parse_trackslot_payload(payload: str):
    """Return list of (x,y,z) floats or (nan,nan,nan) for each slot."""
    out = []
    for t in payload.split(";"):
        t = t.strip()
        if not t:
            continue
        parts = t.split(",")
        if len(parts) < 3:
            continue
        try:
            x = float(parts[0]); y = float(parts[1]); z = float(parts[2])
        except Exception:
            x = y = z = float("nan")
        out.append((x, y, z))
    return out


def infer_ball_slot_index_from_txt(txt_path: str, sample_lines: int = 3000) -> int | None:
    """
    Scan first N lines and infer which slot is the ball.
    Returns slot index (0-based) or None.
    """
    import math

    # stats per slot
    counts = []          # non-nan count
    move_sum = []        # sum of frame-to-frame movement in xy
    z_var_sum = []       # rough variability in z (sum of abs dz)
    prev_xy = []
    prev_z = []

    def ensure(n):
        nonlocal counts, move_sum, z_var_sum, prev_xy, prev_z
        while len(counts) < n:
            counts.append(0)
            move_sum.append(0.0)
            z_var_sum.append(0.0)
            prev_xy.append(None)
            prev_z.append(None)

    def finite(a): 
        return a is not None and not (math.isnan(a) or math.isinf(a))

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            s = line.strip()
            if ":" not in s or ";" not in s:
                continue

            parts = s.split(":")
            if len(parts) < 2:
                continue
            payload = parts[1].strip()
            slots = _parse_trackslot_payload(payload)
            ensure(len(slots))

            for idx, (x, y, z) in enumerate(slots):
                if finite(x) and finite(y):
                    counts[idx] += 1
                    # movement
                    p = prev_xy[idx]
                    if p is not None:
                        dx = x - p[0]; dy = y - p[1]
                        move_sum[idx] += (dx*dx + dy*dy) ** 0.5
                    prev_xy[idx] = (x, y)

                if finite(z):
                    pz = prev_z[idx]
                    if pz is not None:
                        z_var_sum[idx] += abs(z - pz)
                    prev_z[idx] = z

    if not counts:
        return None

    # Score slots: prefer high presence + high movement + some z variability
    best_idx = None
    best_score = -1.0
    for idx in range(len(counts)):
        if counts[idx] < 50:   # must appear reasonably often in sample
            continue
        score = counts[idx] * (1.0 + move_sum[idx]) * (1.0 + 0.25 * z_var_sum[idx])
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx

# -----------------------------
# XML team parsing (attempt to map teamId -> team name)
# -----------------------------
def parse_fifa_xml_teams(xml_path: str) -> Dict[str, str]:
    """
    Best-effort parser to extract team id -> team name from the FIFA XML.
    This is intentionally defensive because XML schemas vary between providers.
    """
    team_map: Dict[str, str] = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return team_map

    def local(tag: str) -> str:
        return tag.split("}")[-1] if tag else tag

    # Common patterns: <Team id="..."><Name>FC Barcelona</Name>...</Team>
    # Sometimes: <TeamData teamId="..."><TeamName>...</TeamName></TeamData>
    for el in root.iter():
        tag = local(el.tag)
        if tag == "Player":
            continue

        # Heuristically accept nodes that look like team containers
        looks_like_team = tag in {"Team", "TeamData", "TeamInfo"} or tag.endswith("Team")
        if not looks_like_team:
            continue

        tid = el.attrib.get("teamId") or el.attrib.get("teamID") or el.attrib.get("team_id") or el.attrib.get("id") or el.attrib.get("uID") or el.attrib.get("uid")
        if not tid:
            continue

        name = None
        for child in list(el):
            ctag = local(child.tag)
            ctext = (child.text or "").strip()
            if ctag in {"Name", "TeamName", "ShortName", "ClubName"} and ctext:
                name = ctext
                break

        # Some schemas store name in attributes
        if not name:
            for key in ("name", "teamName", "shortName", "clubName"):
                v = el.attrib.get(key)
                if v and v.strip():
                    name = v.strip()
                    break

        if name:
            team_map[str(tid)] = name

    return team_map


def infer_barca_team_id(team_id_to_name: Dict[str, str]) -> Optional[str]:
    """
    Identify Barcelona's teamId by team name tokens. Returns None if unknown.
    """
    tokens = ("barcelona", "fc barcelona", "barça", "fcb", "f.c. barcelona")
    for tid, name in team_id_to_name.items():
        if not name:
            continue
        n = name.lower()
        if any(tok in n for tok in tokens):
            return tid
    return None


# -----------------------------
# Raw tracking parsing
# -----------------------------
def _is_intlike(x: str) -> bool:
    try:
        xf = float(x)
        return abs(xf - int(xf)) < 1e-9
    except Exception:
        return False


import math
import re
from typing import List, Tuple, Optional

# If you already have NUM_RE/NUM_RE compiled elsewhere, keep yours.
# This pattern matches floats/ints with optional sign/exponent.
NUM_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

# --- module-level state for frame->time when only frames exist ---
_TRACK_BASE_FRAME = None
_TRACK_FPS = 25.0  # you can change to 30.0 if your feed is 30fps


def _is_finite(x: float) -> bool:
    return x is not None and not (math.isnan(x) or math.isinf(x))


def parse_tracking_line(line: str):
    """
    Returns:
      (time_s, frame_idx, players: List[(player_id, x, y)], ball_xy, slots_xyz)

    slots_xyz:
      - For track-slot format lines: list of (x,y,z) for each slot (may contain NaNs).
      - For numeric pid/x/y format lines: None.
    """
    s = line.strip()
    if not s:
        return None

    # -----------------------------
    # Track-slot format: "frame: x,y,z; x,y,z; NaN,NaN,NaN; ..."
    # -----------------------------
    if ":" in s and ";" in s:
        parts = s.split(":")
        try:
            frame_idx = int(float(parts[0].strip()))
        except Exception:
            return None

        payload = parts[1].strip() if len(parts) >= 2 else ""
        if not payload:
            return None

        global _TRACK_BASE_FRAME
        if _TRACK_BASE_FRAME is None:
            _TRACK_BASE_FRAME = frame_idx

        time_s = float(frame_idx - _TRACK_BASE_FRAME) / float(_TRACK_FPS)

        slots_xyz = []
        players = []
        ball_xy = None  # we'll fill it later using ball_slot_idx

        triplets = payload.split(";")
        for slot_i, t in enumerate(triplets):
            t = t.strip()
            if not t:
                continue
            xyz = t.split(",")
            if len(xyz) < 3:
                continue
            try:
                x = float(xyz[0])
                y = float(xyz[1])
                z = float(xyz[2])
            except Exception:
                x = y = z = float("nan")

            slots_xyz.append((x, y, z))

            # Build pseudo "players" list from any finite (x,y)
            if not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y)):
                pid = str(slot_i + 1)
                players.append((pid, x, y))

        if not players and not slots_xyz:
            return None

        return time_s, frame_idx, players, ball_xy, slots_xyz

    # -----------------------------
    # Numeric pid,x,y format (your old parser behavior)
    # -----------------------------
    nums = re.findall(NUM_RE, s)
    if len(nums) < 6:
        return None

    a0, a1 = nums[0], nums[1]

    # Candidates
    frame_a = int(float(a0)) if _is_intlike(a0) else None
    time_a = float(a1)
    time_b = float(a0)
    frame_b = int(float(a1)) if _is_intlike(a1) else None

    def time_plausible(t: float) -> bool:
        return 0 <= t <= 3 * 60 * 60

    choose_b = False
    if time_b < 10 and frame_b is not None and frame_b >= 10:
        choose_b = True
    elif frame_a is not None and time_a < 10:
        choose_b = False
    else:
        if frame_b is not None and time_plausible(time_b) and not (frame_a is not None and time_plausible(time_a)):
            choose_b = True
        elif frame_a is None and frame_b is not None:
            choose_b = True

    if choose_b and frame_b is not None and time_plausible(time_b):
        time_s = float(time_b)
        frame_idx = int(frame_b)
        start_i = 2
    else:
        if frame_a is None:
            return None
        time_s = float(time_a)
        frame_idx = int(frame_a)
        start_i = 2

    rest = nums[start_i:]
    players = []
    ball_xy = None

    triple_count = len(rest) // 3
    for i in range(triple_count):
        pid_s = rest[i * 3 + 0]
        x_s = rest[i * 3 + 1]
        y_s = rest[i * 3 + 2]

        if not _is_intlike(pid_s):
            continue

        pid_int = int(float(pid_s))
        if pid_int == 0:
            continue

        try:
            xf = float(x_s)
            yf = float(y_s)
        except Exception:
            continue

        players.append((str(pid_int), xf, yf))

    leftover = rest[triple_count * 3 :]
    if len(leftover) >= 2:
        try:
            bx = float(leftover[-2])
            by = float(leftover[-1])
            if abs(bx) < 5000 and abs(by) < 5000:
                ball_xy = (bx, by)
        except Exception:
            pass

    if not players and ball_xy is None:
        return None

    return time_s, frame_idx, players, ball_xy, None


# -----------------------------
# Writing outputs
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_player_positions_csv(out_path: str, rows: List[dict]):
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
def process_match_folder(match_dir: str, out_root: str, write_ball: bool = True, overwrite: bool = False) -> bool:
    match_name = os.path.basename(match_dir.rstrip("\\/"))

    xml_path, txt_path, pattern_path = find_match_files(match_dir)
    if not xml_path or not txt_path:
        print(f"  SKIP: Missing FIFA XML or RawData TXT in: {match_name}")
        return False

    out_dir = os.path.join(out_root, match_name)
    if os.path.exists(out_dir) and not overwrite:
        # if output exists, skip unless overwrite
        if os.path.exists(os.path.join(out_dir, "player_positions.csv")):
            print(f"  SKIP: Already parsed (use --overwrite to re-run): {match_name}")
            return True

    ensure_dir(out_dir)

    print(f"  XML: {os.path.basename(xml_path)}")
    print(f"  TXT: {os.path.basename(txt_path)}")
    if pattern_path:
        print(f"  PATTERN: {os.path.basename(pattern_path)}")

    roster = parse_fifa_xml_roster(xml_path)
    team_id_to_name = parse_fifa_xml_teams(xml_path)
    barca_team_id = infer_barca_team_id(team_id_to_name)

    # Opponent is the other team id (if exactly 2 teams exist)
    opp_team_id = None
    if barca_team_id and len(team_id_to_name) >= 2:
        for tid in team_id_to_name.keys():
            if tid != barca_team_id:
                opp_team_id = tid
                break

    # Write team mapping artifact for downstream feature engineering
    team_map_path = os.path.join(out_dir, "team_map.json")
    try:
        with open(team_map_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "barca_team_id": barca_team_id,
                    "opponent_team_id": opp_team_id,
                    "team_id_to_name": team_id_to_name,
                    "barca_name": team_id_to_name.get(barca_team_id) if barca_team_id else None,
                    "opponent_name": team_id_to_name.get(opp_team_id) if opp_team_id else None,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        pass

    player_rows: List[dict] = []
    ball_rows: List[dict] = []

    global _TRACK_BASE_FRAME
    _TRACK_BASE_FRAME = None

    ball_slot_idx = None
    if write_ball:
        ball_slot_idx = infer_ball_slot_index_from_txt(txt_path, sample_lines=3000)
        print(f"  Inferred ball slot index: {ball_slot_idx}")

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_tracking_line(line)
            if not parsed:
                continue

            time_s, frame_idx, players, ball_xy, slots_xyz = parsed

            for pid, x, y in players:
                info = roster.get(pid, {})
                player_rows.append(
                    {
                        "match": match_name,
                        "time_s": time_s,
                        "frame": frame_idx,
                        "player_id": pid,
                        "player_name": info.get("name"),
                        "shirt_number": info.get("shirt_number"),
                        "team_id": info.get("team_id"),
                        "x": x,
                        "y": y,
                    }
                )

            # if track-slot format and ball not set, pick it from slots
            if write_ball and ball_xy is None and slots_xyz is not None and ball_slot_idx is not None:
                if 0 <= ball_slot_idx < len(slots_xyz):
                    bx, by, bz = slots_xyz[ball_slot_idx]
                    if not (math.isnan(bx) or math.isnan(by)):
                        ball_xy = (bx, by)

            # append ball row
            if write_ball and ball_xy is not None:
                bx, by = ball_xy
                ball_rows.append(
                    {"match": match_name, "time_s": time_s, "frame": frame_idx, "ball_x": bx, "ball_y": by}
                )

    # -------------------------------
    # NEW: Correct ball time using frame->time mapping from players
    # -------------------------------
    if write_ball and ball_rows and player_rows:
        try:
            player_df_tmp = pd.DataFrame(player_rows)
            ball_df_tmp = pd.DataFrame(ball_rows)

            player_df_tmp["time_s"] = pd.to_numeric(player_df_tmp["time_s"], errors="coerce")
            player_df_tmp["frame"] = pd.to_numeric(player_df_tmp["frame"], errors="coerce")
            ball_df_tmp["frame"] = pd.to_numeric(ball_df_tmp["frame"], errors="coerce")

            print("  DEBUG frames:")
            print("    player frame min/max:", player_df_tmp["frame"].min(), player_df_tmp["frame"].max())
            print("    ball frame min/max:", ball_df_tmp["frame"].min(), ball_df_tmp["frame"].max())

            frame_time = (
                player_df_tmp.dropna(subset=["frame", "time_s"])
                .groupby("frame", as_index=False)["time_s"]
                .median()
            )

            ball_df_tmp = ball_df_tmp.drop(columns=["time_s"], errors="ignore")
            ball_df_tmp = ball_df_tmp.merge(frame_time, on="frame", how="left")

            before = len(ball_df_tmp)
            ball_df_tmp = ball_df_tmp.dropna(subset=["time_s"])
            after = len(ball_df_tmp)

            ball_rows = ball_df_tmp.to_dict(orient="records")

            if before != after:
                print(f"  Ball time remap: dropped {before - after} ball rows with no frame->time mapping")
        except Exception as e:
            print(f"  WARN: Failed to remap ball time from frames ({e}). Keeping original ball time_s.")
    
    if write_ball and ball_rows and not player_rows:
        print("  WARN: No player rows parsed, cannot remap ball time from frames.")

    # Write CSVs
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
    parser.add_argument("--out_dir", default="parsed", help="Where to write parsed CSVs")
    parser.add_argument("--no_ball", action="store_true", help="Do not write ball_positions.csv")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parsed outputs")
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

    print(f"Found {len(match_folders)} match folders with FIFA XML + RawData TXT.\n")

    ok = 0
    for folder in match_folders:
        print(f"Processing: {os.path.basename(folder)}")
        try:
            if process_match_folder(folder, out_root, write_ball=write_ball, overwrite=args.overwrite):
                ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    print(f"Done. Successfully processed {ok}/{len(match_folders)} folders.")
    print(f"Outputs written under: {out_root}/")


if __name__ == "__main__":
    main()