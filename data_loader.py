# data_loader.py
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import pandas as pd
import json
from pathlib import Path


MATCHES_DIR = Path("matches")


def get_halftime_offset(events_df: pd.DataFrame) -> pd.Timedelta:
    """
    Compute the halftime break duration as the gap between the end of the
    last 1st-half event and the start of the first 2nd-half event.
    Returns Timedelta(0) if either half is missing.
    """
    half_col = "Half" if "Half" in events_df.columns else "half"

    h1 = events_df[events_df[half_col] == "1st Half"]
    h2 = events_df[events_df[half_col] == "2nd Half"]

    if h1.empty or h2.empty:
        return pd.Timedelta(0)

    h1_end   = h1["end_timestamp"].max()
    h2_start = h2["timestamp"].min()
    offset   = h2_start - h1_end

    return offset if offset > pd.Timedelta(0) else pd.Timedelta(0)

def list_matches(matches_dir: Path = MATCHES_DIR) -> list[str]:
    """
    Returns match folder names under ./matches
    """
    if not matches_dir.exists():
        return []
    return sorted([p.name for p in matches_dir.iterdir() if p.is_dir()])


def _strip_ns(tag: str) -> str:
    """
    Removes XML namespace if present: {ns}Tag -> Tag
    """
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_pattern_xml(match_dir: Path) -> Optional[Path]:
    """
    Finds the *_pattern.xml file inside a match folder.
    """
    candidates = glob.glob(str(match_dir / "*_pattern.xml"))
    if not candidates:
        return None
    return Path(candidates[0])


def load_events(match_name: str, matches_dir: Path = MATCHES_DIR) -> pd.DataFrame:
    """
    Loads events from the match's *_pattern.xml.

    Output columns (minimum needed by the rest of your pipeline):
      - match
      - code
      - team
      - timestamp   (pd.Timedelta)
      - end_timestamp (pd.Timedelta)
      - start_s, end_s (float seconds)
      - half (optional string if present)
    """
    match_dir = matches_dir / match_name
    if not match_dir.exists():
        raise FileNotFoundError(f"Missing match folder: {match_dir}")

    pattern_path = _find_pattern_xml(match_dir)
    if pattern_path is None:
        raise FileNotFoundError(f"Missing *_pattern.xml in: {match_dir}")

    tree = ET.parse(pattern_path)
    root = tree.getroot()

    rows: list[dict] = []

    # The file structure you showed uses <instance> ... </instance>
    for inst in root.iter():
        if _strip_ns(inst.tag) != "instance":
            continue

        start_s = None
        end_s = None
        code = None
        team = None
        half = None

        for child in list(inst):
            tag = _strip_ns(child.tag)
            text = (child.text or "").strip()

            if tag == "start":
                try:
                    start_s = float(text)
                except ValueError:
                    start_s = None
            elif tag == "end":
                try:
                    end_s = float(text)
                except ValueError:
                    end_s = None
            elif tag == "code":
                code = text
            elif tag == "label":
                # labels look like:
                # <label><text>FC Barcelona</text><group>Team</group></label>
                label_text = None
                label_group = None
                for lab_child in list(child):
                    lab_tag = _strip_ns(lab_child.tag)
                    lab_val = (lab_child.text or "").strip()
                    if lab_tag == "text":
                        label_text = lab_val
                    elif lab_tag == "group":
                        label_group = lab_val

                if label_group == "Team":
                    team = label_text
                elif label_group == "Half":
                    half = label_text

        # Keep only valid timed instances
        if start_s is None or end_s is None or code is None:
            continue

        rows.append(
            {
                "match_name": match_name,
                "code": code,
                "Team": team or "N/A",
                "Half": half,
                "start_s": float(start_s),
                "end_s": float(end_s),
                "timestamp": pd.to_timedelta(start_s, unit="s"),
                "end_timestamp": pd.to_timedelta(end_s, unit="s"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No <instance> events parsed from: {pattern_path}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def get_halftime_offset(events_df: pd.DataFrame) -> pd.Timedelta:
    """
    Compute the halftime break duration as the gap between the end of the
    last 1st-half event and the start of the first 2nd-half event.
    Returns Timedelta(0) if either half is missing.
    """
    half_col = "Half" if "Half" in events_df.columns else "half"

    h1 = events_df[events_df[half_col] == "1st Half"]
    h2 = events_df[events_df[half_col] == "2nd Half"]

    if h1.empty or h2.empty:
        return pd.Timedelta(0)

    h1_end   = h1["end_timestamp"].max()
    h2_start = h2["timestamp"].min()
    offset   = h2_start - h1_end

    return offset if offset > pd.Timedelta(0) else pd.Timedelta(0)


def load_team_map(match_parsed_dir: Path) -> dict:
    """
    Load team_map.json from a parsed match folder (created by tracking_batch_parser.py).
    Returns {} if missing or invalid.
    """
    try:
        p = Path(match_parsed_dir) / "team_map.json"
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}