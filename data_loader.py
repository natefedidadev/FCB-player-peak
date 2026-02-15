"""
Data loader for FCB Player Peak — converts match files into pandas DataFrames.
Uses kloppy for parsing Metrica Sports EPTS tracking and SportsCoding events.

Three data sources per match:
  *_pattern.xml         → events_df     (tactical phase annotations via kloppy.sportscode)
  *_FifaData.xml +
  *_FifaDataRawData.txt → tracking_df   (per-frame x/y/speed via kloppy.metrica EPTS)
"""

from pathlib import Path
from kloppy import metrica, sportscode
import pandas as pd


MATCHES_DIR = Path("matches")


def list_matches(matches_dir: Path = MATCHES_DIR) -> list[str]:
    """Return sorted list of match folder names."""
    if not matches_dir.exists():
        return []
    return sorted(d.name for d in matches_dir.iterdir() if d.is_dir())


def _match_path(match: str | int, matches_dir: Path = MATCHES_DIR) -> Path:
    """Resolve a match name or index to its folder path."""
    if isinstance(match, int):
        names = list_matches(matches_dir)
        return matches_dir / names[match]
    return matches_dir / match


def _find_file(match_path: Path, pattern: str) -> Path:
    """Find a single file matching a glob pattern in a match folder."""
    files = list(match_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} in {match_path}")
    return files[0]


# ---------------------------------------------------------------------------
# Events (pattern.xml via kloppy sportscode)
# ---------------------------------------------------------------------------

def load_events(match: str | int, matches_dir: Path = MATCHES_DIR) -> pd.DataFrame:
    """
    Parse *_pattern.xml → DataFrame using kloppy sportscode.

    Columns: code_id, period_id, timestamp, end_timestamp, code,
             Team, Half, Type, Side, Direction of ball entry,
             Max Players in the box, match_name
    """
    mp = _match_path(match, matches_dir)
    xml_file = _find_file(mp, "*_pattern.xml")
    dataset = sportscode.load(str(xml_file))
    df = dataset.to_df()
    df["match_name"] = mp.name
    return df


# ---------------------------------------------------------------------------
# Tracking (FifaData.xml + FifaDataRawData.txt via kloppy metrica EPTS)
# ---------------------------------------------------------------------------

def load_tracking(
    match: str | int,
    matches_dir: Path = MATCHES_DIR,
    sample_rate: float | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Parse tracking data via kloppy metrica EPTS → DataFrame.

    Uses *_FifaData.xml (metadata) + *_FifaDataRawData.txt (raw positions).

    Columns: period_id, timestamp, frame_id, ball_x/y/z/speed,
             <player_id>_x, <player_id>_y, <player_id>_d, <player_id>_s
             for each player track.

    Args:
        sample_rate: Downsample to this frame rate (e.g. 5.0 for 5fps instead of 25fps).
        limit: Only load the first N frames.
    """
    mp = _match_path(match, matches_dir)
    meta_file = _find_file(mp, "*_FifaData.xml")
    raw_file = _find_file(mp, "*_FifaDataRawData.txt")
    dataset = metrica.load_tracking_epts(
        meta_data=str(meta_file),
        raw_data=str(raw_file),
        sample_rate=sample_rate,
        limit=limit,
    )
    df = dataset.to_df()
    df["match_name"] = mp.name
    return df


def load_tracking_dataset(
    match: str | int,
    matches_dir: Path = MATCHES_DIR,
    sample_rate: float | None = None,
    limit: int | None = None,
):
    """
    Load tracking as a kloppy TrackingDataset object (not a DataFrame).
    Useful when you need access to metadata, teams, players, pitch dimensions, etc.
    """
    mp = _match_path(match, matches_dir)
    meta_file = _find_file(mp, "*_FifaData.xml")
    raw_file = _find_file(mp, "*_FifaDataRawData.txt")
    return metrica.load_tracking_epts(
        meta_data=str(meta_file),
        raw_data=str(raw_file),
        sample_rate=sample_rate,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Convenience: load everything for a match
# ---------------------------------------------------------------------------

def load_match(
    match: str | int,
    matches_dir: Path = MATCHES_DIR,
    sample_rate: float | None = None,
    tracking_limit: int | None = None,
) -> dict:
    """
    Load all data for a match. Returns dict with:
      events_df     — tactical phase annotations (DataFrame)
      tracking_df   — per-frame positions (DataFrame)
      dataset       — kloppy TrackingDataset (for metadata/teams/players)
    """
    dataset = load_tracking_dataset(match, matches_dir, sample_rate, tracking_limit)
    return {
        "events_df": load_events(match, matches_dir),
        "tracking_df": dataset.to_df(),
        "dataset": dataset,
    }


if __name__ == "__main__":
    matches = list_matches()
    print(f"Found {len(matches)} matches:\n")
    for i, m in enumerate(matches):
        print(f"  [{i}] {m}")

    if matches:
        print(f"\nLoading match 0: {matches[0]}")
        events = load_events(0)
        ds = load_tracking_dataset(0, limit=10)
        print(f"  Events: {len(events)} rows, codes: {events['code'].nunique()}")
        print(f"  Frame rate: {ds.metadata.frame_rate}")
        print(f"  Pitch: {ds.metadata.pitch_dimensions.pitch_length}x{ds.metadata.pitch_dimensions.pitch_width}")
        for team in ds.metadata.teams:
            print(f"  Team: {team.name} ({len(team.players)} players)")
