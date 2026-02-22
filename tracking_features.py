# tracking_features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
import json


@dataclass(frozen=True)
class TrackingCols:
    time_col: str = "time_s"
    team_col: str = "team_id"
    player_col: str = "player_id"
    x_col: str = "x"
    y_col: str = "y"
    ball_x_col: str = "ball_x"
    ball_y_col: str = "ball_y"


DEFAULT_COLS = TrackingCols()


def _slice_window(df: pd.DataFrame, t0: float, t1: float, time_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()


def _team_frame(players_t: pd.DataFrame, team_value: str, team_col: str) -> pd.DataFrame:
    if players_t is None or players_t.empty:
        return players_t
    # team_id might be numeric or string or NaN
    s = players_t[team_col].astype(str)
    return players_t[s == str(team_value)].copy()


def _shape_metrics(team_df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, float]:
    xs = pd.to_numeric(team_df[x_col], errors="coerce").to_numpy()
    ys = pd.to_numeric(team_df[y_col], errors="coerce").to_numpy()
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    if len(xs) == 0 or len(ys) == 0:
        return {}

    width = float(np.nanpercentile(xs, 95) - np.nanpercentile(xs, 5))
    length = float(np.nanpercentile(ys, 95) - np.nanpercentile(ys, 5))
    centroid_x = float(np.nanmean(xs))
    centroid_y = float(np.nanmean(ys))

    return {
        "team_width": width,
        "team_length": length,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
    }


def _nearest_distance(team_df: pd.DataFrame, point_xy: Tuple[float, float], x_col: str, y_col: str) -> Optional[float]:
    if team_df is None or team_df.empty:
        return None
    px, py = point_xy
    xs = pd.to_numeric(team_df[x_col], errors="coerce").to_numpy()
    ys = pd.to_numeric(team_df[y_col], errors="coerce").to_numpy()
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if len(xs) == 0:
        return None
    d = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
    if d.size == 0 or np.all(np.isnan(d)):
        return None
    return float(np.nanmin(d))


def _median_ball_xy(ball_w: pd.DataFrame, cols: TrackingCols) -> Tuple[float, float]:
    if ball_w is None or ball_w.empty:
        return (float("nan"), float("nan"))
    bx = pd.to_numeric(ball_w[cols.ball_x_col], errors="coerce").median()
    by = pd.to_numeric(ball_w[cols.ball_y_col], errors="coerce").median()
    return (float(bx), float(by))


def summarize_window(
    players_df: pd.DataFrame,
    ball_df: Optional[pd.DataFrame],
    t0: float,
    t1: float,
    *,
    defending_team_id: Optional[str] = None,
    attacking_team_id: Optional[str] = None,
    preferred_time_s: Optional[float] = None,
    cols: TrackingCols = DEFAULT_COLS,
    overload_radius: float = 0.08,
    normalize_direction: bool = False,
    halftime_s: Optional[float] = None,
    assume_normalized_xy: bool = True,
) -> Dict[str, Any]:
    """
    Returns a compact tracking summary you can safely show the LLM.

    Snapshot behavior (IMPORTANT):
    - Choose ONE snapshot timestamp:
        1) preferred_time_s if provided and finite (e.g., danger peak)
        2) else midpoint of [t0, t1]
    - Select *all* rows at the closest available timestamp in the window (no tolerance slicing),
      so we don’t accidentally drop most players.
    - Reduce to one row per (team_id, player_id).

    Diagnostics:
    - Adds tracking_coverage_warning + team counts when team filtering yields too few players.
    """
    if players_df is None or players_df.empty:
        return {"window": {"start_s": float(t0), "end_s": float(t1)}, "error": "players_df empty"}

    tc = cols.time_col

    # Basic schema guard
    for c in (tc, cols.team_col, cols.player_col, cols.x_col, cols.y_col):
        if c not in players_df.columns:
            return {"window": {"start_s": float(t0), "end_s": float(t1)}, "error": f"missing required column: {c}"}

    # Slice raw window
    pw_raw = _slice_window(players_df, t0, t1, tc)
    bw_raw = _slice_window(ball_df, t0, t1, tc) if ball_df is not None else None

    # Choose snapshot time
    t_mid = float((t0 + t1) / 2.0)
    try:
        t_pref = float(preferred_time_s) if preferred_time_s is not None else None
    except Exception:
        t_pref = None
    t_target = t_pref if (t_pref is not None and np.isfinite(t_pref)) else t_mid

    out: Dict[str, Any] = {
        "window": {"start_s": float(t0), "end_s": float(t1), "target_s": float(t_target)},
        "direction_normalized": False,
    }

    if pw_raw is None or pw_raw.empty:
        out["error"] = "no player samples in window"
        ball_xy = _median_ball_xy(bw_raw, cols) if bw_raw is not None else (float("nan"), float("nan"))
        out["ball_median_xy"] = {"x": ball_xy[0], "y": ball_xy[1]}
        out["tracking_coverage_warning"] = True
        return out

    pw = pw_raw.copy()
    pw[tc] = pd.to_numeric(pw[tc], errors="coerce")
    pw = pw.dropna(subset=[tc])

    if pw.empty:
        out["error"] = "no numeric time_s in window"
        out["tracking_coverage_warning"] = True
        return out

    # --- Snapshot selection: pick the closest *timestamp value* and take all rows at that timestamp ---
    # Using unique timestamps prevents frame-to-frame drift and avoids tolerance slices that can drop players.
    t_vals = pw[tc].to_numpy(dtype=float)
    idx = int(np.nanargmin(np.abs(t_vals - float(t_target))))
    t_snap = float(t_vals[idx])

    pw = pw[pw[tc] == t_snap].copy()
    out["window"]["snap_s"] = float(t_snap)

    # Reduce to one row per (team, player)
    pw[cols.x_col] = pd.to_numeric(pw[cols.x_col], errors="coerce")
    pw[cols.y_col] = pd.to_numeric(pw[cols.y_col], errors="coerce")

    pw = (
        pw.dropna(subset=[cols.team_col, cols.player_col, cols.x_col, cols.y_col])
          .astype({cols.team_col: "string", cols.player_col: "string"})
          .groupby([cols.team_col, cols.player_col], as_index=False)[[cols.x_col, cols.y_col]].median()
    )

    out["debug_team_ids_in_snapshot"] = pw[cols.team_col].astype(str).value_counts().head(5).to_dict()
    out["debug_expected_team_ids"] = {"defending_team_id": defending_team_id, "attacking_team_id": attacking_team_id}

    # --- Optional direction normalization ---
    did_normalize = False
    ht = halftime_s
    if normalize_direction:
        if ht is None:
            ht = infer_halftime_s(players_df, cols=cols)

        # Decide half based on snapshot time
        if ht is not None and t_snap >= float(ht) and assume_normalized_xy:
            pw = pw.copy()
            pw = _flip_x_inplace(pw, cols.x_col)

            if bw_raw is not None and not bw_raw.empty:
                bw_raw = bw_raw.copy()
                bw_raw = _flip_x_inplace(bw_raw, cols.ball_x_col)

            did_normalize = True

        out["halftime_s_used"] = float(ht) if ht is not None else None
        out["assume_normalized_xy"] = bool(assume_normalized_xy)

    out["direction_normalized"] = bool(did_normalize)

    # Ball median (within full window slice; OK because ball is single object)
    ball_xy = _median_ball_xy(bw_raw, cols) if bw_raw is not None else (float("nan"), float("nan"))
    out["ball_median_xy"] = {"x": ball_xy[0], "y": ball_xy[1]}

    # --- Team filtering ---
    def_team = _team_frame(pw, defending_team_id, cols.team_col) if defending_team_id is not None else pd.DataFrame()
    att_team = _team_frame(pw, attacking_team_id, cols.team_col) if attacking_team_id is not None else pd.DataFrame()

    # Coverage diagnostics (don’t block features, just warn)
    n_total = int(len(pw))
    n_def = int(len(def_team)) if defending_team_id is not None else None
    n_att = int(len(att_team)) if attacking_team_id is not None else None

    # Heuristic thresholds: in a snapshot we expect ~8–11 per team, but tolerate lower in some windows.
    warn = False
    if defending_team_id is not None and (n_def is None or n_def < 6):
        warn = True
    if attacking_team_id is not None and (n_att is None or n_att < 6):
        warn = True
    if n_total < 12:
        warn = True

    out["tracking_counts"] = {
        "n_total_players": n_total,
        "n_def_players": n_def,
        "n_att_players": n_att,
    }
    out["tracking_coverage_warning"] = bool(warn)

    if defending_team_id is not None:
        out["def_team"] = {
            "team_id": str(defending_team_id),
            "shape": _shape_metrics(def_team, cols.x_col, cols.y_col),
            "n_players": int(len(def_team)),
        }
        out["nearest_defender_to_ball"] = _nearest_distance(def_team, ball_xy, cols.x_col, cols.y_col)

    if attacking_team_id is not None:
        out["att_team"] = {
            "team_id": str(attacking_team_id),
            "shape": _shape_metrics(att_team, cols.x_col, cols.y_col),
            "n_players": int(len(att_team)),
        }
        out["nearest_attacker_to_ball"] = _nearest_distance(att_team, ball_xy, cols.x_col, cols.y_col)

    # --- Overload near ball (sanity-guarded) ---
    if (
        np.isfinite(ball_xy[0]) and np.isfinite(ball_xy[1])
        and defending_team_id is not None and attacking_team_id is not None
        and not def_team.empty and not att_team.empty
    ):
        def_dx = pd.to_numeric(def_team[cols.x_col], errors="coerce") - ball_xy[0]
        def_dy = pd.to_numeric(def_team[cols.y_col], errors="coerce") - ball_xy[1]
        att_dx = pd.to_numeric(att_team[cols.x_col], errors="coerce") - ball_xy[0]
        att_dy = pd.to_numeric(att_team[cols.y_col], errors="coerce") - ball_xy[1]

        def_dist = np.sqrt(def_dx * def_dx + def_dy * def_dy)
        att_dist = np.sqrt(att_dx * att_dx + att_dy * att_dy)

        defenders = int((def_dist <= overload_radius).sum())
        attackers = int((att_dist <= overload_radius).sum())

        # hard cap: should never exceed players present
        defenders = min(defenders, int(len(def_team)))
        attackers = min(attackers, int(len(att_team)))

        out["ball_side_overload"] = {
            "radius": float(overload_radius),
            "defenders": defenders,
            "attackers": attackers,
            "overload": bool(attackers > defenders),
        }

    return out

def load_team_map(parsed_match_dir):
    p = Path(parsed_match_dir) / "team_map.json"
    print(f"[TEAM MAP] expected_path={p} exists={p.exists()}")

    # If the exact filename isn't found, try any *team_map*.json in the folder
    if not p.exists():
        candidates = sorted(Path(parsed_match_dir).glob("*team_map*.json"))
        print(f"[TEAM MAP] candidates={candidates}")
        if candidates:
            p = candidates[0]
            print(f"[TEAM MAP] using_candidate={p}")

    try:
        if not p.exists():
            return {}
        txt = p.read_text(encoding="utf-8-sig")  # handles BOM if present
        data = json.loads(txt)
        print(f"[TEAM MAP] loaded keys={list(data.keys())}")
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[TEAM MAP] failed to load: {e}")
        return {}