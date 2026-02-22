# explainer.py
from __future__ import annotations

import json
import os
import re
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -------------------------
# Config
# -------------------------

CACHE_DIR = Path("cache/explanations")
DEFAULT_MODEL = "openai/gpt-4o-mini"

@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_sec: int = 60
    max_retries: int = 3
    retry_backoff_sec: float = 1.5

def _load_config() -> LLMConfig:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    return LLMConfig(api_key=api_key, model=model)

def _get_client(cfg: LLMConfig) -> OpenAI:
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

# -------------------------
# Output hard-constraints
# -------------------------

# Detect common timestamp patterns the model might output.
_TS_PATTERNS = [
    r"\b\d{1,3}:\d{2}\b",          # 12:34 or 102:15
    r"\b\d{1,3}'\b",               # 60'
    r"\b\d{1,3}\+\d{1,2}'\b",      # 45+2'
    r"\bminute\s+\d{1,3}\b",       # minute 62
    r"\bminutes?\s+\d{1,3}\b",     # minutes 62
]

def _strip_timestamps(text: str) -> str:
    out = text
    for pat in _TS_PATTERNS:
        out = re.sub(pat, "[time withheld]", out, flags=re.IGNORECASE)
    return out

def _remove_bullets_and_numbering(text: str) -> str:
    lines = []
    for line in text.splitlines():
        # remove leading bullets like "-", "*", "1.", "1)"
        line = re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", line)
        lines.append(line)
    return "\n".join(lines).strip()

def _keep_3_to_5_sentences(text: str) -> str:
    # Very lightweight sentence splitting; good enough for enforcement.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 5:
        return " ".join(parts)
    return " ".join(parts[:5])

def _postprocess_pattern(text: str) -> str:
    # Keep it readable and safe, but do NOT force 3–5 sentences.
    t = (text or "").strip()
    t = _remove_bullets_and_numbering(t)
    t = _strip_timestamps(t)
    # Soft limit: avoid giant essays, keep first ~10 sentences.
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 10:
        t = " ".join(parts[:10])
    return t.strip()

def _postprocess_moment(text: str) -> str:
    """
    Enforce:
    - 3–5 sentences
    - No bullets or numbering
    - No timestamps
    """
    t = (text or "").strip()
    t = _remove_bullets_and_numbering(t)
    t = _strip_timestamps(t)

    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) < 3:
        return " ".join(parts)
    if len(parts) > 5:
        parts = parts[:5]

    return " ".join(parts).strip()

# -------------------------
# Prompting
# -------------------------

SYSTEM_PROMPT_MOMENT = (
    "You are a football tactical analyst working with FC Barcelona's coaching staff.\n"
    "We are coaching FC Barcelona’s defensive phase.\n"
    "Treat Barcelona as the defending team in the danger window (do not flip teams).\n"
    "You are given event-code tags and (optionally) a tracking/spatial summary computed from tracking data.\n\n"
    "HARD RULES (must follow):\n"
    "- Do NOT mention or infer timestamps, minutes, or match clock.\n"
    "- Do NOT write bullet points or numbered lists.\n"
    "- Write EXACTLY 3–5 sentences.\n"
    "- Be tactically specific and actionable for coaches.\n"
    "- Do not invent details beyond the provided event codes and tracking evidence.\n"
    "- When referencing dynamics, cite the exact event tags in [BRACKETS] (e.g., [DEFENSIVE TRANSITION]).\n\n"
    "INTERPRETATION RULES (grounding):\n"
    "- Only claim a 'ball-side overload' if attackers > defenders within the stated radius.\n"
    "- If attackers==0 and defenders==0 within the radius, say there is 'no crowding near the ball within the radius'.\n"
    "- Only claim a 'numerical disadvantage' near the ball if attackers > defenders within the radius.\n"
    "- For nearest distance to ball (normalized units 0..1):\n"
    "  * <=0.05: very tight\n"
    "  * <=0.12: close\n"
    "  * <=0.25: moderate\n"
    "  * >0.25: far\n"
    "- If tracking_coverage_warning is true or either team has <6 tracked players, explicitly say tracking coverage is limited.\n"
)

SYSTEM_PROMPT_PATTERN = (
    "You are a football tactical analyst working with FC Barcelona's coaching staff.\n"
    "We are coaching FC Barcelona’s defensive phase across multiple matches.\n\n"
    "HARD RULES (must follow):\n"
    "- Do NOT mention or infer timestamps, minutes, or match clock.\n"
    "- Do NOT write bullet points or numbered lists.\n"
    "- Write as 1–2 short paragraphs.\n"
    "- Be conservative: do NOT claim something is recurring unless match_count >= 2.\n"
    "- Do not invent triggers beyond the provided event-code combo and match-level evidence.\n"
)



def build_moment_prompt(
    danger_moment: Dict[str, Any],
    match_name: str,
    opponent: str,
    tracking_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Builds an LLM prompt grounded in detector output.

    Supports both:
      - new schema: active_event_codes, peak:{time_s,score}, danger_window:{start_s,end_s}
      - old schema: active_codes, peak_score, start_s/end_s, etc.
    """

    # --- Resolve window times ---
    dw = danger_moment.get("danger_window") or {}
    start_s = (
        dw.get("start_s")
        if isinstance(dw, dict)
        else None
    )
    end_s = (
        dw.get("end_s")
        if isinstance(dw, dict)
        else None
    )

    # fallbacks
    if start_s is None:
        start_s = danger_moment.get("start_s") or danger_moment.get("window_start_s")
    if end_s is None:
        end_s = danger_moment.get("end_s") or danger_moment.get("window_end_s")

    # --- Resolve peak time + score ---
    peak = danger_moment.get("peak") or {}
    peak_t = (
        peak.get("time_s")
        if isinstance(peak, dict)
        else None
    )
    peak_score = (
        peak.get("score")
        if isinstance(peak, dict)
        else None
    )

    # fallbacks
    if peak_t is None:
        peak_t = danger_moment.get("peak_time_s") or danger_moment.get("peak_t")
    if peak_score is None:
        peak_score = danger_moment.get("peak_score") or danger_moment.get("score")

    # --- Resolve severity ---
    severity = danger_moment.get("severity") or danger_moment.get("risk_level") or "unknown"

    # --- Resolve active codes ---
    active_codes = (
        danger_moment.get("active_event_codes")
        or danger_moment.get("active_codes")
        or danger_moment.get("event_codes")
        or []
    )
    # normalize to list[str]
    if isinstance(active_codes, str):
        active_codes = [c.strip() for c in active_codes.split(",") if c.strip()]
    elif not isinstance(active_codes, list):
        active_codes = []

    # --- Resolve outcome / goal ---
    # Prefer an explicit outcome if present; otherwise infer something readable.
    outcome = danger_moment.get("outcome")
    if not outcome:
        resulted_in_goal = danger_moment.get("resulted_in_goal")
        if resulted_in_goal is True:
            outcome = "Goal conceded"
        elif resulted_in_goal is False:
            outcome = "No goal (threat significant)"
        else:
            # other possible fields
            outcome = danger_moment.get("danger_outcome") or "Unknown"

    # --- Formatting helpers (no timestamps desired in output, but prompt can include them as raw numbers)
    def _fmt_s(x: Any) -> str:
        try:
            if x is None:
                return "?"
            return f"{float(x):.2f}s"
        except Exception:
            return "?"

    def _fmt_score(x: Any) -> str:
        try:
            if x is None:
                return "?"
            return f"{float(x):.2f}"
        except Exception:
            return "?"

    codes_str = ", ".join(active_codes) if active_codes else "None"

    # --- Inject spatial summary as evidence (compact JSON) ---
    tracking_block = ""
    if tracking_summary:
        try:
            tracking_block = "\n\nTracking/Spatial summary (evidence):\n" + json.dumps(
                tracking_summary, ensure_ascii=False
            )
        except Exception:
            tracking_block = "\n\nTracking/Spatial summary (evidence):\n" + str(tracking_summary)

    prompt = (
        f"Match: {match_name}\n"
        f"Opponent: {opponent}\n"
        f"Danger window: {_fmt_s(start_s)} - {_fmt_s(end_s)} (peak at {_fmt_s(peak_t)})\n"
        f"Risk score at peak: { _fmt_score(peak_score) }/100 (severity: {severity})\n"
        f"Active event codes during peak: {codes_str}\n"
        f"Outcome: {outcome}\n"
        f"{tracking_block}\n\n"
        "Task:\n"
        "Explain what went wrong defensively in this passage of play. "
        "What tactical patterns led to this danger moment, and what should the coaching staff address?\n\n"
        "Constraints:\n"
        "- 3–5 sentences.\n"
        "- No timestamps (do not write times like 80:16 or 1:20).\n"
        "- No bullet points.\n"
        "- If you use bracketed tags like [DEFENSIVE TRANSITION], ONLY use tags that appear in 'Active event codes during peak'.\n"
        "- Interpretation rules: only claim a ball-side overload or numerical disadvantage near the ball if attackers > defenders within the given radius; if both are 0, say there is no crowding within the radius.\n"
        "- Distance language rules (nearest distance to ball): <=0.05 very tight, <=0.12 close, <=0.25 moderate, >0.25 far.\n"
        "- If tracking_coverage_warning is true or either team has <6 tracked players, explicitly say tracking coverage is limited.\n"
        "- Ground claims in the provided event codes and the tracking/spatial evidence (if present). "
        "If tracking evidence is missing or unclear, say so rather than guessing.\n"
    )

    return prompt

def build_window_prompt(
    events_in_window: list[dict],
    match_name: str,
    opponent: str,
    avg_risk: float,
) -> str:
    # Keep it code-driven; do not include any times.
    event_lines = []
    for e in events_in_window:
        code = e.get("code", "UNKNOWN")
        team = e.get("Team", "UNKNOWN")
        event_lines.append(f"- {code} (Team: {team})")

    events_str = "\n".join(event_lines) if event_lines else "(no events recorded)"

    return (
        f"Match: {match_name}\n"
        f"Opponent: {opponent}\n\n"
        f"5-minute window summary context:\n"
        f"- Average risk score: {avg_risk:.1f}/100\n"
        f"- Events active: \n{events_str}\n\n"
        f"Task:\n"
        f"Provide a tactical summary of the defensive dynamics in this window: "
        f"pressure vs control vs transition risk, and what the staff should adjust."
    )

def build_pattern_prompt(pattern: dict) -> str:
    pattern_name = (
        pattern.get("pattern_name")
        or pattern.get("name")
        or pattern.get("pattern")  # IMPORTANT: your schema uses this a lot
        or "Defensive pattern"
    )

    match_count = pattern.get("match_count")
    baseline_count = pattern.get("baseline_match_count")
    examples = pattern.get("examples") or []
    evidence = pattern.get("evidence") or pattern.get("supporting_evidence") or {}

    # Pull event codes from wherever they exist (depends on your formatter)
    event_codes = (
        pattern.get("event_codes")
        or pattern.get("active_event_codes")
        or (evidence.get("event_codes") if isinstance(evidence, dict) else None)
        or []
    )
    if isinstance(event_codes, str):
        event_codes = [c.strip() for c in event_codes.split(",") if c.strip()]
    if not isinstance(event_codes, list):
        event_codes = []

    # Make prompts unique even when match_count is 1 by anchoring on matches (names only, no times)
    ex_match_names = []
    for ex in examples[:6]:
        m = ex.get("match") or ex.get("match_name")
        if m:
            ex_match_names.append(str(m))
    ex_match_names = list(dict.fromkeys(ex_match_names))  # dedupe, keep order

    # Uniqueness anchor (helps avoid cache collisions if everything else is similar)
    anchor = {
        "pattern_name": pattern_name,
        "match_count": match_count,
        "event_codes": event_codes,
        "example_matches": ex_match_names[:3],
    }

    prompt = f"""
    You are writing for FC Barcelona coaching staff. This is a cross-match defensive pattern summary.
    Be conservative: do NOT call it recurring unless match_count >= 2.

    Formatting rules (very important):
    - Write as one or two short paragraphs.
    - Do NOT use bullet points, numbered lists, headers, timestamps, or timecodes.

    Pattern: {pattern_name}
    Support: match_count={match_count}, baseline_match_count={baseline_count}
    Event codes for THIS pattern: {", ".join(event_codes) if event_codes else "(missing event codes)"}

    Example matches (names only): {", ".join(ex_match_names[:6]) if ex_match_names else "(no examples)"}

    Additional evidence (if any):
    {json.dumps(evidence, ensure_ascii=False)[:2000]}

    Uniqueness anchor (do not mention this field in your output):
    {json.dumps(anchor, ensure_ascii=False)}

    Write:
    Explain the tactical mechanism behind THIS specific event-code combo, what usually triggers it, and one or two coaching adjustments.
    Explicitly reference at least TWO of the provided event codes by name in your explanation.
    If match_count is 0 or 1, explicitly say there is not enough evidence to call it recurring yet.
    """.strip()

    return prompt

# -------------------------
# Caching
# -------------------------

def _cache_key(prompt: str, system_prompt: Optional[str], model: str) -> str:
    raw = f"model={model}||sys={system_prompt or ''}||prompt={prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def _get_cached(key: str) -> Optional[str]:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("response")
    except Exception:
        return None

def _save_cache(key: str, prompt: str, response: str, model: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    data = {"model": model, "prompt": prompt, "response": response}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# -------------------------
# LLM calls
# -------------------------

def call_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    cfg = _load_config()
    client = _get_client(cfg)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_err: Optional[Exception] = None
    for attempt in range(cfg.max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                timeout=cfg.timeout_sec,
            )
            text = resp.choices[0].message.content or ""
            return text
        except Exception as e:
            last_err = e
            if attempt < cfg.max_retries - 1:
                time.sleep(cfg.retry_backoff_sec * (attempt + 1))
            else:
                raise

    raise RuntimeError(f"LLM call failed: {last_err}")

def call_llm_cached(
    prompt: str,
    system_prompt: Optional[str] = None,
    *,
    postprocess_fn=None,
) -> str:
    cfg = _load_config()
    key = _cache_key(prompt, system_prompt, cfg.model)

    cached = _get_cached(key)
    if cached is not None:
        return cached

    try:
        raw = call_llm(prompt, system_prompt)
        if postprocess_fn is None:
            cleaned = _postprocess_llm(raw)
        else:
            cleaned = postprocess_fn(raw)
    except Exception as e:
        cleaned = f"[Explanation unavailable: {e}]"

    _save_cache(key, prompt, cleaned, cfg.model)
    return cleaned

# -------------------------
# Public API
# -------------------------

def explain_moment(
    danger_moment: dict,
    match_name: str,
    opponent: str,
    tracking_summary: dict | None = None,
) -> str:
    prompt = build_moment_prompt(
        danger_moment=danger_moment,
        match_name=match_name,
        opponent=opponent,
        tracking_summary=tracking_summary,
    )
    return call_llm_cached(
        prompt,
        system_prompt=SYSTEM_PROMPT_MOMENT,
        postprocess_fn=_postprocess_moment,
    )

def explain_window(
    events_in_window: list[dict],
    match_name: str,
    opponent: str,
    avg_risk: float,
) -> str:
    prompt = build_window_prompt(events_in_window, match_name, opponent, avg_risk)
    return call_llm_cached(prompt, system_prompt=SYSTEM_PROMPT_MOMENT)

def explain_pattern(pattern: dict) -> str:
    prompt = build_pattern_prompt(pattern)
    return call_llm_cached(prompt, system_prompt=SYSTEM_PROMPT_PATTERN, postprocess_fn=_postprocess_pattern)

if __name__ == "__main__":
    cfg = _load_config()
    print(f"Model: {cfg.model}")
    print(f"API key set: {bool(cfg.api_key)}")

    result = call_llm_cached(
        prompt="Explain how a team should organize rest-defense after losing the ball.",
        system_prompt=SYSTEM_PROMPT,
    )
    print(result)