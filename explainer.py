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
    - Preserve three-part structure (Context, Defensive Error, Coach Note)
    - No timestamps
    - Clean up excess whitespace
    """
    t = (text or "").strip()
    t = _strip_timestamps(t)

    # Normalize whitespace but preserve the structure
    lines = [line.strip() for line in t.split('\n')]
    lines = [line for line in lines if line]  # Remove empty lines

    # Rejoin with double newlines between sections for readability
    result = []
    for line in lines:
        if line.startswith(('Context:', 'Defensive Error:', 'Coach Note:')):
            if result:  # Add spacing before new section (except first)
                result.append('')
            result.append(line)
        else:
            # Continuation of previous section
            if result:
                result[-1] += ' ' + line
            else:
                result.append(line)

    return '\n'.join(result).strip()

# -------------------------
# Prompting
# -------------------------

SYSTEM_PROMPT_MOMENT = (
    "You are a football tactical analyst working with FC Barcelona's coaching staff.\n"
    "We are coaching FC Barcelona's defensive phase.\n"
    "Treat Barcelona as the defending team in the danger window (do not flip teams).\n"
    "You are given event-code tags and (optionally) a tracking/spatial summary computed from tracking data.\n\n"
    "OUTPUT FORMAT (REQUIRED):\n"
    "You must structure your response in exactly three sections:\n"
    "Context: [3-5 sentences providing detailed description of what happened in this event]\n\n"
    "Defensive Error: [3-5 sentences with in-depth analysis of what went wrong with the defending]\n\n"
    "Coach Note: [3-5 sentences with specific, actionable tactical suggestions on what to fix]\n\n"
    "HARD RULES (must follow):\n"
    "- Do NOT mention or infer timestamps, minutes, or match clock.\n"
    "- Do NOT write bullet points or numbered lists.\n"
    "- NEVER mention specific unit measurements (e.g., '0.04 units') - use qualitative terms only.\n"
    "- NEVER use bracketed tags like [DEFENSIVE TRANSITION] - write naturally.\n"
    "- NEVER reference 'event codes' or mention the data sources - explain as if you witnessed the play.\n"
    "- Use common sense - if tracking shows 10v11, it's a LOCAL overload near the ball, NOT the whole team.\n"
    "- Be REALISTIC - focus on tactically relevant insights, not literal data interpretation.\n"
    "- When tactically relevant, mention player numbers (e.g., 'the number 5 lost his marker', 'number 9 exploited space').\n"
    "- Only reference player numbers when it adds value to the tactical explanation - don't force it.\n"
    "- Each section should be 3-5 sentences with practical tactical analysis.\n"
    "- Write as a coach explaining to another coach - natural, fluent, tactical language.\n\n"
    "INTERPRETATION RULES (grounding):\n"
    "- Numerical situations: ALWAYS describe overloads and transitions using tactical numbers (e.g., '4 on 2', '3 v 1', '5 on 3').\n"
    "- Use 'on' or 'v' format: '4 on 2 situation', '3 v 1 disadvantage', '2 on 1 break'.\n"
    "- Ball-side overload: Focus on LOCAL numerical advantage near the ball (typically realistic numbers like 2v1, 3v2, 4v2).\n"
    "- Defensive compactness: Describe team shape and spacing in tactical terms.\n"
    "- Distance to ball: Use qualitative terms ONLY (tight pressure, close, moderate distance, far).\n"
    "- Player counts: If tracking shows unusual numbers (>8 defenders or attackers), it's likely a data artifact - use realistic tactical numbers instead.\n"
    "- If tracking_coverage_warning is true, mention limited tracking data and rely more on event codes.\n"
    "- Prioritize actionable tactical insights over technical data details.\n"
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
        "Provide an IN-DEPTH tactical analysis of this danger moment using the three-part structure:\n"
        "1. Context: Describe what happened in this passage of play with specific details from the event codes.\n"
        "2. Defensive Error: Explain precisely what went wrong with Barcelona's defending - be specific about positioning, pressure, transitions, spacing, etc.\n"
        "3. Coach Note: Provide detailed, actionable tactical adjustments the coaching staff should implement.\n\n"
        "Constraints:\n"
        "- MUST use the format: 'Context: ...\\n\\nDefensive Error: ...\\n\\nCoach Note: ...'\n"
        "- Each section: 3-5 sentences with PRACTICAL, ACTIONABLE analysis.\n"
        "- Write NATURALLY - as if explaining to a coach based on what you observed.\n"
        "- NEVER use bracketed event tags - explain in plain tactical language.\n"
        "- NEVER say 'based on event codes' or reference data sources - just explain what happened.\n"
        "- NEVER mention unit measurements - describe distances qualitatively (tight, close, moderate, far).\n"
        "- When tactically relevant, reference player numbers (e.g., 'the number 5', 'their number 9') - but only when it adds tactical value.\n"
        "- No timestamps (do not write times like 80:16 or 1:20).\n"
        "- No bullet points or numbered lists.\n"
        "- Ball-side overload: Describe LOCAL advantage near the ball (2v1, 3v2, etc.), not whole-team counts.\n"
        "- Be realistic and coach-friendly - focus on what matters tactically.\n"
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