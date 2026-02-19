import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# key
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# choose model
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

def _get_client() -> OpenAI:
    api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def call_llm(prompt: str, system_prompt: str | None = None) -> str:
    client = _get_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    model = OPENROUTER_MODEL or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    response = client.chat.completions.create(model=model, messages=messages)

    # response object has a `choices` list, each choice has a `message` (this is the LLM output)
    return response.choices[0].message.content

CACHE_DIR = Path("cache/explanations")

def _cache_key(prompt: str, system_prompt: str | None) -> str:
    raw = (system_prompt or "") + "||" + prompt
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _get_cached(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["response"]
    return None

def _save_cache(key: str, prompt: str, response: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    data = {"prompt": prompt, "response": response}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def call_llm_cached(prompt: str, system_prompt: str | None = None) -> str:
    key = _cache_key(prompt, system_prompt)
    cached = _get_cached(key)
    if cached is not None:
        return cached
    response = call_llm(prompt, system_prompt)
    _save_cache(key, prompt, response)
    return response

SYSTEM_PROMPT = (
    "You are a football tactical analyst working with FC Barcelona's coaching staff. "
    "You analyze defensive vulnerabilities using event timeline data from matches. "
    "Your explanations should be concise (3-5 sentences), tactically specific, and "
    "actionable for coaches. Use professional football terminology. "
    "Do not speculate beyond what the data shows. "
    "When referencing events, cite the exact event code tags in [BRACKETS] "
    "(e.g. [BALL IN THE BOX], [DEFENSIVE TRANSITION]) so we can trace your analysis back to the data."
)

def _fmt(seconds: int | float, halftime_offset: int = 0, h2_start_sec: int = 0) -> str:
    """Convert raw XML seconds to match-time MM:SS.
    For second-half timestamps (>= h2_start_sec), subtracts halftime_offset."""
    s = int(seconds)
    if halftime_offset and h2_start_sec and s >= h2_start_sec:
        s -= halftime_offset
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}"

def build_moment_prompt(danger_moment: dict, match_name: str, opponent: str,
                        halftime_offset: int = 0, h2_start_sec: int = 0) -> str:
    fmt = lambda secs: _fmt(secs, halftime_offset, h2_start_sec)
    peak = fmt(danger_moment["peak_time"])
    w_start = fmt(danger_moment["window_start"])
    w_end = fmt(danger_moment["window_end"])
    codes = ", ".join(danger_moment["active_codes"])
    score = danger_moment["peak_score"]
    severity = danger_moment["severity"]
    goal = danger_moment["resulted_in_goal"]

    outcome = "This resulted in a GOAL conceded." if goal else "No goal was scored, but the threat was significant."

    return (
        f"Match: {match_name}\n"
        f"Opponent: {opponent}\n"
        f"Danger window: {w_start} - {w_end} (peak at {peak})\n"
        f"Risk score at peak: {score}/100 (severity: {severity})\n"
        f"Active event codes during peak: {codes}\n"
        f"{outcome}\n\n"
        f"Explain what went wrong defensively in this passage of play. "
        f"What tactical patterns led to this danger moment, and what should "
        f"the coaching staff address?"
    )

def build_window_prompt(events_in_window: list[dict], window_start: int, window_end: int,
                        match_name: str, opponent: str, avg_risk: float,
                        halftime_offset: int = 0, h2_start_sec: int = 0) -> str:
    fmt = lambda secs: _fmt(secs, halftime_offset, h2_start_sec)
    w_start = fmt(window_start)
    w_end = fmt(window_end)

    event_lines = []
    for e in events_in_window:
        t0 = fmt(e["start_sec"])
        t1 = fmt(e["end_sec"])
        event_lines.append(f"  - [{t0}-{t1}] {e['code']} (Team: {e['Team']})")

    events_str = "\n".join(event_lines) if event_lines else "  (no events in this window)"

    return (
        f"Match: {match_name}\n"
        f"Opponent: {opponent}\n"
        f"Window: {w_start} - {w_end}\n"
        f"Average risk score: {avg_risk:.1f}/100\n\n"
        f"Events active during this window:\n{events_str}\n\n"
        f"Provide a tactical summary of this 5-minute passage of play. "
        f"Focus on the defensive dynamics - was Barca under pressure, in control, "
        f"or transitioning? What does this window tell the coaching staff?"
    )

def build_pattern_prompt(pattern: dict) -> str:
    seq = " -> ".join(pattern["sequence"])
    freq = pattern.get("frequency", "unknown")
    examples = ", ".join(pattern.get("example_matches", []))
    time_to_goal = pattern.get("avg_time_to_goal")

    # Confidence gating (backend signal; not necessarily shown to coaches)
    conf = pattern.get("confidence_score", None)
    tier = pattern.get("confidence_tier", None)
    threshold = 0.60  # tweakable: "confident enough to call recurring"

    # Decide wording: only call it "recurring" if we have enough confidence
    if isinstance(conf, (int, float)) and conf >= threshold:
        recurrence_line = (
            "This pattern is a **recurring** vulnerability across matches "
            "(supported by the cross-match analysis)."
        )
    else:
        # Avoid overstating recurrence
        recurrence_line = (
            "This pattern appears in the data, but cross-match evidence is **not strong enough** "
            "to confidently label it recurring yet. Treat it as a candidate theme to monitor."
        )

    # Optional internal metadata line (useful for logs / dev, can remove anytime)
    conf_line = ""
    if isinstance(conf, (int, float)):
        conf_line = f"\nConfidence (internal): {conf:.2f}" + (f" ({tier})" if tier else "")

    time_line = ""
    if time_to_goal is not None:
        time_line = f"\nAverage time from this pattern to goal conceded: {time_to_goal:.0f} seconds"

    return (
        f"Defensive vulnerability pattern detected:\n\n"
        f"Event sequence: {seq}\n"
        f"Frequency: appeared in {freq}\n"
        f"Example matches: {examples}"
        f"{time_line}\n\n"
        f"{recurrence_line}"
        f"Explain why this sequence represents a tactical vulnerability. "
        f"What structural or tactical issue does it reveal, and what adjustments "
        f"would you recommend to the coaching staff?"
    )

def explain_moment(danger_moment: dict, match_name: str, opponent: str,
                   halftime_offset: int = 0, h2_start_sec: int = 0) -> str:
    prompt = build_moment_prompt(danger_moment, match_name, opponent, halftime_offset, h2_start_sec)
    return call_llm_cached(prompt, system_prompt=SYSTEM_PROMPT)

def explain_window(events_in_window: list[dict], window_start: int,
                   window_end: int, match_name: str, opponent: str,
                   avg_risk: float, halftime_offset: int = 0, h2_start_sec: int = 0) -> str:
    prompt = build_window_prompt(events_in_window, window_start, window_end,
                                 match_name, opponent, avg_risk, halftime_offset, h2_start_sec)
    return call_llm_cached(prompt, system_prompt=SYSTEM_PROMPT)

def explain_pattern(pattern: dict) -> str:
    prompt = build_pattern_prompt(pattern)
    return call_llm_cached(prompt, system_prompt=SYSTEM_PROMPT)

if __name__ == "__main__":
    print(f"Model:  {OPENROUTER_MODEL}")
    print(f"API key set: {bool(OPENROUTER_API_KEY)}")
    print()

    try:
        result = call_llm(
            prompt="In one sentence, what is a defensive transition in football?",
            system_prompt="You are a football tactical analyst. Be concise.",
        )
        print(f"Response: {result}")
    except ValueError as e:
        print(f"Setup error: {e}")
    except Exception as e:
        print(f"API error: {e}")
