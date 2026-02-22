# Pressure Cooker

FC Barcelona defensive analysis for the **More than a Hack 2026** hackathon.

Processes 11 matches of Metrica Sports Smart Tagging + tracking data to find when and how Barcelona's defence breaks down. Outputs a continuous risk score, flags danger moments, detects recurring vulnerability patterns across matches, and explains each one using an LLM.

---

## Setup

### 1. Data

Download the match data from the shared drive (`dataset_only_tracking_and_events`). Only download the **masculi** folder — skip the `.pl3container` files (those are only for viewing in Metrica Nexus).

Rename the folder to `matches` and place it in the repo root:

```
matches/
  AC Milan - Barça (0-1) .../
    *_pattern.xml
    *_FifaData.xml
    *_FifaDataRawData.txt
    *.mp4
  Arsenal - Barça (5-3) .../
  ...
```

### 2. Python environment

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `scipy`, `kloppy`, `lxml`, `openai`, `python-dotenv`, `fastapi`, `uvicorn`.

### 3. Environment variables

Copy `_env` to `.env` and fill in your key:

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

---

## Project structure

```
├── data_loader.py              # Loads events from *_pattern.xml per match
├── risk_engine.py              # Converts events → continuous 0–100 risk score
├── danger_detector.py          # Finds danger windows where risk exceeds threshold
├── pattern_analyzer.py         # Cross-match pattern detection (signatures + lift)
├── explainer.py                # LLM prompt building, caching, post-processing
├── generate_llm_insights.py    # Batch run: all matches → danger explanations + patterns
│
├── tracking_parser.py          # Single-match tracking parser (early version)
├── tracking_batch_parser.py    # Batch parser: raw ATD → player_positions.csv + ball_positions.csv
├── tracking_features.py        # Spatial features: team shape, overload, ball proximity
│
├── api.py                      # FastAPI backend for the dashboard
├── dashboard/                  # React frontend (Vite)
│
├── data_exploration.ipynb      # Explore one match interactively
├── project_overview.ipynb      # Full pipeline walkthrough with visualizations
│
├── tune_danger_detector_all_matches.py   # Run detector across all matches, print stats
├── gridsearch_danger_detector.py         # Grid search over detector parameters
├── test_danger_detector_quick.py         # Synthetic smoke test
├── test_one_match_danger_detector.py     # Single-match detector test + plot
├── test_pattern_analyzer.py              # Pattern analyzer test
│
├── matches/                    # Match data (gitignored)
├── parsed/                     # Parsed tracking CSVs (gitignored, created by tracking_batch_parser)
├── cache/                      # LLM response cache (gitignored)
└── outputs/                    # Generated insights JSON (gitignored)
```

---

## How it works

### Risk scoring (`risk_engine.py`)

Each match's Smart Tagging events are laid onto a 0.25s time grid. Opponent attacking events (e.g. BALL IN THE BOX: +1.55, DEFENSIVE TRANSITION: +1.35) add risk; Barcelona possession events (e.g. POSSESSION: −0.35) subtract it. The raw signal is smoothed with a 3-second moving average and min-max scaled to 0–100.

### Danger detection (`danger_detector.py`)

Continuous segments where risk > 45 are flagged as danger windows. Windows shorter than 5 seconds are discarded. Windows separated by < 12 seconds are merged into one sustained spell. Each window gets a severity label based on peak score (High ≥ 80, Moderate ≥ 50, Low ≥ 25).

### Pattern analysis (`pattern_analyzer.py`)

Each danger moment has a signature: the sorted set of active event codes at peak time. Signatures are compared across matches — patterns that appear in 2+ matches get a lift score (prevalence in goal moments vs. all danger moments) and a composite confidence score.

### Tracking features (`tracking_features.py`)

For each danger moment, a spatial snapshot is extracted from parsed tracking data: team shape (width/length/centroid), nearest defender/attacker to ball, ball-side overload count, and a coverage warning when < 6 players per team are tracked.

### LLM explanations (`explainer.py`)

Each danger moment is packed into a structured prompt (event codes, risk score, severity, spatial summary) and sent to GPT-4o-mini via OpenRouter. The system prompt enforces 3–5 sentences, no timestamps, bracketed code citations, and spatial interpretation rules. Outputs are post-processed (timestamp stripping, length capping) and cached by SHA-256 hash.

---

## Running the pipeline

### Explore a single match

```bash
jupyter notebook data_exploration.ipynb
```

Change `MATCH = 5` to pick a different match (0–10).

### Parse tracking data (all matches)

```bash
python tracking_batch_parser.py --matches_dir matches --out_dir parsed
```

Creates `parsed/<match_name>/player_positions.csv`, `ball_positions.csv`, and `team_map.json` for each match.

### Generate LLM insights (all matches)

```bash
python generate_llm_insights.py
```

Outputs `outputs/llm_insights/danger_moment_explanations.json` and `pattern_explanations.json`.

### Run the dashboard

```bash
# Backend
uvicorn api:app --reload --port 8000

# Frontend (in another terminal)
cd dashboard
npm install
npm run dev
```

Open `http://localhost:5173`. Pick a match, explore the risk timeline, click danger moments to see LLM explanations, and seek to the corresponding video timestamp.

### Run tests

```bash
python test_danger_detector_quick.py     # Synthetic smoke test
python test_one_match_danger_detector.py  # Single match + plot
python test_pattern_analyzer.py           # Cross-match patterns
```

### Tune detector parameters

```bash
python tune_danger_detector_all_matches.py   # Current config across all matches
python gridsearch_danger_detector.py          # Grid search for optimal params
```

---

## Output summary (11 matches)

| Metric | Value |
|---|---|
| Danger moments detected | 144 |
| Goal coverage | 17/17 (100%) |
| Recurring patterns found | 3 |
| Top pattern lift | 2.66× (attacking → defensive transition) |
