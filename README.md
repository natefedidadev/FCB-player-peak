# FCB Pressure Cooker

**Defensive analytics system for FC Barcelona matches**

A comprehensive tool that analyzes defensive vulnerabilities, detects danger moments, and generates AI-powered tactical insights from match data. Built for the F.C. Barcelona "More than a Hack" competition.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![React](https://img.shields.io/badge/React-19-blue.svg)

---

## What It Does

1. **Computes continuous Defensive Risk Score** (0-100) across match timeline
2. **Detects danger moments** — peaks in defensive risk indicating vulnerabilities
3. **Finds recurring patterns** — common defensive weaknesses across matches
4. **Generates tactical explanations** — AI-powered coach-friendly analysis
5. **Interactive dashboard** — visualize risk, danger moments, and video sync

**Example Output**: "The opponent created a 4 on 2 overload in the final third after winning the ball in transition. The number 5 lost his marker, creating space for the through ball..."

---

## Key Features

- Real-time risk scoring using weighted event analysis
- Interactive timeline visualization with clickable danger moments
- Video synchronization — jump to exact moments in match footage
- AI-powered tactical insights combining event + spatial tracking data
- Pattern detection across multiple matches
- Smart caching for performance

---

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Metrica Nexus format match data
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Installation

```bash
# Clone and setup Python environment
git clone https://github.com/yourusername/FCB-Pressure-Cooker.git
cd FCB-Pressure-Cooker
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Setup frontend
cd dashboard
npm install
cd ..
```

### Configuration

Create `.env` file in project root:

```bash
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

### Data Setup

Place match data in `matches/` folder:

```
matches/
├── Match-Name-1/
│   ├── Match-Name-1_pattern.xml        (REQUIRED)
│   ├── Match-Name-1_FifaData.xml       (REQUIRED)
│   ├── Match-Name-1_FifaDataRawData.txt  (optional - for spatial analysis)
│   └── Match-Name-1.mp4                (optional - for video playback)
```

**Note**: Requires Metrica Nexus XML format. If you participated in the competition, use the provided dataset.

### Running

```bash
# Parse tracking data (optional, if you have .txt files)
python tracking_batch_parser.py

# Start backend (in one terminal)
uvicorn api:app --reload

# Start frontend (in another terminal)
cd dashboard
npm run dev

# Open http://localhost:5173
```

---

## How It Works

### Pipeline

```
Raw Match Data (XML)
    ↓
1. Data Loading — Parse XML event timeline
    ↓
2. Risk Scoring — Apply weighted risk model (0-100)
    ↓
3. Danger Detection — Find peaks and classify severity
    ↓
4. Tracking Analysis — Compute spatial metrics (optional)
    ↓
5. LLM Explanation — Generate tactical insights
    ↓
6. Dashboard — Interactive visualization
```

### Risk Scoring Model

Manually tuned weights based on tactical significance:

**Opponent Events**: GOALS (+10), BALL IN THE BOX (+8), CREATING CHANCES (+7), BALL IN FINAL THIRD (+5), ATTACKING TRANSITION (+4), PROGRESSION (+3), SET PIECES (+3)

**Barcelona Defensive Events**: DEFENDING IN DEFENSIVE THIRD (+4), DEFENSIVE TRANSITION (+3), DEFENDING IN MIDDLE THIRD (+2), DEFENDING IN ATTACKING THIRD (+1)

Final score is smoothed (15s rolling average) and normalized (0-100).

**Why not ML?** With only 17 goals across 11 friendly matches, ML models (logistic regression, random forest) performed at or below random chance (ROC-AUC 0.44-0.55). Manual weights outperform with this dataset size.

---

## Project Structure

```
FCB-Pressure-Cooker/
├── matches/                     # Match data (XML/TXT/MP4)
├── parsed/                      # Auto-generated tracking CSVs
├── cache/                       # LLM explanation cache
├── dashboard/                   # React frontend
│   ├── src/components/
│   │   ├── RiskTimeline.jsx
│   │   └── DangerList.jsx
│   └── ...
├── api.py                       # FastAPI backend
├── data_loader.py               # XML parsing
├── risk_engine.py               # Risk score computation
├── danger_detector.py           # Danger moment detection
├── pattern_analyzer.py          # Cross-match patterns
├── explainer.py                 # LLM explanations
├── tracking_features.py         # Spatial analysis
├── tracking_batch_parser.py     # Parse tracking data
├── requirements.txt
├── .env
└── README.md
```

---

## Using the Dashboard

1. **Select a match** from the dropdown
2. **View risk timeline** — chart shows 0-100 risk score over 90 minutes
3. **Click danger dots** — see AI explanations
   - Red = Critical (80-100)
   - Orange = High (60-80)
   - Yellow = Moderate (40-60)
   - Green = Low (<40)
4. **Custom analysis** — click two points on timeline to analyze specific window
5. **Watch video** — jump to exact moments (if video available)

Each danger moment shows:
- Peak time and risk score
- Three-part AI analysis: **Context**, **Defensive Error**, **Coach Note**
- Goal indicator if moment resulted in goal

---

## Troubleshooting

**Backend won't start**: Ensure venv activated and `pip install -r requirements.txt` completed

**No matches appearing**: Check `matches/` folder exists with `*_pattern.xml` files

**LLM not working**: Verify `.env` has valid `OPENROUTER_API_KEY` and account has credits

**Tracking data not loading**: Run `python tracking_batch_parser.py` to parse `.txt` files

**Cache issues**: Clear with `rm -rf cache/explanations/*`

---

## Advanced

### Customize Risk Weights

Edit `OPPONENT_WEIGHTS` and `BARCA_WEIGHTS` in [risk_engine.py](risk_engine.py) (lines 15-30)

### Change LLM Model

Update `.env`:
- Budget: `openai/gpt-4o-mini`
- Best: `anthropic/claude-3.5-sonnet`
- Balance: `openai/gpt-4o`

### Customize Prompts

Edit `SYSTEM_PROMPT_MOMENT` in [explainer.py](explainer.py) (lines 28-100)

### Export Data

```python
from data_loader import list_matches, load_events
from risk_engine import compute_risk_score

match_name = list_matches()[0]
events_df = load_events(match_name)
risk_df = compute_risk_score(events_df)
risk_df.to_csv(f"export_{match_name}.csv", index=False)
```

---

**FC Barcelona "More than a Hack" Competition Submission**
