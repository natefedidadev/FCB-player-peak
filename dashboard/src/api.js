const BASE = "http://localhost:8000";

export async function fetchMatches() {
  const res = await fetch(`${BASE}/api/matches`);
  if (!res.ok) throw new Error("Failed to fetch matches");
  return res.json();
}

export async function fetchRisk(matchIndex) {
  const res = await fetch(`${BASE}/api/matches/${matchIndex}/risk`);
  if (!res.ok) throw new Error("Failed to fetch risk data");
  return res.json();
}

export async function fetchDangers(matchIndex) {
  const res = await fetch(`${BASE}/api/matches/${matchIndex}/dangers`);
  if (!res.ok) throw new Error("Failed to fetch danger moments");
  return res.json();
}

export async function analyzeWindow(matchIndex, startSec, endSec) {
  const res = await fetch(`${BASE}/api/matches/${matchIndex}/analyze-window`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start_sec: startSec, end_sec: endSec }),
  });
  if (!res.ok) throw new Error("Failed to analyze window");
  return res.json();
}
