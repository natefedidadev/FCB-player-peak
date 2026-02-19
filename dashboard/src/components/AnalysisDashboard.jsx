import { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { fetchMatches, fetchRisk, fetchDangers } from "../api";
import RiskTimeline from "./RiskTimeline";
import DangerPanel from "./DangerPanel";
import DangerList from "./DangerList";
import WindowAnalysis from "./WindowAnalysis";

export default function AnalysisDashboard() {
  const { matchIndex } = useParams();
  const index = Number(matchIndex);

  const [matchName, setMatchName] = useState("");
  const [riskData, setRiskData] = useState(null);
  const [dangersData, setDangersData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedDanger, setSelectedDanger] = useState(null);
  const [windowStart, setWindowStart] = useState(null);
  const [windowEnd, setWindowEnd] = useState(null);

  useEffect(() => {
    setLoading(true);
    setSelectedDanger(null);
    setWindowStart(null);
    setWindowEnd(null);

    Promise.all([
      fetchMatches(),
      fetchRisk(index),
      fetchDangers(index),
    ])
      .then(([matches, risk, dangers]) => {
        const m = matches.find((m) => m.index === index);
        setMatchName(m?.name || `Match ${index}`);
        setRiskData(risk);
        setDangersData(dangers);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [index]);

  const handleChartClick = (point) => {
    if (!windowStart) {
      setWindowStart(point);
      setWindowEnd(null);
    } else if (!windowEnd) {
      setWindowEnd(point);
    } else {
      setWindowStart(point);
      setWindowEnd(null);
    }
  };

  const handleClearWindow = () => {
    setWindowStart(null);
    setWindowEnd(null);
  };

  const showRightPanel = selectedDanger || (windowStart && windowEnd);

  if (loading) {
    return (
      <div className="flex items-center justify-center" style={{ height: "calc(100vh - 64px)" }}>
        <p className="text-muted text-lg">Loading match data...</p>
      </div>
    );
  }

  return (
    <div className="mx-auto px-4 py-4 flex flex-col" style={{ height: "calc(100vh - 64px)" }}>
      {/* Back link + match header */}
      <div className="flex items-center gap-4 mb-4 shrink-0">
        <Link
          to="/matches"
          className="text-muted hover:text-white transition-colors no-underline flex items-center gap-1 text-sm"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
          </svg>
          Back to matches
        </Link>
        <div className="h-4 w-px bg-white/10" />
        <h1 className="text-xl font-bold text-white">{matchName}</h1>
      </div>

      {riskData && dangersData && (
        <div className="flex gap-4 flex-1 min-h-0">
          {/* Left side: chart + dangers */}
          <div className={`flex flex-col gap-4 min-h-0 transition-all duration-300 ${showRightPanel ? "w-[60%]" : "w-full"}`}>
            <div className="flex-1 min-h-0">
              <RiskTimeline
                timeline={riskData.timeline}
                dangers={dangersData.dangers}
                onDangerClick={setSelectedDanger}
                windowStart={windowStart}
                windowEnd={windowEnd}
                onChartClick={handleChartClick}
                compact={showRightPanel}
              />
            </div>
            <div className="shrink-0">
              <DangerList
                dangers={dangersData.dangers}
                selectedDanger={selectedDanger}
                onSelect={setSelectedDanger}
              />
            </div>
          </div>

          {/* Right side: explanation panel */}
          {showRightPanel && (
            <div className="w-[40%] flex flex-col gap-4 min-h-0 animate-[fadeIn_0.3s_ease]">
              {selectedDanger && (
                <DangerPanel
                  danger={selectedDanger}
                  onClose={() => setSelectedDanger(null)}
                  matchIndex={index}
                  videoOffset={dangersData.video_pre_match_offset || 0}
                />
              )}
              {windowStart && windowEnd && (
                <WindowAnalysis
                  matchIndex={index}
                  windowStart={windowStart}
                  windowEnd={windowEnd}
                  onClear={handleClearWindow}
                />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
