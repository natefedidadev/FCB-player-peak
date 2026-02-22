import { useRef } from "react";

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

const SEVERITY_COLORS = {
  critical: "border-red-500/30 bg-red-500/10",
  high: "border-orange-500/30 bg-orange-500/10",
  moderate: "border-yellow-500/30 bg-yellow-500/10",
  low: "border-green-500/30 bg-green-500/10",
};

const SEVERITY_DOT = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  moderate: "bg-yellow-500",
  low: "bg-green-500",
};

function ArrowButton({ direction, onClick }) {
  return (
    <button
      onClick={onClick}
      className="absolute top-1/2 -translate-y-1/2 z-10 w-9 h-9 rounded-full
                 bg-surface border border-white/10 shadow-lg
                 flex items-center justify-center
                 text-white/70 hover:text-white hover:border-white/20
                 transition-colors cursor-pointer"
      style={{ [direction === "left" ? "left" : "right"]: -4 }}
    >
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
        {direction === "left" ? (
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
        ) : (
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
        )}
      </svg>
    </button>
  );
}

export default function DangerList({ dangers, selectedDanger, onSelect }) {
  const scrollRef = useRef(null);

  if (!dangers || dangers.length === 0) return null;

  const sorted = [...dangers].sort((a, b) => a.display_peak_sec - b.display_peak_sec);

  const scroll = (dir) => {
    if (!scrollRef.current) return;
    const amount = 300;
    scrollRef.current.scrollBy({ left: dir === "left" ? -amount : amount, behavior: "smooth" });
  };

  return (
    <div className="bg-surface rounded-2xl p-5 border border-white/5 shadow-[0_4px_30px_rgba(0,0,0,0.3)]">
      <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-3">
        Danger Moments ({dangers.length})
      </h3>
      <div className="relative">
        <ArrowButton direction="left" onClick={() => scroll("left")} />
        <ArrowButton direction="right" onClick={() => scroll("right")} />

        <div
          ref={scrollRef}
          className="overflow-x-auto px-6 py-1"
          style={{ scrollbarWidth: "none", msOverflowStyle: "none", WebkitOverflowScrolling: "touch" }}
        >
          <style>{`.danger-scroll::-webkit-scrollbar { display: none; }`}</style>
          <div className="grid grid-rows-2 grid-flow-col gap-3 danger-scroll" style={{ width: "max-content", paddingRight: 24, paddingTop: 4, paddingBottom: 4 }}>
            {sorted.map((d, i) => {
              const isSelected = selectedDanger && selectedDanger.peak_time === d.peak_time;
              return (
                <button
                  key={i}
                  onClick={() => onSelect(d)}
                  className={`text-center px-6 py-5 rounded-xl border transition-all duration-75 min-w-[140px]
                    ${isSelected ? "border-barca-blue/50 bg-barca-blue/15 shadow-[0_0_15px_rgba(0,77,152,0.2)] scale-105" : SEVERITY_COLORS[d.severity]}
                    hover:shadow-[0_0_15px_rgba(0,77,152,0.15)] hover:scale-[1.03] cursor-pointer`}
                  style={{ minHeight: 110 }}
                >
                  <div className="flex items-center justify-center gap-2 mb-1">
                    <span className={`w-2.5 h-2.5 rounded-full ${SEVERITY_DOT[d.severity]}`} />
                    <span className="text-lg font-semibold text-white">
                      {fmtTime(d.display_peak_sec)}
                    </span>
                  </div>
                  <span className="text-muted text-sm uppercase">{d.severity}</span>
                  <div className="mt-1.5">
                    {d.resulted_in_goal ? (
                      <span className="text-red-400 text-sm font-bold">GOAL</span>
                    ) : (
                      <span className="text-muted text-sm">{d.peak_score.toFixed(0)}/100</span>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
