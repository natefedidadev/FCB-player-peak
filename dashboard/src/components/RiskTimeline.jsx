import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceDot,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

const SEVERITY_COLORS = {
  critical: "#ef4444",
  high: "#f97316",
  moderate: "#eab308",
  low: "#22c55e",
};

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-surface rounded-xl px-4 py-2.5 text-sm shadow-lg border border-white/10">
      <p className="text-muted">{Math.floor(d.match_minute)}&apos; {String(Math.round((d.match_minute % 1) * 60)).padStart(2, "0")}&quot;</p>
      <p className="font-semibold text-white">Risk: {d.risk_score.toFixed(1)}</p>
    </div>
  );
}

export default function RiskTimeline({
  timeline,
  dangers,
  onDangerClick,
  windowStart,
  windowEnd,
  onChartClick,
  compact,
}) {
  const handleClick = (e) => {
    if (!e || !e.activePayload) return;
    const point = e.activePayload[0].payload;
    onChartClick(point);
  };

  return (
    <div className="bg-surface rounded-2xl p-5 border border-white/5 shadow-[0_4px_30px_rgba(0,0,0,0.3)] h-full flex flex-col">
      <div className="flex justify-between items-center mb-3 shrink-0">
        <h2 className="text-base font-semibold text-white">Defensive Risk Timeline</h2>
        <p className="text-xs text-muted">Click two points to select an analysis window</p>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={timeline} onClick={handleClick} style={{ cursor: "crosshair" }}>
          <defs>
            <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#A50044" stopOpacity={0.6} />
              <stop offset="40%" stopColor="#ef4444" stopOpacity={0.2} />
              <stop offset="100%" stopColor="#22c55e" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="match_minute"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v) => `${Math.round(v)}'`}
            stroke="#3a1f2d"
            tick={{ fill: "#8a7580", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 100]}
            stroke="#3a1f2d"
            tick={{ fill: "#8a7580", fontSize: 11 }}
            width={35}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="risk_score"
            fill="url(#riskGradient)"
            stroke="#A50044"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />

          {windowStart && windowEnd && (
            <ReferenceArea
              x1={Math.min(windowStart.match_minute, windowEnd.match_minute)}
              x2={Math.max(windowStart.match_minute, windowEnd.match_minute)}
              fill="#004D98"
              fillOpacity={0.15}
              stroke="#004D98"
              strokeDasharray="4 4"
            />
          )}

          {windowStart && !windowEnd && (
            <ReferenceLine
              x={windowStart.match_minute}
              stroke="#004D98"
              strokeWidth={2}
              strokeDasharray="4 4"
            />
          )}

          {dangers.map((d, i) => (
            <ReferenceDot
              key={i}
              x={d.display_peak_minute}
              y={d.peak_score}
              r={d.severity === "critical" ? 8 : d.severity === "high" ? 7 : 6}
              fill={SEVERITY_COLORS[d.severity]}
              stroke="#1a0c12"
              strokeWidth={2.5}
              onClick={() => onDangerClick(d)}
              style={{ cursor: "pointer", filter: "drop-shadow(0 0 6px rgba(239,68,68,0.4))" }}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
