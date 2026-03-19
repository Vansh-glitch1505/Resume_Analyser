function CircleProgress({ value, size = 80, strokeWidth = 6, color }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;
  return (
    <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
      <circle cx={size/2} cy={size/2} r={radius} fill="none"
        stroke="rgba(255,255,255,0.05)" strokeWidth={strokeWidth} />
      <circle cx={size/2} cy={size/2} r={radius} fill="none"
        stroke={color} strokeWidth={strokeWidth}
        strokeDasharray={circumference} strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: "stroke-dashoffset 1s ease-in-out" }} />
    </svg>
  );
}

function quality(v) {
  if (v >= 80) return "Excellent";
  if (v >= 60) return "Good";
  if (v >= 40) return "Fair";
  return "Needs Work";
}

function ScoreItem({ label, value, description, variant, hexColor }) {
  const v = Math.round(value ?? 0);
  return (
    <div className={`score-item score-item--${variant}`}>
      <div className="score-item__glow" />
      <div className="score-item__top">
        <div>
          <div className="score-item__label">{label}</div>
          <div className="score-item__desc">{description}</div>
        </div>
        <div className="ring-wrap">
          <CircleProgress value={v} color={hexColor} />
          <span className="ring-wrap__val">{v}</span>
        </div>
      </div>
      <div className="score-bar">
        <div className="score-bar__fill" style={{ width: `${v}%`, background: hexColor }} />
      </div>
      <div className="score-item__bottom">
        <span className="score-item__minmax">0</span>
        <span className="score-item__quality">{quality(v)}</span>
        <span className="score-item__minmax">100</span>
      </div>
    </div>
  );
}

export default function ScoreCard({ resumeScore, jobMatchScore, semanticMatchScore }) {
  const avg = Math.round((resumeScore + jobMatchScore + semanticMatchScore) / 3);
  const items = [
    { label: "Resume Score",    value: resumeScore,        description: "Overall resume quality and relevance", variant: "violet",  hexColor: "#9CD5FF" },
    { label: "Job Match",       value: jobMatchScore,      description: "Direct skill alignment with the role",  variant: "cyan",    hexColor: "#7AAACE" },
    { label: "Semantic Match",  value: semanticMatchScore, description: "Contextual similarity via TF-IDF",      variant: "fuchsia", hexColor: "#F7F8F0" },
  ];
  return (
    <div className="scorecard">
      <div className="scorecard__header">
        <span className="scorecard__title">Scores</span>
        <div className="scorecard__avg">
          <span className="scorecard__avg-label">Avg</span>
          <span className="scorecard__avg-val">{avg}</span>
        </div>
      </div>
      {items.map((s) => <ScoreItem key={s.label} {...s} />)}
    </div>
  );
}