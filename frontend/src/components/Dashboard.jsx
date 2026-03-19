import ScoreCard from "./ScoreCard";
import Skills from "./Skills";

function MetaBadge({ label, value, icon }) {
  return (
    <div className="meta-badge">
      <div className="meta-badge__icon">{icon}</div>
      <div>
        <div className="meta-badge__label">{label}</div>
        <div className="meta-badge__value">{value}</div>
      </div>
    </div>
  );
}

function EntityList({ title, items }) {
  return (
    <div className="entity-list">
      <div className="entity-list__header">
        <span className="entity-list__title">{title}</span>
        <span className="entity-list__count">{items.length}</span>
      </div>
      {items.length > 0 ? (
        <div className="entity-list__tags">
          {items.map((item) => (
            <span key={item} className="entity-tag">{item}</span>
          ))}
        </div>
      ) : (
        <p className="entity-list__empty">None detected</p>
      )}
    </div>
  );
}

export default function Dashboard({ data }) {
  const {
    resume_score, skills_found, job_skills, matched_skills,
    missing_skills, job_match_score, semantic_match_score,
    achievement_count, companies, locations,
  } = data;

  const grade =
    resume_score >= 80 ? "A" :
    resume_score >= 70 ? "B" :
    resume_score >= 60 ? "C" :
    resume_score >= 50 ? "D" : "F";

  return (
    <div className="dashboard">
      {/* Summary bar */}
      <div className="summary-bar">
        <div className="summary-bar__grade-block">
          <span className={`summary-bar__grade grade--${grade}`}>{grade}</span>
          <div>
            <div className="summary-bar__grade-label">Overall Grade</div>
            <div className="summary-bar__grade-sub">Based on resume score {resume_score}/100</div>
          </div>
        </div>
        <div className="summary-bar__divider" />
        <div className="summary-bar__meta">
          <MetaBadge label="Achievements" value={`${achievement_count} detected`} icon="🏆" />
          <MetaBadge label="Skills Found" value={`${skills_found.length} skills`}  icon="⚡" />
          <MetaBadge label="Job Required" value={`${job_skills.length} skills`}    icon="📋" />
        </div>
      </div>

      {/* Grid */}
      <div className="dashboard__grid">
        <ScoreCard
          resumeScore={resume_score}
          jobMatchScore={job_match_score}
          semanticMatchScore={semantic_match_score}
        />
        <Skills
          skillsFound={skills_found}
          matchedSkills={matched_skills}
          missingSkills={missing_skills}
        />
      </div>

      {/* Entities */}
      <div className="dashboard__entities">
        <EntityList title="Locations Mentioned" items={locations} />
      </div>
    </div>
  );
}