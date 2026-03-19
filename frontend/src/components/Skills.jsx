function SkillGroup({ title, skills, variant, emptyMessage }) {
  return (
    <div className="skill-group">
      <div className="skill-group__header">
        <div className="skill-group__heading">
          <span className={`skill-group__dot skill-group__dot--${variant}`} />
          <span className={`skill-group__label skill-group__label--${variant}`}>{title}</span>
        </div>
        <span className="skill-group__count">{skills.length}</span>
      </div>
      {skills.length > 0 ? (
        <div className="skill-group__tags">
          {skills.map((s) => (
            <span key={s} className={`tag tag--${variant}`}>{s}</span>
          ))}
        </div>
      ) : (
        <p className="skill-group__empty">{emptyMessage}</p>
      )}
    </div>
  );
}

export default function Skills({ skillsFound, matchedSkills, missingSkills }) {
  return (
    <div className="skills">
      <span className="skills__title">Skills Analysis</span>
      <SkillGroup title="Skills Found"    skills={skillsFound}    variant="green" emptyMessage="No skills detected in resume" />
      <SkillGroup title="Matched Skills"  skills={matchedSkills}  variant="blue"  emptyMessage="No matching skills with job description" />
      <SkillGroup title="Missing Skills"  skills={missingSkills}  variant="red"   emptyMessage="Great — no missing skills!" />
    </div>
  );
}