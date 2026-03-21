from flask import Flask, request, jsonify
import pdfplumber
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


# -----------------------------
# Skill database
# -----------------------------
SKILLS = [
    # --- Frontend / Web ---
    "html", "css", "javascript", "typescript",
    "react", "next.js", "vue", "angular", "svelte",
    "tailwind", "sass", "webpack", "vite", "graphql",
    "storybook", "jest", "cypress", "playwright",

    # --- Backend / Systems ---
    "python", "java", "go", "rust", "c++", "c#", "php", "ruby",
    "node", "express", "fastapi", "django", "flask",
    "spring", "spring boot", "rails", ".net",
    "sql", "postgresql", "mysql", "sqlite", "mongodb",
    "redis", "kafka", "rabbitmq", "grpc", "rest api", "microservices",

    # --- Data / ML / AI ---
    "machine learning", "deep learning", "nlp",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "keras", "xgboost", "spark", "hadoop",
    "airflow", "dbt", "snowflake", "bigquery", "databricks",
    "data pipelines", "feature engineering", "llm", "langchain",

    # --- DevOps / Cloud / Infra ---
    "docker", "kubernetes", "terraform", "ansible", "helm",
    "aws", "gcp", "azure", "ci/cd",
    "github actions", "jenkins", "gitlab ci",
    "linux", "bash", "nginx", "prometheus", "grafana",
    "datadog", "opentelemetry",

    # --- Mobile ---
    "react native", "flutter", "swift", "kotlin",
    "android", "ios", "expo", "xcode",

    # --- Cybersecurity ---
    "penetration testing", "owasp", "siem",
    "firewalls", "iam", "zero trust", "soc",
    "cryptography", "vulnerability assessment",
    "network security", "devsecops",
]

# -----------------------------
# Alias map — variants → canonical skill name
# Detected aliases are normalized before matching/scoring
# -----------------------------
SKILL_ALIASES = {
    "node.js":      "node",
    "nodejs":       "node",
    "reactjs":      "react",
    "react.js":     "react",
    "react native": "react native",
    "vuejs":        "vue",
    "vue.js":       "vue",
    "nextjs":       "next.js",
    "angularjs":    "angular",
    "postgres":     "postgresql",
    "tf":           "tensorflow",
    "sklearn":      "scikit-learn",
    "scikit learn": "scikit-learn",
    "k8s":          "kubernetes",
    "gha":          "github actions",
}

# -----------------------------
# Skill groups — if resume has any member, treat group as satisfied
# Prevents "sql (missing)" when resume has postgresql/mysql
# -----------------------------
SKILL_GROUPS = {
    "sql":              ["sql", "postgresql", "mysql", "sqlite"],
    "node":             ["node", "node.js", "nodejs"],
    "react":            ["react", "reactjs", "react.js"],
    "machine learning": ["machine learning", "ml", "scikit-learn", "tensorflow", "pytorch"],
    "javascript":       ["javascript", "typescript"],
    "cloud":            ["aws", "gcp", "azure"],
    "ci/cd":            ["ci/cd", "github actions", "jenkins", "gitlab ci"],
}

ACTION_VERBS = [
    "spearhead", "architect", "engineer", "optimize",
    "improve", "increase", "reduce", "establish",
    "build", "launch", "deliver", "deploy",
    "design", "develop", "automate"
]

RESULT_TERMS = [
    "increase", "improve", "reduce", "optimize", "productivity",
    "system", "platform", "pipeline", "service", "application",
    "users", "customers"
]


# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {str(e)}")

    if not text.strip():
        raise ValueError("PDF appears to be empty or unreadable.")

    return text


# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.+# ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# -----------------------------
# Tokenization
# -----------------------------
def tokenize_text(doc):
    tokens = []
    for token in doc:
        if not token.is_stop and token.is_alpha:
            tokens.append(token.lemma_)
    return tokens


# -----------------------------
# Detect N-grams
# -----------------------------
def detect_ngrams(text):
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    vectorizer.fit([text])
    return set(vectorizer.get_feature_names_out())


# -----------------------------
# Skill matching helper — word boundary to prevent false positives
# e.g. "java" won't match inside "javascript"
# -----------------------------
def skill_in_text(skill, text):
    return bool(re.search(rf'\b{re.escape(skill)}\b', text))


# -----------------------------
# Normalize extracted skills via alias map
# e.g. ["node.js", "nodejs", "node"] → ["node"]
# -----------------------------
def normalize_skills(skills):
    normalized = set()
    for skill in skills:
        canonical = SKILL_ALIASES.get(skill, skill)
        normalized.add(canonical)
    return list(normalized)


# -----------------------------
# Skill extraction
# -----------------------------
def extract_skills(tokens, text):
    found_skills = []
    joined_tokens = " ".join(tokens)
    ngrams = detect_ngrams(text)

    for skill in SKILLS:
        if skill_in_text(skill, joined_tokens) or skill_in_text(skill, text) or skill in ngrams:
            found_skills.append(skill)

    # Also check aliases so variant spellings in resumes are caught
    for alias, canonical in SKILL_ALIASES.items():
        if skill_in_text(alias, text):
            found_skills.append(canonical)

    return normalize_skills(found_skills)


# -----------------------------
# Entity extraction
# FIX: Cross-references SKILLS + SKILL_ALIASES to prevent tech terms
# (PostgreSQL, React, Node, Vue.js etc.) from being tagged as locations
# -----------------------------
def extract_entities(doc):
    companies = []
    dates = []
    locations = []

    # Build blocklist from known skills and aliases for GPE filtering
    skills_lower = {s.lower() for s in SKILLS}
    aliases_lower = {a.lower() for a in SKILL_ALIASES}
    tech_blocklist = skills_lower | aliases_lower

    for ent in doc.ents:
        value = ent.text.strip()
        value = re.sub(r"\s+", " ", value)

        if len(value) < 3:
            continue

        if ent.label_ == "DATE":
            dates.append(value)
            continue

        if ent.label_ == "GPE":
            if value.lower() in tech_blocklist:
                continue
            if re.search(r"\.(js|ts|py|net|io|dev)$", value, re.IGNORECASE):
                continue
            if re.search(r"[A-Z][a-z]+[A-Z]", value):          # EmberJs, VueJs
                continue
            if re.search(r"[a-zA-Z](js|ts|py)$", value):       # Emberjs, Nodejs, Vuejs
                continue
            locations.append(value)
            continue

        if ent.label_ == "ORG":
            # Reject path-like patterns
            if "/" in value:
                continue
            # Reject URL/file extension patterns, but allow names like "J.P. Morgan"
            if re.search(r"\.\w{2,4}$", value) or re.search(r"\b\w+\.\w+/", value):
                continue
            # Reject single-word uppercase tech acronyms (AWS, PHP, SQL)
            if value.isupper() and len(value) <= 6:
                continue
            # Reject bullet fragments
            if value.startswith("•"):
                continue
            # Reject very long garbage strings
            if len(value.split()) > 4:
                continue
            # Reject if contains digits
            if re.search(r"\d", value):
                continue
            companies.append(value)

    return {
        "companies": list({c.lower(): c for c in companies}.values()),
        "dates":     list(set(dates)),
        "locations": list({l.lower(): l for l in locations}.values()),
    }


# -----------------------------
# Achievement detection
# -----------------------------
def find_quantified_achievements(text):
    bullets = re.split(r"[•\-\*▪→\n]", text)
    achievements = []

    for bullet in bullets:
        bullet = bullet.strip().lower()

        if len(bullet.split()) < 5:
            continue
        if len(bullet.split()) > 50:
            continue

        has_action = any(v in bullet for v in ACTION_VERBS)
        has_result = (
            bool(re.search(r"\d+", bullet)) or
            any(term in bullet for term in RESULT_TERMS)
        )

        if has_action and has_result:
            achievements.append(bullet)

    return achievements


# -----------------------------
# Extract job skills (with alias normalization)
# -----------------------------
def extract_job_skills(job_description):
    job_description = job_description.lower()
    job_skills = []

    for skill in SKILLS:
        if skill_in_text(skill, job_description):
            job_skills.append(skill)

    for alias, canonical in SKILL_ALIASES.items():
        if skill_in_text(alias, job_description):
            job_skills.append(canonical)

    return list(set(normalize_skills(job_skills)))


# -----------------------------
# Group-aware skill comparison
# "sql missing" when resume has "postgresql" is correctly treated as matched
# -----------------------------
def compare_skills(resume_skills, job_skills):
    matched = []
    missing = []
    resume_set = set(resume_skills)

    for skill in job_skills:
        if skill in resume_set:
            matched.append(skill)
            continue

        # Group match — resume satisfies requirement via any group member
        group = SKILL_GROUPS.get(skill, [])
        if any(member in resume_set for member in group):
            matched.append(skill)
        else:
            missing.append(skill)

    score = 0 if len(job_skills) == 0 else int((len(matched) / len(job_skills)) * 100)
    return matched, missing, score


# -----------------------------
# Semantic similarity
# -----------------------------
def compute_resume_job_similarity(resume_text, job_description):
    if job_description.strip() == "":
        return 0

    documents = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(matrix[0], matrix[1])
    return round(similarity[0][0] * 100, 2)


# -----------------------------
# Resume score
# Rebalanced: equal weight on skill match + semantic alignment,
# gentler achievement ramp, floor of 35 for resumes with real signal
# -----------------------------
def calculate_resume_score(skill_score, semantic_score, achievements, skills):
    score = 0

    # For technical resumes, skill match is the most reliable signal
    score += skill_score * 0.45         # primary weight — direct skill match
    score += semantic_score * 0.20      # reduced — TF-IDF undersells tech resumes
    score += min(len(achievements) * 3, 15)  # achievements: cap 15
    score += min(len(skills) / len(SKILLS), 1.0) * 10  # skill breadth: 0–10

    # Bonus: reward resumes with strong skill matches
    if skill_score >= 70:
        score += 8
    elif skill_score >= 50:
        score += 4

    # Floor: any resume with real signal scores at least 40
    if skill_score > 0 or semantic_score > 20:
        score = max(score, 40)

    return min(round(score), 100)


# -----------------------------
# Resume feedback
# -----------------------------
def generate_resume_feedback(job_match_score, missing_skills, achievements, semantic_score):
    feedback = []

    if missing_skills:
        feedback.append(
            f"You are missing {len(missing_skills)} skill(s) from the job description: "
            + ", ".join(missing_skills)
            + ". Consider adding these if you have experience with them."
        )
    else:
        feedback.append("Great — your resume covers all the skills mentioned in the job description.")

    if len(achievements) == 0:
        feedback.append(
            "No quantified achievements were detected. Add bullet points with numbers "
            "(e.g., 'Improved load time by 40%', 'Served 10,000+ users') to strengthen your resume."
        )
    elif len(achievements) < 5:
        feedback.append(
            f"Only {len(achievements)} quantified achievement(s) detected. "
            "Aim for at least 5 bullet points with measurable outcomes."
        )
    else:
        feedback.append(
            f"Good job — {len(achievements)} quantified achievements found. "
            "These significantly improve recruiter confidence."
        )

    if job_match_score < 50:
        feedback.append(
            f"Your resume matches only {job_match_score}% of the required job skills. "
            "Consider heavily tailoring your resume for this role."
        )
    elif job_match_score < 70:
        feedback.append(
            f"Your resume matches {job_match_score}% of the job requirements. "
            "A few additions could push this higher."
        )
    else:
        feedback.append(
            f"Strong match — your resume aligns with {job_match_score}% of the job requirements."
        )

    if semantic_score < 40:
        feedback.append(
            f"Semantic similarity score is {semantic_score}%. "
            "Try mirroring the language and terminology from the job description more closely."
        )
    else:
        feedback.append(
            f"Semantic similarity score is {semantic_score}% — your resume language aligns reasonably well with the job posting."
        )

    return feedback


# -----------------------------
# API endpoint
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload_resume():

    if "resume" not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400

    file = request.files["resume"]

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported. Please upload a .pdf file."}), 400

    job_description = request.form.get("job_description", "")

    try:
        text = extract_text_from_pdf(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    cleaned_text = clean_text(text)

    # spaCy doc built once, reused across tokenize + entity extraction
    doc = nlp(text)

    tokens = tokenize_text(doc)
    skills = extract_skills(tokens, cleaned_text)
    entities = extract_entities(doc)
    achievements = find_quantified_achievements(text)
    job_skills = extract_job_skills(job_description)
    matched, missing, score = compare_skills(skills, job_skills)
    semantic_score = compute_resume_job_similarity(text, job_description)
    resume_score = calculate_resume_score(score, semantic_score, achievements, skills)
    feedback = generate_resume_feedback(score, missing, achievements, semantic_score)

    return jsonify({
        "resume_score":            resume_score,
        "skills_found":            skills,
        "job_skills":              job_skills,
        "matched_skills":          matched,
        "missing_skills":          missing,
        "job_match_score":         score,
        "semantic_match_score":    semantic_score,
        "companies":               entities["companies"],
        "dates":                   entities["dates"],
        "locations":               entities["locations"],
        "quantified_achievements": achievements,
        "achievement_count":       len(achievements),
        "resume_feedback":         feedback,
    })


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    print("Resume Analyzer running at http://localhost:5000")
    app.run(debug=True)