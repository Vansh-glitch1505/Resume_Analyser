from flask import Flask, request, jsonify
import pdfplumber
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

# --- BERT semantic similarity ---
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
CORS(app)

# -----------------------------
# SpaCy — safe load with auto-download
# -----------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# -----------------------------
# BERT — lazy load + background warm-up
# -----------------------------
BERT_MODEL = None

def get_bert_model():
    global BERT_MODEL
    if BERT_MODEL is None:
        print("Loading BERT model...")
        BERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("BERT model loaded.")
    return BERT_MODEL



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
# Alias map
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
# Skill groups
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

JD_ENRICHMENT_MAP = {
    "full stack":       "full stack developer frontend backend web application REST API database integration",
    "frontend":         "frontend developer UI components web design responsive layout browser DOM",
    "backend":          "backend developer server side API database business logic scalability performance",
    "react":            "react javascript frontend component state management hooks JSX UI rendering SPA",
    "node":             "node.js backend javascript server express REST API async event loop npm",
    "node.js":          "node.js backend javascript server express REST API async event loop npm",
    "python":           "python scripting automation backend data processing django fastapi flask OOP",
    "django":           "django python web framework ORM models views templates REST authentication",
    "fastapi":          "fastapi python async REST API backend microservices pydantic OpenAPI",
    "postgresql":       "postgresql relational database SQL queries schema indexing joins transactions",
    "mysql":            "mysql relational database SQL queries schema indexing stored procedures",
    "mongodb":          "mongodb nosql database collections documents aggregation atlas",
    "aws":              "amazon web services cloud infrastructure EC2 S3 Lambda RDS deployment IAM",
    "gcp":              "google cloud platform cloud services kubernetes bigquery deployment pubsub",
    "azure":            "microsoft azure cloud services devops pipelines deployment blob storage",
    "machine learning": "machine learning model training prediction scikit-learn feature engineering data preprocessing",
    "deep learning":    "deep learning neural networks tensorflow pytorch GPU training inference CNN RNN",
    "devops":           "devops CI CD docker kubernetes infrastructure automation deployment monitoring",
    "docker":           "docker containerization deployment microservices image registry orchestration compose",
    "kubernetes":       "kubernetes container orchestration deployment scaling cluster management helm",
    "data engineer":    "data engineering pipelines ETL airflow spark bigquery data warehouse transformation",
    "android":          "android mobile development kotlin java UI jetpack compose material design",
    "ios":              "ios mobile development swift xcode UIKit SwiftUI app store",
    "flutter":          "flutter dart cross platform mobile development UI widgets state management",
    "cybersecurity":    "cybersecurity penetration testing vulnerability OWASP network security IAM zero trust",
    "javascript":       "javascript ES6 async await promises DOM manipulation web APIs frontend backend",
    "typescript":       "typescript typed javascript interfaces generics decorators compile time safety",
    "graphql":          "graphql API query language schema resolvers mutations subscriptions apollo",
    "rest api":         "REST API HTTP endpoints JSON CRUD authentication authorization swagger",
    "microservices":    "microservices distributed systems service mesh API gateway event driven architecture",
}


def enrich_job_description(job_description):
    jd_lower     = job_description.lower()
    enrichments  = []

    for keyword, expansion in JD_ENRICHMENT_MAP.items():
        if keyword in jd_lower:
            enrichments.append(expansion)

    if enrichments:
        all_words    = " ".join(enrichments).split()
        seen         = set()
        unique_words = []
        for w in all_words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)
        enriched = job_description + " " + " ".join(unique_words)
    else:
        enriched = job_description

    return enriched


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
# Skill matching helper
# -----------------------------
def skill_in_text(skill, text):
    return bool(re.search(rf'\b{re.escape(skill)}\b', text))


# -----------------------------
# Normalize extracted skills via alias map
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
    found_skills  = []
    joined_tokens = " ".join(tokens)
    ngrams        = detect_ngrams(text)

    for skill in SKILLS:
        if skill_in_text(skill, joined_tokens) or skill_in_text(skill, text) or skill in ngrams:
            found_skills.append(skill)

    for alias, canonical in SKILL_ALIASES.items():
        if skill_in_text(alias, text):
            found_skills.append(canonical)

    return normalize_skills(found_skills)


# -----------------------------
# Entity extraction
# -----------------------------
def extract_entities(doc):
    companies = []
    dates     = []
    locations = []

    skills_lower   = {s.lower() for s in SKILLS}
    aliases_lower  = {a.lower() for a in SKILL_ALIASES}
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
            if re.search(r"[A-Z][a-z]+[A-Z]", value):
                continue
            if re.search(r"[a-zA-Z](js|ts|py)$", value):
                continue
            locations.append(value)
            continue

        if ent.label_ == "ORG":
            if "/" in value:
                continue
            if re.search(r"\.\w{2,4}$", value) or re.search(r"\b\w+\.\w+/", value):
                continue
            if value.isupper() and len(value) <= 6:
                continue
            if value.startswith("•"):
                continue
            if len(value.split()) > 4:
                continue
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
    bullets      = re.split(r"[•\-\*▪→\n]", text)
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
# Extract job skills
# -----------------------------
def extract_job_skills(job_description):
    job_description = job_description.lower()
    job_skills      = []

    for skill in SKILLS:
        if skill_in_text(skill, job_description):
            job_skills.append(skill)

    for alias, canonical in SKILL_ALIASES.items():
        if skill_in_text(alias, job_description):
            job_skills.append(canonical)

    return list(set(normalize_skills(job_skills)))


# -----------------------------
# Group-aware skill comparison
# -----------------------------
def compare_skills(resume_skills, job_skills):
    matched    = []
    missing    = []
    resume_set = set(resume_skills)

    for skill in job_skills:
        if skill in resume_set:
            matched.append(skill)
            continue

        group = SKILL_GROUPS.get(skill, [])
        if any(member in resume_set for member in group):
            matched.append(skill)
        else:
            missing.append(skill)

    score = 0 if len(job_skills) == 0 else int((len(matched) / len(job_skills)) * 100)
    return matched, missing, score


RESUME_SIGNAL_KEYWORDS = [
    "experience", "skill", "built", "developed", "designed",
    "engineer", "architect", "deploy", "manage", "led", "worked",
    "implemented", "created", "maintained", "optimized", "integrated",
    "python", "javascript", "react", "node", "aws", "api", "database",
    "backend", "frontend", "fullstack", "full stack", "cloud", "docker",
    "kubernetes", "sql", "postgresql", "mongodb", "java", "typescript",
    "project", "application", "system", "platform", "service", "pipeline",
    "framework", "library", "tool", "stack", "microservice", "server",
    "responsible", "collaborated", "contributed", "achieved", "improved",
]

RESUME_NOISE_KEYWORDS = [
    "references", "hobbies", "declaration", "date of birth",
    "nationality", "marital", "father", "mother", "permanent address",
    "correspondence address", "gender", "languages known", "religion",
]


def extract_relevant_resume_section(text):
    lines    = text.split("\n")
    relevant = []

    for line in lines:
        line_stripped = line.strip()
        line_lower    = line_stripped.lower()

        if not line_lower or len(line_lower.split()) < 3:
            continue

        if any(noise in line_lower for noise in RESUME_NOISE_KEYWORDS):
            continue

        if any(signal in line_lower for signal in RESUME_SIGNAL_KEYWORDS):
            relevant.append(line_stripped)

    result = " ".join(relevant)

    if len(result.split()) < 60:
        return text

    return result


# -----------------------------
# BERT chunking helpers
# -----------------------------
def chunk_text(text, chunk_size=400):
    words  = text.split()
    chunks = []
    step   = chunk_size - 50
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks if chunks else [text]


def get_bert_embedding(text):
    chunks     = chunk_text(text)
    embeddings = get_bert_model().encode(chunks, convert_to_numpy=True)
    return np.mean(embeddings, axis=0, keepdims=True)


# -----------------------------
# BERT semantic similarity
# -----------------------------
def compute_resume_job_similarity(resume_text, job_description):
    if not job_description.strip():
        return 0

    focused_resume = extract_relevant_resume_section(resume_text)
    clean_resume   = clean_text(focused_resume)
    clean_job      = clean_text(job_description)

    job_embedding    = get_bert_embedding(clean_job)
    chunks           = chunk_text(clean_resume)
    chunk_embeddings = get_bert_model().encode(chunks, convert_to_numpy=True)

    similarities = [
        float(cosine_similarity(emb.reshape(1, -1), job_embedding)[0][0])
        for emb in chunk_embeddings
    ]

    best_sim = max(similarities)
    mean_sim = float(np.mean(similarities))
    blended  = (best_sim * 0.65) + (mean_sim * 0.35)

    FLOOR    = 0.20
    CEIL     = 0.85
    rescaled = (blended - FLOOR) / (CEIL - FLOOR)
    rescaled = max(0.0, min(1.0, rescaled))

    return round(rescaled * 100, 2)


# -----------------------------
# Resume score
# -----------------------------
def calculate_resume_score(skill_score, semantic_score, achievements, skills):
    score = 0

    score += skill_score * 0.45
    score += semantic_score * 0.20
    score += min(len(achievements) * 3, 15)
    score += min(len(skills) / len(SKILLS), 1.0) * 10

    if skill_score >= 70:
        score += 8
    elif skill_score >= 50:
        score += 4

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
            f"Semantic similarity score is {semantic_score}% (BERT). "
            "Try mirroring the language and terminology from the job description more closely."
        )
    else:
        feedback.append(
            f"Semantic similarity score is {semantic_score}% (BERT) — "
            "your resume language aligns well with the job posting."
        )

    return feedback


# -----------------------------
# Root route — health check
# -----------------------------
@app.route("/")
def home():
    return "Backend is running ✅"


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
    doc          = nlp(text)

    tokens       = tokenize_text(doc)
    skills       = extract_skills(tokens, cleaned_text)
    entities     = extract_entities(doc)
    achievements = find_quantified_achievements(text)
    job_skills   = extract_job_skills(job_description)
    matched, missing, score = compare_skills(skills, job_skills)

    enriched_jd    = enrich_job_description(job_description)
    semantic_score = compute_resume_job_similarity(text, enriched_jd)

    resume_score = calculate_resume_score(score, semantic_score, achievements, skills)
    feedback     = generate_resume_feedback(score, missing, achievements, semantic_score)

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