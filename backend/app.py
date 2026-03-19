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
    "python","java","javascript","react","react native",
    "node","node.js","express","sql","postgresql","mysql",
    "docker","aws","html","css","tailwind","kubernetes",
    "machine learning","pandas","numpy"
]

KNOWN_LOCATIONS = [
    "mumbai","delhi","bangalore","pune","hyderabad",
    "chennai","kolkata","india","berlin","frankfurt","ghana"
]

ACTION_VERBS = [
    "spearhead","architect","engineer","optimize",
    "improve","increase","reduce","establish",
    "build","launch","deliver","deploy",
    "design","develop","automate"
]

RESULT_TERMS = [
    "increase","improve","reduce","optimize","productivity",
    "system","platform","pipeline","service","application",
    "users","customers"
]


# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text_from_pdf(file):

    text = ""

    # IMPROVEMENT: Wrapped in try/except to handle corrupted or unreadable PDFs
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
# IMPROVEMENT: Now accepts a pre-built spaCy doc instead of raw text
# to avoid running nlp() twice (performance fix)
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

    vectorizer = CountVectorizer(ngram_range=(1,3))

    vectorizer.fit([text])

    return set(vectorizer.get_feature_names_out())


# -----------------------------
# Skill extraction
# IMPROVEMENT: Uses whole-word regex matching to prevent false positives
# e.g. "java" no longer matches inside "javascript"
# -----------------------------
def skill_in_text(skill, text):
    return bool(re.search(rf'\b{re.escape(skill)}\b', text))


def extract_skills(tokens, text):

    found_skills = []

    joined_tokens = " ".join(tokens)

    ngrams = detect_ngrams(text)

    for skill in SKILLS:

        # IMPROVEMENT: skill_in_text uses \b word boundary instead of plain `in`
        if skill_in_text(skill, joined_tokens) or skill_in_text(skill, text) or skill in ngrams:
            found_skills.append(skill)

    return list(set(found_skills))


# -----------------------------
# Entity extraction
# IMPROVEMENT: Now accepts a pre-built spaCy doc instead of raw text
# to avoid running nlp() twice (performance fix)
# -----------------------------
def extract_entities(doc):

    companies = []
    dates = []
    locations = []

    for ent in doc.ents:

        value = ent.text.strip()
        value = re.sub(r"\s+", " ", value)

        if len(value) < 3:
            continue

        if ent.label_ == "DATE":
            dates.append(value)
            continue

        if ent.label_ == "GPE":
            if value.lower() in KNOWN_LOCATIONS:
                locations.append(value)
            continue

        if ent.label_ == "ORG":

            # reject obvious tech patterns
            if "/" in value or "." in value:
                continue

            # reject single-word uppercase tech (AWS, PHP, SQL)
            if value.isupper() and len(value) <= 6:
                continue

            # reject bullet fragments
            if value.startswith("•"):
                continue

            # reject very long garbage strings
            if len(value.split()) > 4:
                continue

            # reject if it contains digits
            if re.search(r"\d", value):
                continue

            companies.append(value)

    return {
        "companies": list(set(companies)),
        "dates": list(set(dates)),
        "locations": list(set(locations))
    }


# -----------------------------
# Achievement detection
# IMPROVEMENT: Expanded bullet splitter to handle more bullet styles
# IMPROVEMENT: Loosened word count range from (6,30) to (5,50)
# -----------------------------
def find_quantified_achievements(text):

    # IMPROVEMENT: Added -, *, ▪, → as valid bullet delimiters
    bullets = re.split(r"[•\-\*▪→\n]", text)

    achievements = []

    for bullet in bullets:

        bullet = bullet.strip().lower()

        # IMPROVEMENT: Loosened range to 5–50 words to capture more valid achievements
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

    job_skills = []

    for skill in SKILLS:

        if skill in job_description:
            job_skills.append(skill)

    return list(set(job_skills))


# -----------------------------
# Compare skills
# -----------------------------
def compare_skills(resume_skills, job_skills):

    matched = []
    missing = []

    for skill in job_skills:

        if skill in resume_skills:
            matched.append(skill)
        else:
            missing.append(skill)

    if len(job_skills) == 0:
        score = 0
    else:
        score = int((len(matched) / len(job_skills)) * 100)

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

    score = similarity[0][0] * 100

    return round(score, 2)


# -----------------------------
# Resume score
# IMPROVEMENT: Added min(score, 100) cap to prevent scores exceeding 100
# IMPROVEMENT: Normalized skill count contribution using ratio instead of raw count
# -----------------------------
def calculate_resume_score(skill_score, semantic_score, achievements, skills):

    score = 0

    score += skill_score * 0.4
    score += semantic_score * 0.3
    score += min(len(achievements) * 5, 20)

    # IMPROVEMENT: Normalize skill count to 0-10 range based on total available skills
    skill_ratio = min(len(skills) / len(SKILLS), 1.0)
    score += skill_ratio * 10

    # IMPROVEMENT: Cap at 100 to prevent overflow
    return min(round(score), 100)


# -----------------------------
# Resume feedback
# IMPROVEMENT: Feedback now references actual data points (score numbers,
# specific missing skill categories, positive reinforcement when score is high)
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
    elif semantic_score >= 40:
        feedback.append(
            f"Semantic similarity score is {semantic_score}% — your resume language aligns reasonably well with the job posting."
        )

    return feedback


# -----------------------------
# API endpoint
# IMPROVEMENT: Added file extension validation and error handling
# IMPROVEMENT: spaCy doc is built once and shared across tokenize + entity functions
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload_resume():

    if "resume" not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400

    file = request.files["resume"]

    # IMPROVEMENT: Reject non-PDF files before processing
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported. Please upload a .pdf file."}), 400

    job_description = request.form.get("job_description", "")

    # IMPROVEMENT: Wrapped extraction in try/except to return clean error on bad PDFs
    try:
        text = extract_text_from_pdf(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    cleaned_text = clean_text(text)

    # IMPROVEMENT: Run spaCy ONCE and reuse doc in both tokenize and entity extraction
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

        "resume_score": resume_score,

        "skills_found": skills,

        "job_skills": job_skills,

        "matched_skills": matched,

        "missing_skills": missing,

        "job_match_score": score,

        "semantic_match_score": semantic_score,

        "companies": entities["companies"],

        "dates": entities["dates"],

        "locations": entities["locations"],

        "quantified_achievements": achievements,

        "achievement_count": len(achievements),

        "resume_feedback": feedback

    })


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":

    print("Resume Analyzer running at http://localhost:5000")

    app.run(debug=True)