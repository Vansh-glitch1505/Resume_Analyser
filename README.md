# 🚀 ML Resume Analyzer

An intelligent Resume Analyzer that evaluates resumes against job descriptions using NLP and semantic similarity (BERT).

Built using **React, Flask, NLP (spaCy), and Sentence Transformers (BERT)**.

---

## 📌 Features

- 📄 Upload resume (PDF)
- 🧠 Extract skills, experience, and entities
- 🎯 Match resume with job description
- 📊 Generate:
  - Resume Score
  - Job Match Score
  - Semantic Similarity Score (BERT-based)
- 🛠 Identify:
  - Matched Skills
  - Missing Skills
  - Quantified Achievements
- 💡 Provide actionable feedback to improve resume

---

## 🧠 How It Works

### 1. Resume Processing
- Extracts text from PDF using `pdfplumber`
- Cleans and preprocesses text
- Uses **spaCy** for tokenization and entity extraction

---

### 2. Skill Extraction
- Matches resume content with a predefined skill database
- Handles variations using alias mapping (e.g., Node.js → node)

---

### 3. Job Matching
- Compares resume skills with job description
- Uses group-based matching (e.g., SQL → PostgreSQL/MySQL)

---

### 4. Semantic Similarity (BERT 🔥)
- Uses `SentenceTransformer (all-MiniLM-L6-v2)`
- Converts resume & job description into embeddings
- Computes similarity using cosine similarity

> Unlike TF-IDF, BERT understands contextual meaning  
> (e.g., "developer" ≈ "engineer")

---

### 5. Scoring System

- 🧩 Skill Match Score (Primary)
- 🧠 Semantic Similarity Score (BERT)
- 🏆 Achievement Detection
- 📚 Skill Coverage

Final score is a weighted combination of all factors.

---

## 🛠 Tech Stack

### Frontend
- React
- CSS

### Backend
- Flask
- spaCy
- Sentence Transformers (BERT)
- scikit-learn

---

## 📸 Demo



---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/resume-analyzer.git
cd resume-analyzer
