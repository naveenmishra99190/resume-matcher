

# 🚀 AI Resume Matching System

An **AI-powered resume screening system** that intelligently matches resumes with job descriptions using **state-of-the-art NLP models**. It ranks candidates, highlights skill matches, and provides actionable recommendations in real time.

## ✨ Features

* 📄 Supports **PDF & DOCX** resumes
* 🤖 **Multi-model AI matching** (SBERT, GloVe, Doc2Vec)
* 📊 Automatic **candidate ranking**
* 🧠 Semantic + keyword-based analysis
* ⚡ Real-time batch processing
* 📱 Responsive web UI
* 📤 Export results as JSON


## 🧠 AI Models Used

* **SBERT (all-MiniLM-L6-v2)** – Semantic document similarity
* **GloVe (glove-wiki-gigaword-100)** – Keyword & word-level matching
* **Doc2Vec** – Document structure & context understanding

Final score = average of all three models.

---

## 🛠 Tech Stack

**Backend**

* Flask, Flask-CORS
* Python, NumPy, SciPy, scikit-learn
* NLTK, Gensim, Sentence-Transformers
* PyPDF2, python-docx

**Frontend**

* HTML5, CSS3
* Vanilla JavaScript (Fetch API, Drag & Drop)

**DevOps**

* Docker, Docker Compose
* Git

---

## 🔄 How It Works

1. User uploads resumes & job description
2. Text extracted from PDF/DOCX
3. NLP preprocessing (tokenization, stopwords removal)
4. Similarity computed using 3 AI models
5. Scores aggregated & candidates ranked
6. Skills matched & recommendations generated
7. Results returned as JSON and displayed in UI

