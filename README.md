# Resume RAG System

A Retrieval-Augmented Generation (RAG) system for intelligent resume–job matching.

## Project Structure

```
resume-rag-system/
├── data/
│   ├── resumes/            # 35 synthetic resumes (.txt)
│   └── job_descriptions/   # 6 job descriptions (.txt)
├── chroma_db/              # Vector database (created at runtime)
├── generate_dataset.py     # Generates the synthetic dataset
├── resume_rag.py           # Part A — RAG pipeline (load → chunk → embed → store → search)
├── job_matcher.py          # Part B — hybrid search + scoring engine
├── analysis.ipynb          # Jupyter notebook: experiments, metrics, analysis
├── .env.example            # Environment variable template
├── requirements.txt
└── README.md
```

## Environment

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

| Variable | Required? | Default | What it does |
|----------|-----------|---------|--------------|
| `HF_TOKEN` | No | — | HuggingFace token; avoids rate-limit warnings when downloading models |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Sentence-transformers model used for embeddings |
| `CHROMA_DIR` | No | `chroma_db` | Where ChromaDB persists the vector store |
| `OPENAI_API_KEY` | No | — | Only needed if you swap code to OpenAI embeddings |

> **All variables are optional.** The project runs fully offline with defaults —
> no API keys needed. `.env` is loaded automatically at startup.

### Python

- Python **3.10+** (tested on 3.14)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python generate_dataset.py
```

This creates 35 diverse resumes and 6 job descriptions under `data/`.

### 3. Index the resumes

```bash
python resume_rag.py --index
```

### 4. Search

```bash
python resume_rag.py --search "machine learning engineer with TensorFlow experience" --top-k 5
```

### 5. Match a job description

```bash
python job_matcher.py --jd data/job_descriptions/senior_python_engineer.txt --top-k 10
```

Filter by must-have skills:

```bash
python job_matcher.py --jd data/job_descriptions/ml_engineer.txt --must-have Python TensorFlow --min-years 3
```

Export JSON:

```bash
python job_matcher.py --jd data/job_descriptions/data_analyst.txt --json --out results.json
```

## Output Format

```json
{
  "job_description": "...",
  "top_matches": [
    {
      "candidate_name": "John Doe",
      "resume_path": "data/resumes/john_doe.txt",
      "match_score": 92,
      "matched_skills": ["Python", "Machine Learning"],
      "relevant_excerpts": ["..."],
      "reasoning": "Strong match for ML experience..."
    }
  ]
}
```

## How It Works

| Step | Module | What happens |
|------|--------|-------------|
| Load | `resume_rag.py` | Reads `.txt` / `.pdf` files from disk |
| Chunk | `resume_rag.py` | Splits by section headers (EXPERIENCE, EDUCATION, SKILLS…) |
| Embed | `resume_rag.py` | `all-MiniLM-L6-v2` sentence-transformer (local, free) |
| Store | `resume_rag.py` | ChromaDB persistent vector store |
| Retrieve | `job_matcher.py` | Semantic search (cosine similarity), top-K |
| Hybrid | `job_matcher.py` | Adds keyword overlap for critical skills |
| Score | `job_matcher.py` | Weighted composite: semantic (45%) + skills (35%) + must-have (20%) |
| Filter | `job_matcher.py` | Hard-filters candidates missing must-have skills |

## Notebook

Open `analysis.ipynb` for:

- Data exploration
- Indexing & retrieval experiments
- Latency and retrieval-accuracy metrics
- Side-by-side comparison of multiple job descriptions
