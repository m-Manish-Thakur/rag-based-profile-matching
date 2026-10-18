"""
Resume RAG System — Part A
==========================
Document loading → intelligent chunking → metadata extraction →
embeddings → vector database (ChromaDB).

Usage as a library:
    rag = ResumeRAG()
    rag.index_resumes("data/resumes")
    results = rag.search("Python machine learning engineer", top_k=5)
"""

import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # load .env if present

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration  (override via .env — see .env.example)
# ---------------------------------------------------------------------------

RESUMES_DIR = "data/resumes"
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = "resumes"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = 10

# Canonical skill vocabulary used for consistent matching between
# resumes and job descriptions.
CANONICAL_SKILLS = {
    # Languages
    "Python", "Java", "JavaScript", "TypeScript", "Go", "C++", "R",
    "Ruby", "PHP", "Swift", "Kotlin", "Rust", "Bash",
    # Frontend
    "React", "Angular", "Vue.js", "Next.js", "HTML", "CSS", "Sass",
    "Tailwind CSS", "Webpack", "Figma",
    # Backend / APIs
    "Node.js", "Express.js", "GraphQL", "REST APIs", "Microservices",
    # Data
    "SQL", "PostgreSQL", "MongoDB", "Redis", "Snowflake", "BigQuery",
    "Pandas", "NumPy", "Spark", "Hadoop", "Airflow", "dbt", "ETL",
    "Data Modeling", "Data Visualization", "Tableau", "Power BI", "Looker",
    "Excel",
    # ML / AI
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
    "scikit-learn", "NLP", "Computer Vision", "Statistics", "A/B Testing",
    "MLOps", "Model Deployment",
    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "CI/CD",
    "Terraform", "Ansible", "ArgoCD", "Helm", "CloudFormation",
    "Linux", "Monitoring", "Prometheus", "Grafana", "Security", "Networking",
    # Practices / Soft
    "Git", "Agile", "Scrum", "Leadership", "Communication",
    "Problem Solving", "Team Leadership", "Team Management",
    "Presentation Skills", "Business Acumen", "Accessibility",
    "Performance Optimization", "Testing", "Jest", "Responsive Design",
    "Cloud Infrastructure",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ResumeMetadata:
    """Structured metadata extracted from a resume."""
    candidate_name: str
    email: str
    phone: str
    skills: list[str]
    experience_years: float
    education: str
    resume_path: str


@dataclass
class ResumeChunk:
    """A single chunk of a resume, ready for embedding."""
    text: str
    metadata: dict
    chunk_id: str


# ---------------------------------------------------------------------------
# 1. Document Loading
# ---------------------------------------------------------------------------

def load_resume_files(resumes_dir: str) -> list[dict]:
    """Load every .txt / .pdf file from *resumes_dir*.

    Returns a list of ``{"path": str, "text": str}`` dicts.
    """
    resumes = []
    path = Path(resumes_dir)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {resumes_dir}")

    for file in sorted(path.iterdir()):
        if file.suffix.lower() == ".txt":
            resumes.append({"path": str(file), "text": file.read_text(encoding="utf-8")})
        elif file.suffix.lower() == ".pdf":
            resumes.append({"path": str(file), "text": _read_pdf(file)})
        # skip everything else silently

    print(f"[load] Loaded {len(resumes)} resume files from {resumes_dir}")
    return resumes


def _read_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF using pdfplumber (optional dependency)."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Install pdfplumber to read PDFs:  pip install pdfplumber")
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# 2. Intelligent Chunking  (section-aware)
# ---------------------------------------------------------------------------

# Regex that matches known section headers (case-insensitive, own line)
_SECTION_RE = re.compile(
    r"^(SUMMARY|EXPERIENCE|EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS"
    r"|PROFESSIONAL SUMMARY|WORK EXPERIENCE|TECHNICAL SKILLS"
    r"|WORK HISTORY|QUALIFICATIONS)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def chunk_resume(text: str, resume_path: str) -> list[ResumeChunk]:
    """Split a resume into section-based chunks.

    Strategy
    --------
    1. Detect section headers with regex.
    2. Each section becomes one chunk (EXPERIENCE is split per job entry).
    3. Prepend candidate name + section label to every chunk so the
       embedding captures context.
    """
    lines = text.splitlines()

    # --- Find candidate name (first non-empty line) ---
    candidate_name = ""
    for ln in lines:
        if ln.strip():
            candidate_name = ln.strip()
            break

    # --- Locate section boundaries ---
    sections: list[tuple[str, list[str]]] = []
    current_label = "PREAMBLE"
    current_lines: list[str] = []

    for line in lines:
        if _SECTION_RE.match(line.strip()):
            if current_lines:
                sections.append((current_label, current_lines))
            current_label = line.strip().upper()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_label, current_lines))

    # --- Build chunks ---
    chunks: list[ResumeChunk] = []
    chunk_counter = 0

    for label, sec_lines in sections:
        section_text = "\n".join(sec_lines).strip()
        if not section_text:
            continue

        if label in ("EXPERIENCE", "WORK EXPERIENCE", "WORK HISTORY"):
            # Split each experience entry into its own chunk
            sub_chunks = _split_experience(section_text)
        else:
            sub_chunks = [section_text]

        for sub in sub_chunks:
            chunk_text = f"Candidate: {candidate_name}\nSection: {label}\n{sub}"
            meta = {
                "candidate_name": candidate_name,
                "section": label,
                "resume_path": resume_path,
                "text_preview": sub[:200],
            }
            chunks.append(ResumeChunk(
                text=chunk_text,
                metadata=meta,
                chunk_id=f"{Path(resume_path).stem}::chunk_{chunk_counter}",
            ))
            chunk_counter += 1

    return chunks


def _split_experience(section_text: str) -> list[str]:
    """Split an EXPERIENCE section into individual job entries.

    A new entry starts when a line matches the pattern
    ``Title | Company | Dates``.
    """
    lines = section_text.splitlines()
    entries: list[str] = []
    current: list[str] = []

    # Pattern: lines that look like "Title | Company | 2020 - 2023"
    entry_header = re.compile(r"\s*\|\s*.*\|\s*\d{4}")

    for line in lines:
        if entry_header.search(line) and current:
            entries.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append("\n".join(current).strip())

    return entries if entries else [section_text]


# ---------------------------------------------------------------------------
# 3. Metadata Extraction
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
_YEAR_RANGE_RE = re.compile(r"(\d{4})\s*[-–]\s*(\d{4}|Present|Current)", re.IGNORECASE)


def extract_metadata(text: str, resume_path: str) -> ResumeMetadata:
    """Pull structured fields out of raw resume text."""

    lines = text.splitlines()
    candidate_name = ""
    for ln in lines:
        stripped = ln.strip()
        if stripped and not stripped.upper().startswith(("SUMMARY", "EMAIL", "PHONE")):
            candidate_name = stripped
            break

    email_match = _EMAIL_RE.search(text)
    phone_match = _PHONE_RE.search(text)

    # --- Skills ---
    skills = _extract_skills(text)

    # --- Experience years ---
    exp_years = _estimate_experience_years(text)

    # --- Education ---
    education = _extract_education(text)

    return ResumeMetadata(
        candidate_name=candidate_name,
        email=email_match.group() if email_match else "",
        phone=phone_match.group() if phone_match else "",
        skills=skills,
        experience_years=exp_years,
        education=education,
        resume_path=resume_path,
    )


def _extract_skills(text: str) -> list[str]:
    """Find canonical skills mentioned anywhere in the resume text."""
    text_lower = text.lower()
    found = []
    for skill in CANONICAL_SKILLS:
        # Word-boundary-ish match (handle multi-word and punctuation skills)
        pattern = re.escape(skill.lower())
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(found)


def _estimate_experience_years(text: str) -> float:
    """Estimate total years of professional experience from date ranges."""
    ranges = _YEAR_RANGE_RE.findall(text)
    if not ranges:
        return 0.0

    total = 0
    seen: set[tuple[str, str]] = set()
    for start, end in ranges:
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        start_y = int(start)
        end_y = 2025 if end.lower() in ("present", "current") else int(end)
        if end_y >= start_y:
            total += end_y - start_y

    # Heuristic: cap at a reasonable number; also take the max of
    # summed ranges vs. span from earliest to latest (handles overlapping).
    if ranges:
        starts = [int(s) for s, _ in ranges]
        ends = [2025 if e.lower() in ("present", "current") else int(e) for _, e in ranges]
        span = max(ends) - min(starts)
        total = max(total, span) if span > 0 else total

    return float(total)


def _extract_education(text: str) -> str:
    """Return a short education summary string."""
    degree_re = re.compile(
        r"(Bachelor|Master|PhD|Doctorate|B\.S\.|M\.S\.|B\.A\.|M\.A\.|M\.Eng\.)"
        r"[^\n]*",
        re.IGNORECASE,
    )
    match = degree_re.search(text)
    if match:
        # Grab the full line
        line = match.group().strip()
        # Try to append university from the next non-empty line
        after = text[match.end():].strip().splitlines()
        for candidate in after:
            c = candidate.strip()
            if c and not c.upper().startswith(("SKILLS", "PROJECTS")):
                return f"{line} — {c}" if c else line
        return line
    return ""


# ---------------------------------------------------------------------------
# 4. ResumeRAG — ties everything together
# ---------------------------------------------------------------------------

class ResumeRAG:
    """Full RAG pipeline: load → chunk → embed → store → search."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.embedding_model = self._load_model(model_name)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Cache of resume-level metadata, keyed by candidate name
        self._resume_meta: dict[str, ResumeMetadata] = {}

    @staticmethod
    def _load_model(model_name: str, retries: int = 3) -> SentenceTransformer:
        """Load the embedding model with retries (handles flaky downloads)."""
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                return SentenceTransformer(model_name)
            except Exception as exc:
                last_err = exc
                print(f"[model] Attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries:
                    time.sleep(2 * attempt)
        raise RuntimeError(
            f"Could not load embedding model '{model_name}' after {retries} attempts. "
            f"Check your network connection or set HF_TOKEN in .env.\nLast error: {last_err}"
        )

    # ---- indexing --------------------------------------------------------

    def index_resumes(self, resumes_dir: str = RESUMES_DIR) -> dict:
        """Run the full pipeline over every file in *resumes_dir*."""
        t0 = time.time()
        files = load_resume_files(resumes_dir)

        all_ids: list[str] = []
        all_texts: list[str] = []
        all_metadatas: list[dict] = []

        for f in files:
            # Chunk
            chunks = chunk_resume(f["text"], f["path"])

            # Resume-level metadata (extracted once per file)
            meta = extract_metadata(f["text"], f["path"])
            self._resume_meta[meta.candidate_name] = meta

            for chunk in chunks:
                # Merge chunk metadata with resume-level fields
                md = {
                    **chunk.metadata,
                    "skills": ", ".join(meta.skills),
                    "experience_years": meta.experience_years,
                    "education": meta.education,
                    "email": meta.email,
                }
                all_ids.append(chunk.chunk_id)
                all_texts.append(chunk.text)
                all_metadatas.append(md)

        # Batch-encode and upsert into ChromaDB
        if all_ids:
            embeddings = self.embedding_model.encode(
                all_texts, show_progress_bar=False, convert_to_numpy=True
            ).tolist()

            # ChromaDB handles batching internally for large corpora,
            # but we upsert in batches of 500 to be safe.
            batch = 500
            for i in range(0, len(all_ids), batch):
                self.collection.upsert(
                    ids=all_ids[i : i + batch],
                    embeddings=embeddings[i : i + batch],
                    documents=all_texts[i : i + batch],
                    metadatas=all_metadatas[i : i + batch],
                )

        elapsed = time.time() - t0
        stats = {
            "resumes_indexed": len(files),
            "chunks_indexed": len(all_ids),
            "indexing_time_s": round(elapsed, 2),
        }
        print(f"[index] {stats}")
        return stats

    # ---- search ----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic search over indexed resume chunks.

        Parameters
        ----------
        query : str
            Natural-language query.
        top_k : int
            Number of results to return.
        where : dict, optional
            ChromaDB metadata filter, e.g.
            ``{"experience_years": {"$gte": 5}}``.

        Returns
        -------
        list of dicts with keys ``chunk_text``, ``candidate_name``,
        ``section``, ``resume_path``, ``score``.
        """
        query_emb = self.embedding_model.encode(
            [query], convert_to_numpy=True
        ).tolist()

        kwargs = dict(
            query_embeddings=query_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        formatted: list[dict] = []
        for i in range(len(results["ids"][0])):
            md = results["metadatas"][0][i]
            formatted.append({
                "chunk_id": results["ids"][0][i],
                "chunk_text": results["documents"][0][i],
                "candidate_name": md.get("candidate_name", ""),
                "section": md.get("section", ""),
                "resume_path": md.get("resume_path", ""),
                "skills": md.get("skills", ""),
                "experience_years": md.get("experience_years", 0),
                "distance": results["distances"][0][i],
                # Cosine similarity = 1 - cosine distance
                "score": round(1 - results["distances"][0][i], 4),
            })
        return formatted

    # ---- helpers ---------------------------------------------------------

    def get_resume_metadata(self, candidate_name: str) -> Optional[ResumeMetadata]:
        return self._resume_meta.get(candidate_name)

    def list_candidate_names(self) -> list[str]:
        return sorted(self._resume_meta.keys())

    def reset_collection(self):
        """Drop and recreate the collection (useful during development)."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._resume_meta.clear()
        print("[reset] Collection cleared.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resume RAG — index & search")
    parser.add_argument("--index", action="store_true", help="Index resumes")
    parser.add_argument("--search", type=str, default=None, help="Search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--resumes-dir", type=str, default=RESUMES_DIR)
    args = parser.parse_args()

    rag = ResumeRAG()

    if args.index:
        rag.index_resumes(args.resumes_dir)

    if args.search:
        results = rag.search(args.search, top_k=args.top_k)
        for r in results:
            print(f"\n{'='*60}")
            print(f"  {r['candidate_name']}  |  Section: {r['section']}")
            print(f"  Score: {r['score']:.4f}  |  Path: {r['resume_path']}")
            print(f"  Preview: {r['chunk_text'][:150]}...")
