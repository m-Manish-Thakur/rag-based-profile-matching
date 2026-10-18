"""
Job Matcher Engine — Part B
============================
Accepts a job description, retrieves top-K similar resumes (semantic +
keyword hybrid search), scores them 0–100, and produces structured output
with match reasoning.

Usage:
    python job_matcher.py --jd data/job_descriptions/senior_python_engineer.txt
    python job_matcher.py --jd-text "We need a Python engineer with 5+ years..." \
                           --must-have Python Docker --top-k 5
"""

import re
import json
import time
import argparse
from pathlib import Path
from typing import Optional

from resume_rag import ResumeRAG, CANONICAL_SKILLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_K = 10

# Weights for the composite match score
W_SEMANTIC = 0.45
W_SKILLS = 0.35
W_MUST_HAVE = 0.20


# ---------------------------------------------------------------------------
# Job Description Parsing
# ---------------------------------------------------------------------------

def parse_jd_skills(jd_text: str) -> list[str]:
    """Extract canonical skills mentioned in a job description text."""
    text_lower = jd_text.lower()
    found = []
    for skill in CANONICAL_SKILLS:
        if re.search(re.escape(skill.lower()), text_lower):
            found.append(skill)
    return sorted(found)


def parse_jd_min_years(jd_text: str) -> Optional[float]:
    """Try to extract a minimum years-of-experience requirement."""
    # Look for patterns like "5+ years", "3 or more years", "at least 4 years"
    patterns = [
        r"(\d+)\s*\+?\s*years",
        r"at\s+least\s+(\d+)\s*years",
        r"(\d+)\s*or\s+more\s*years",
    ]
    for pat in patterns:
        match = re.search(pat, jd_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Hybrid scoring
# ---------------------------------------------------------------------------

def keyword_overlap(query_skills: list[str], candidate_skills: list[str]) -> float:
    """Simple keyword overlap ratio: |intersection| / |query|."""
    if not query_skills:
        return 0.0
    q = {s.lower() for s in query_skills}
    c = {s.lower() for s in candidate_skills}
    return len(q & c) / len(q)


def compute_match_score(
    semantic_score: float,
    matched_skills: list[str],
    required_skills: list[str],
    must_have_skills: list[str],
) -> tuple[int, dict]:
    """Return (final_score 0-100, breakdown dict).

    Components
    ----------
    semantic_score : float in [0, 1]
        Cosine similarity between JD embedding and resume chunks.
    matched_skills : list[str]
        Skills the candidate has that overlap with the JD.
    required_skills : list[str]
        All skills extracted from the JD.
    must_have_skills : list[str]
        Skills that are non-negotiable.
    """
    semantic_component = semantic_score * 100

    if required_skills:
        skill_ratio = len(matched_skills) / len(required_skills)
    else:
        skill_ratio = 0.3  # neutral fallback when no skills detected
    skill_component = skill_ratio * 100

    # Must-have compliance
    if must_have_skills:
        matched_lower = {s.lower() for s in matched_skills}
        missing = [s for s in must_have_skills if s.lower() not in matched_lower]
        must_have_ratio = 1.0 - (len(missing) / len(must_have_skills))
    else:
        missing = []
        must_have_ratio = 1.0
    must_have_component = must_have_ratio * 100

    final = (
        W_SEMANTIC * semantic_component
        + W_SKILLS * skill_component
        + W_MUST_HAVE * must_have_component
    )
    final = max(0, min(100, round(final)))

    breakdown = {
        "semantic_component": round(semantic_component, 1),
        "skill_component": round(skill_component, 1),
        "must_have_component": round(must_have_component, 1),
        "missing_must_have": missing,
    }
    return final, breakdown


def build_reasoning(
    candidate_name: str,
    matched_skills: list[str],
    required_skills: list[str],
    must_have_skills: list[str],
    breakdown: dict,
    exp_years: float,
    min_years: Optional[float],
    top_sections: list[str],
) -> str:
    """Generate a human-readable match explanation."""
    parts = []

    # Skill match summary
    if required_skills:
        ratio = len(matched_skills) / len(required_skills)
        parts.append(
            f"Matched {len(matched_skills)}/{len(required_skills)} required skills "
            f"({ratio:.0%})."
        )

    # Must-have check
    if breakdown["missing_must_have"]:
        parts.append(
            f"Missing must-have skills: {', '.join(breakdown['missing_must_have'])}."
        )
    elif must_have_skills:
        parts.append("All must-have skills satisfied.")

    # Semantic similarity
    sim = breakdown["semantic_component"] / 100
    if sim >= 0.80:
        parts.append("Very high semantic similarity with the job description.")
    elif sim >= 0.60:
        parts.append("Good semantic alignment with the job description.")
    elif sim >= 0.40:
        parts.append("Moderate semantic overlap — some relevant experience.")
    else:
        parts.append("Low semantic similarity — may not be the strongest fit.")

    # Experience fit
    if min_years is not None:
        if exp_years >= min_years:
            parts.append(
                f"Candidate has ~{int(exp_years)} years experience "
                f"(requirement: {int(min_years)}+)."
            )
        else:
            parts.append(
                f"Candidate has ~{int(exp_years)} years experience — "
                f"below the {int(min_years)}+ year requirement."
            )

    # Which sections were most relevant
    if top_sections:
        parts.append(f"Strongest sections: {', '.join(top_sections[:3])}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# JobMatcher
# ---------------------------------------------------------------------------

class JobMatcher:
    """Wraps ResumeRAG and adds hybrid search + scoring + output formatting."""

    def __init__(self, rag: Optional[ResumeRAG] = None):
        self.rag = rag or ResumeRAG()

    def match(
        self,
        job_description: str,
        top_k: int = TOP_K,
        must_have_skills: Optional[list[str]] = None,
        min_years: Optional[float] = None,
    ) -> dict:
        """Run the full matching pipeline.

        Parameters
        ----------
        job_description : str
            Raw JD text.
        top_k : int
            How many candidates to return.
        must_have_skills : list[str], optional
            Skills that candidates must have (filter / heavy penalty).
        min_years : float, optional
            Minimum years of experience. Candidates below this get a
            large score penalty but are not removed.

        Returns
        -------
        dict  matching the required output format.
        """
        t0 = time.time()

        # --- Parse JD ---
        required_skills = parse_jd_skills(job_description)
        if min_years is None:
            min_years = parse_jd_min_years(job_description)
        must_have = must_have_skills or []

        # --- Step 1: Semantic retrieval (over-fetch for re-ranking) ---
        fetch_k = max(top_k * 3, 30)
        semantic_results = self.rag.search(job_description, top_k=fetch_k)

        # --- Step 2: Aggregate chunk-level results into candidate-level ---
        candidates: dict[str, dict] = {}
        for r in semantic_results:
            name = r["candidate_name"]
            if name not in candidates:
                candidates[name] = {
                    "candidate_name": name,
                    "resume_path": r["resume_path"],
                    "best_score": r["score"],
                    "chunks": [],
                    "sections": [],
                    "skills_str": r["skills"],
                    "experience_years": r["experience_years"],
                }
            candidates[name]["chunks"].append(r)
            candidates[name]["sections"].append(r["section"])
            # Keep the highest-scoring chunk as the representative
            if r["score"] > candidates[name]["best_score"]:
                candidates[name]["best_score"] = r["score"]

        # --- Step 3: Hybrid scoring ---
        scored: list[dict] = []
        for cand in candidates.values():
            cand_skills = [s.strip() for s in cand["skills_str"].split(",") if s.strip()]

            # Semantic component: average of top 3 chunk scores for this candidate
            top_chunk_scores = sorted(
                (c["score"] for c in cand["chunks"]), reverse=True
            )[:3]
            avg_semantic = sum(top_chunk_scores) / len(top_chunk_scores) if top_chunk_scores else 0

            matched = [s for s in required_skills if s.lower() in {cs.lower() for cs in cand_skills}]

            score, breakdown = compute_match_score(
                semantic_score=avg_semantic,
                matched_skills=matched,
                required_skills=required_skills,
                must_have_skills=must_have,
            )

            # Experience penalty if below minimum
            if min_years and cand["experience_years"] < min_years:
                score = max(0, score - 15)

            # Most relevant sections (dedup, preserve order by frequency)
            from collections import Counter
            section_counts = Counter(cand["sections"])
            top_sections = [s for s, _ in section_counts.most_common(3)]

            # Relevant excerpts: top 2 chunk previews
            excerpts = [
                c["chunk_text"][:250] + ("..." if len(c["chunk_text"]) > 250 else "")
                for c in cand["chunks"][:2]
            ]

            reasoning = build_reasoning(
                candidate_name=cand["candidate_name"],
                matched_skills=matched,
                required_skills=required_skills,
                must_have_skills=must_have,
                breakdown=breakdown,
                exp_years=cand["experience_years"],
                min_years=min_years,
                top_sections=top_sections,
            )

            scored.append({
                "candidate_name": cand["candidate_name"],
                "resume_path": cand["resume_path"],
                "match_score": score,
                "matched_skills": matched,
                "relevant_excerpts": excerpts,
                "reasoning": reasoning,
                "_breakdown": breakdown,
                "_semantic_avg": round(avg_semantic, 4),
                "_experience_years": cand["experience_years"],
            })

        # --- Step 4: Hard filter on must-have (optional) ---
        if must_have:
            filtered = []
            for s in scored:
                if not s["_breakdown"]["missing_must_have"]:
                    filtered.append(s)
            # If filtering removed everything, fall back to unfiltered
            # but keep the penalty already applied.
            if filtered:
                scored = filtered

        # --- Step 5: Sort by score descending, take top_k ---
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        top_matches = scored[:top_k]

        elapsed = time.time() - t0

        # --- Build output ---
        output = {
            "job_description": job_description[:500] + ("..." if len(job_description) > 500 else ""),
            "search_time_s": round(elapsed, 3),
            "required_skills_detected": required_skills,
            "min_years_detected": min_years,
            "top_matches": [
                {
                    "candidate_name": m["candidate_name"],
                    "resume_path": m["resume_path"],
                    "match_score": m["match_score"],
                    "matched_skills": m["matched_skills"],
                    "relevant_excerpts": m["relevant_excerpts"],
                    "reasoning": m["reasoning"],
                }
                for m in top_matches
            ],
        }
        return output

    def match_from_file(
        self,
        jd_path: str,
        top_k: int = TOP_K,
        must_have_skills: Optional[list[str]] = None,
        min_years: Optional[float] = None,
    ) -> dict:
        """Convenience wrapper that reads a JD file from disk."""
        text = Path(jd_path).read_text(encoding="utf-8")
        return self.match(text, top_k=top_k, must_have_skills=must_have_skills, min_years=min_years)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_output(output: dict) -> None:
    """Pretty-print the match result to the terminal."""
    print("\n" + "=" * 70)
    print("  JOB MATCH RESULTS")
    print("=" * 70)
    print(f"  JD snippet : {output['job_description'][:100]}...")
    print(f"  Skills req : {output['required_skills_detected']}")
    print(f"  Min years  : {output['min_years_detected']}")
    print(f"  Time       : {output['search_time_s']}s")
    print(f"  Matches    : {len(output['top_matches'])}")
    print("=" * 70)

    for i, m in enumerate(output["top_matches"], 1):
        print(f"\n  #{i}  {m['candidate_name']}  —  Score: {m['match_score']}/100")
        print(f"      Path     : {m['resume_path']}")
        print(f"      Skills   : {', '.join(m['matched_skills'])}")
        print(f"      Reasoning: {m['reasoning']}")
        if m["relevant_excerpts"]:
            print(f"      Excerpt  : {m['relevant_excerpts'][0][:120]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match resumes against a job description")
    parser.add_argument("--jd", type=str, help="Path to a job description .txt file")
    parser.add_argument("--jd-text", type=str, help="Job description as a raw string")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--must-have", nargs="*", default=[], help="Must-have skills")
    parser.add_argument("--min-years", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--out", type=str, default=None, help="Write JSON to file")
    args = parser.parse_args()

    if not args.jd and not args.jd_text:
        parser.error("Provide --jd <file> or --jd-text <string>")

    matcher = JobMatcher()

    if args.jd:
        result = matcher.match_from_file(
            args.jd, top_k=args.top_k,
            must_have_skills=args.must_have, min_years=args.min_years,
        )
    else:
        result = matcher.match(
            args.jd_text, top_k=args.top_k,
            must_have_skills=args.must_have, min_years=args.min_years,
        )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_output(result)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[saved] JSON written to {args.out}")
