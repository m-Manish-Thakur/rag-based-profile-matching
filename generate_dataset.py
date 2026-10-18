"""
Generate a synthetic dataset of 35 diverse resumes and 6 job descriptions.
Run once:  python generate_dataset.py
"""

import os
import random
import re
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JOBS_DIR = DATA_DIR / "job_descriptions"

# ---------------------------------------------------------------------------
# Pools of realistic data
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Carlos", "Aisha", "Wei",
    "Priya", "Ahmed", "Yuki", "Olga", "Raj", "Fatima", "Hans", "Mei",
    "Dmitri", "Sofia", "Arjun", "Elena", "Kwame", "Ingrid",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Chen", "Patel",
    "Kim", "Singh", "Kowalski", "Nakamura", "Ivanov", "Okafor", "Muller",
    "Rossi", "Silva", "Nguyen", "Ali", "Yamamoto", "Petrov", "Johansson",
    "Ali", "Sharma", "Costa", "Novak", "Berg", "Diallo", "Horvat",
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "protonmail.com"]

COMPANIES = [
    "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix", "Stripe",
    "Shopify", "Airbnb", "Uber", "Lyft", "Snowflake", "Databricks",
    "Palantir", "Salesforce", "Adobe", "Intel", "NVIDIA", "IBM", "Oracle",
    "StartupXYZ", "TechCorp", "DataFlow Inc", "CloudBase", "AI Solutions Ltd",
    "Bright Analytics", "NextGen Software", "Quantum Labs", "DevBridge",
    "CodeCraft", "ByteForge", "CloudNine Systems", "DataPulse", "ScaleOps",
]

UNIVERSITIES = [
    "MIT", "Stanford University", "Carnegie Mellon University",
    "UC Berkeley", "Georgia Tech", "University of Washington",
    "University of Illinois", "Caltech", "Columbia University",
    "NYU", "University of Michigan", "Texas A&M", "Purdue University",
    "University of Texas at Austin", "Cornell University", "Duke University",
    "University of Waterloo", "ETH Zurich", "University of Toronto",
    "IIT Delhi", "University of Oxford", "TU Munich", "Seoul National University",
]

DEGREES = ["Bachelor of Science", "Master of Science", "Bachelor of Arts", "Master of Engineering"]

# ---------------------------------------------------------------------------
# Role definitions — each role has a skill pool and title pool
# ---------------------------------------------------------------------------

ROLES = {
    "software_engineer": {
        "titles": [
            "Software Engineer", "Senior Software Engineer",
            "Backend Software Engineer", "Full Stack Developer",
            "Software Developer", "Lead Software Engineer",
        ],
        "skills": [
            "Python", "Java", "JavaScript", "TypeScript", "React", "Node.js",
            "SQL", "PostgreSQL", "MongoDB", "REST APIs", "Git", "Docker",
            "AWS", "CI/CD", "Microservices", "GraphQL", "Redis",
            "Express.js", "Angular", "Vue.js", "Go", "C++", "Kubernetes",
            "Agile", "Scrum", "Problem Solving", "Team Leadership",
        ],
        "certifications": [
            "AWS Certified Developer", "Oracle Certified Professional",
            "Google Cloud Professional Developer", "Certified Kubernetes Administrator",
        ],
    },
    "data_scientist": {
        "titles": [
            "Data Scientist", "Senior Data Scientist",
            "Lead Data Scientist", "Applied Data Scientist",
            "Research Data Scientist",
        ],
        "skills": [
            "Python", "R", "SQL", "Machine Learning", "Deep Learning",
            "Pandas", "NumPy", "scikit-learn", "TensorFlow", "PyTorch",
            "NLP", "Computer Vision", "Statistics", "Data Visualization",
            "Tableau", "Power BI", "Spark", "Hadoop", "Airflow",
            "A/B Testing", "Git", "AWS", "Azure", "Communication",
            "Problem Solving", "Leadership", "Docker",
        ],
        "certifications": [
            "Google Data Analytics Certificate", "AWS Machine Learning Specialty",
            "TensorFlow Developer Certificate", "Microsoft Certified: Data Scientist",
        ],
    },
    "ml_engineer": {
        "titles": [
            "Machine Learning Engineer", "Senior ML Engineer",
            "ML Platform Engineer", "AI Engineer", "Deep Learning Engineer",
        ],
        "skills": [
            "Python", "Machine Learning", "Deep Learning", "TensorFlow",
            "PyTorch", "scikit-learn", "NLP", "Computer Vision",
            "MLOps", "Docker", "Kubernetes", "AWS", "GCP",
            "CI/CD", "REST APIs", "SQL", "Pandas", "NumPy",
            "Git", "Model Deployment", "Spark", "Hadoop",
            "Leadership", "Communication", "Problem Solving", "Agile",
        ],
        "certifications": [
            "AWS Machine Learning Specialty", "TensorFlow Developer Certificate",
            "Google Professional Machine Learning Engineer",
        ],
    },
    "devops_engineer": {
        "titles": [
            "DevOps Engineer", "Senior DevOps Engineer",
            "Site Reliability Engineer", "Cloud Infrastructure Engineer",
            "Platform Engineer",
        ],
        "skills": [
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins",
            "CI/CD", "Terraform", "Ansible", "Python", "Bash", "Linux",
            "Git", "Monitoring", "Prometheus", "Grafana", "ArgoCD",
            "Microservices", "Networking", "Security", "Agile",
            "Communication", "Problem Solving", "Team Leadership",
            "CloudFormation", "Helm",
        ],
        "certifications": [
            "AWS Certified DevOps Engineer", "Certified Kubernetes Administrator",
            "HashiCorp Terraform Associate", "Azure DevOps Engineer Expert",
        ],
    },
    "data_analyst": {
        "titles": [
            "Data Analyst", "Senior Data Analyst",
            "Business Intelligence Analyst", "Analytics Engineer",
            "Reporting Analyst",
        ],
        "skills": [
            "SQL", "Python", "Excel", "Tableau", "Power BI",
            "Data Visualization", "Statistics", "Pandas", "NumPy",
            "R", "ETL", "Airflow", "Snowflake", "Data Modeling",
            "A/B Testing", "Communication", "Problem Solving",
            "Git", "Looker", "dbt", "BigQuery", "Excel",
            "Presentation Skills", "Business Acumen",
        ],
        "certifications": [
            "Google Data Analytics Certificate", "Microsoft Power BI Data Analyst",
            "Tableau Desktop Specialist", "Certified Analytics Professional",
        ],
    },
    "frontend_engineer": {
        "titles": [
            "Frontend Engineer", "Senior Frontend Developer",
            "UI Engineer", "React Developer", "Web Developer",
        ],
        "skills": [
            "JavaScript", "TypeScript", "React", "Vue.js", "Angular",
            "HTML", "CSS", "Sass", "Webpack", "Git", "REST APIs",
            "Node.js", "Next.js", "Tailwind CSS", "Testing", "Jest",
            "Accessibility", "Performance Optimization", "GraphQL",
            "Agile", "Communication", "Problem Solving", "Figma",
            "Responsive Design",
        ],
        "certifications": [
            "Meta Front-End Developer Certificate",
            "Google Mobile Web Specialist",
        ],
    },
}

# ---------------------------------------------------------------------------
# Experience bullet templates per role
# ---------------------------------------------------------------------------

EXPERIENCE_BULLETS = {
    "software_engineer": [
        "Designed and implemented RESTful APIs serving {n}K+ daily active users",
        "Refactored legacy monolith into microservices reducing latency by {pct}%",
        "Built CI/CD pipeline reducing deployment time from hours to minutes",
        "Led migration from on-premises servers to AWS reducing infrastructure costs by {pct}%",
        "Developed real-time data processing pipeline handling {n}M events/day",
        "Mentored {n} junior developers through code reviews and pairing sessions",
        "Implemented comprehensive test suite increasing code coverage from {pct}% to 95%",
        "Optimized database queries improving response times by {pct}%",
        "Built internal developer tools saving team {n} hours/week",
        "Designed and shipped customer-facing features used by {n}K+ users",
    ],
    "data_scientist": [
        "Built predictive models improving customer churn prediction accuracy by {pct}%",
        "Developed recommendation system increasing user engagement by {pct}%",
        "Designed A/B testing framework for product decisions across {n} teams",
        "Created automated reporting dashboards reducing manual reporting by {pct}%",
        "Conducted deep-dive analysis on customer segments identifying $${n}M in opportunities",
        "Developed NLP pipeline for automated ticket classification with {pct}% accuracy",
        "Built time-series forecasting models for demand planning with {pct}% accuracy",
        "Collaborated with engineering to deploy ML models serving {n}K predictions/day",
        "Performed statistical analysis supporting ${n}M+ strategic decisions",
        "Developed feature engineering pipeline reducing model training time by {pct}%",
    ],
    "ml_engineer": [
        "Designed and deployed ML models serving {n}K+ predictions/second",
        "Built MLOps pipeline for automated model retraining and monitoring",
        "Developed computer vision model achieving {pct}% accuracy on production data",
        "Implemented real-time inference API reducing latency from {n}ms to {pct}ms",
        "Built LLM-powered application processing {n}K documents/day",
        "Created model monitoring system detecting data drift reducing incidents by {pct}%",
        "Optimized model architecture reducing inference cost by {pct}%",
        "Developed distributed training pipeline reducing training time by {n}x",
        "Implemented feature store used by {n} ML teams across the organization",
        "Built automated hyperparameter tuning saving {n} hours per model",
    ],
    "devops_engineer": [
        "Managed Kubernetes clusters with {n}+ microservices across {n} environments",
        "Implemented Infrastructure as Code using Terraform managing {n}+ cloud resources",
        "Reduced mean time to recovery (MTTR) from hours to {pct} minutes",
        "Built automated disaster recovery reducing RTO by {pct}%",
        "Implemented security scanning in CI/CD pipeline catching {n}+ vulnerabilities",
        "Designed multi-region deployment achieving {pct}.9% uptime SLA",
        "Automated server provisioning reducing setup time from days to minutes",
        "Implemented centralized logging and monitoring reducing incident detection time by {pct}%",
        "Managed cloud spending optimization saving $${n}K annually",
        "Built self-service deployment platform used by {n} engineering teams",
    ],
    "data_analyst": [
        "Built executive dashboards in Tableau/Power BI used by {n}+ stakeholders",
        "Developed SQL pipelines processing {n}M+ records daily",
        "Identified $${n}M cost-saving opportunity through customer segmentation analysis",
        "Automated weekly reporting saving {n} hours/week for the team",
        "Conducted cohort analysis improving marketing campaign ROI by {pct}%",
        "Built ETL pipelines consolidating data from {n} source systems",
        "Designed data models supporting self-service analytics for {n} departments",
        "Performed funnel analysis identifying conversion bottleneck improving sign-ups by {pct}%",
        "Created metric definitions and data governance framework",
        "Developed predictive models for inventory optimization reducing waste by {pct}%",
    ],
    "frontend_engineer": [
        "Built responsive web application serving {n}K+ daily active users",
        "Redesigned core user flow increasing conversion by {pct}%",
        "Implemented component library reducing UI development time by {pct}%",
        "Optimized bundle size improving page load time by {pct}%",
        "Built real-time collaborative features using WebSockets",
        "Implemented accessibility improvements achieving WCAG 2.1 AA compliance",
        "Led migration from jQuery to React modernizing codebase",
        "Built animation-rich landing page increasing engagement by {pct}%",
        "Implemented comprehensive testing strategy reducing UI bugs by {pct}%",
        "Developed design system components used across {n} product teams",
    ],
}

PROJECT_TEMPLATES = {
    "software_engineer": [
        ("Open Source CLI Tool", "A command-line tool for automating DevOps workflows, 500+ GitHub stars"),
        ("E-Commerce Platform", "Full-stack e-commerce platform with payment processing and inventory management"),
        ("Chat Application", "Real-time chat app with WebSocket support and message encryption"),
        ("Task Management API", "RESTful API with authentication, rate limiting, and comprehensive documentation"),
        ("Personal Portfolio", "Modern portfolio site built with Next.js and deployed on Vercel"),
    ],
    "data_scientist": [
        ("Customer Churn Predictor", "ML model predicting customer churn with 92% accuracy using XGBoost"),
        ("Sentiment Analysis Dashboard", "NLP pipeline analyzing product reviews with interactive visualizations"),
        ("Stock Price Forecasting", "Time-series model using LSTM for stock price prediction"),
        ("Recommendation Engine", "Collaborative filtering system for personalized content recommendations"),
        ("Automated Report Generator", "NLP-powered tool generating business insights from raw data"),
    ],
    "ml_engineer": [
        ("Image Classification API", "Production-ready image classification API using PyTorch and FastAPI"),
        ("Chatbot with RAG", "LLM-powered chatbot with retrieval-augmented generation"),
        ("Anomaly Detection System", "Real-time anomaly detection for IoT sensor data using autoencoders"),
        ("Model Serving Platform", "Kubernetes-based platform for deploying and monitoring ML models"),
        ("Document Processing Pipeline", "Automated document extraction and classification system"),
    ],
    "devops_engineer": [
        ("Terraform Modules", "Collection of reusable Terraform modules for AWS infrastructure"),
        ("Monitoring Stack", "Complete monitoring solution with Prometheus, Grafana, and alerting"),
        ("CI/CD Pipeline Templates", "Reusable GitHub Actions and Jenkins pipeline templates"),
        ("Kubernetes Operators", "Custom operators for managing stateful applications"),
        ("Security Scanning Tool", "Automated security vulnerability scanner for container images"),
    ],
    "data_analyst": [
        ("Sales Dashboard", "Interactive Power BI dashboard tracking KPIs across regions"),
        ("Customer Segmentation", "K-means clustering analysis revealing 5 distinct customer segments"),
        ("Automated Reporting", "Python-based reporting automation saving 10+ hours weekly"),
        ("A/B Test Analysis Tool", "Statistical analysis toolkit for A/B test evaluation"),
        ("Data Quality Monitor", "Automated data quality checks and alerting system"),
    ],
    "frontend_engineer": [
        ("Component Library", "Open-source React component library with 30+ reusable components"),
        ("Portfolio Website", "Animation-rich portfolio built with React and Framer Motion"),
        ("E-Commerce Frontend", "Responsive shopping experience with cart, checkout, and reviews"),
        ("Real-Time Dashboard", "Interactive data visualization dashboard with live updates"),
        ("Accessibility Browser Extension", "Chrome extension improving web accessibility"),
    ],
}

# ---------------------------------------------------------------------------
# Resume generation
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def generate_phone() -> str:
    return f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"


def generate_experience_entries(role_key: str, years: int) -> list[str]:
    """Generate realistic experience entries for a given role."""
    titles = ROLES[role_key]["titles"]
    bullets = EXPERIENCE_BULLETS[role_key]
    entries = []
    remaining = years
    year_end = 2025

    while remaining > 0:
        chunk = random.randint(1, min(4, remaining))
        title = random.choice(titles)
        company = random.choice(COMPANIES)
        year_start = year_end - chunk
        chosen_bullets = random.sample(bullets, k=min(3, len(bullets)))
        formatted_bullets = []
        for b in chosen_bullets:
            formatted_bullets.append(
                b.replace("{n}", str(random.randint(1, 20)))
                 .replace("{pct}", str(random.randint(15, 80)))
            )
        entry_lines = [f"{title} | {company} | {year_start} - {year_end}"]
        for fb in formatted_bullets:
            entry_lines.append(f"  - {fb}")
        entries.append("\n".join(entry_lines))
        year_end = year_start
        remaining -= chunk

    return entries


def generate_resume(index: int) -> tuple[str, str]:
    """Return (filename, content) for one resume."""
    first = FIRST_NAMES[index % len(FIRST_NAMES)]
    last = LAST_NAMES[index % len(LAST_NAMES)]
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}{random.randint(1, 99)}@{random.choice(EMAIL_DOMAINS)}"
    phone = generate_phone()
    role_key = list(ROLES.keys())[index % len(ROLES)]
    role = ROLES[role_key]
    years = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15])
    num_skills = random.randint(8, min(15, len(role["skills"])))
    skills = random.sample(role["skills"], k=num_skills)
    degree = random.choice(DEGREES)
    university = random.choice(UNIVERSITIES)
    grad_year = 2025 - years - random.randint(0, 2)
    major = random.choice([
        "Computer Science", "Software Engineering", "Information Technology",
        "Data Science", "Mathematics", "Electrical Engineering",
        "Statistics", "Computer Engineering",
    ])
    project_name, project_desc = random.choice(PROJECT_TEMPLATES[role_key])
    cert = random.choice(role["certifications"])
    title = random.choice(role["titles"])

    summary = (
        f"{title} with {years}+ years of experience in software development. "
        f"Skilled in {', '.join(skills[:5])}. "
        f"Passionate about building scalable, high-quality systems."
    )

    experience_entries = generate_experience_entries(role_key, years)

    content = f"""{name}
Email: {email} | Phone: {phone}

SUMMARY
{summary}

EXPERIENCE
{chr(10).join(experience_entries)}

EDUCATION
{degree} in {major}
{university} | {grad_year - random.randint(3, 4)} - {grad_year}

SKILLS
{', '.join(skills)}

PROJECTS
{project_name}
  - {project_desc}

CERTIFICATIONS
{cert}
"""
    filename = f"{slugify(name)}_{index}.txt"
    return filename, content


def generate_job_description(index: int, role_key: str, title: str,
                              required_skills: list[str], nice_skills: list[str],
                              min_years: int) -> str:
    company = random.choice(COMPANIES)
    return f"""JOB TITLE: {title}
COMPANY: {company}

ABOUT THE ROLE
We are looking for a talented {title} to join our growing team at {company}.
You will work on challenging problems and help build products used by millions.

REQUIREMENTS
- {min_years}+ years of professional experience as a {title} or similar role
- Strong proficiency in: {', '.join(required_skills)}
- Experience with: {', '.join(nice_skills)}
- Excellent communication and teamwork skills
- Bachelor's degree in Computer Science or related field

NICE TO HAVE
- Experience with: {', '.join(nice_skills[:3])}
- Contributions to open source projects
- Experience in a startup or fast-paced environment

WHAT WE OFFER
- Competitive salary and equity
- Health, dental, and vision insurance
- Flexible work arrangements
- Professional development budget
"""


def main():
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate 35 resumes
    for i in range(35):
        filename, content = generate_resume(i)
        (RESUMES_DIR / filename).write_text(content, encoding="utf-8")
    print(f"Generated 35 resumes in {RESUMES_DIR}")

    # Generate 6 job descriptions
    jobs = [
        ("senior_python_engineer.txt", "software_engineer", "Senior Python Engineer",
         ["Python", "SQL", "Docker", "AWS", "REST APIs"],
         ["Kubernetes", "Microservices", "Go", "Redis"], 5),
        ("ml_engineer.txt", "ml_engineer", "Machine Learning Engineer",
         ["Python", "Machine Learning", "TensorFlow", "PyTorch", "Docker"],
         ["NLP", "MLOps", "Kubernetes", "AWS"], 3),
        ("data_scientist.txt", "data_scientist", "Senior Data Scientist",
         ["Python", "SQL", "Machine Learning", "Statistics", "Pandas"],
         ["Deep Learning", "NLP", "Spark", "Tableau"], 4),
        ("devops_engineer.txt", "devops_engineer", "Senior DevOps Engineer",
         ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD"],
         ["Python", "Ansible", "Monitoring", "GCP"], 5),
        ("data_analyst.txt", "data_analyst", "Data Analyst",
         ["SQL", "Python", "Excel", "Tableau", "Data Visualization"],
         ["Power BI", "Statistics", "A/B Testing", "Airflow"], 2),
        ("frontend_engineer.txt", "frontend_engineer", "Senior Frontend Engineer",
         ["JavaScript", "TypeScript", "React", "CSS", "Git"],
         ["Next.js", "GraphQL", "Testing", "Node.js"], 4),
    ]

    for filename, role_key, title, req, nice, min_yrs in jobs:
        content = generate_job_description(0, role_key, title, req, nice, min_yrs)
        (JOBS_DIR / filename).write_text(content, encoding="utf-8")
    print(f"Generated 6 job descriptions in {JOBS_DIR}")


if __name__ == "__main__":
    main()
