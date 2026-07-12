"""
FastAPI endpoints for Job Application Helper backend.

Exposes a single POST /analyze endpoint that the React frontend calls.
Run with:
    uvicorn backend.api:app --reload --port 8000
"""

import io
import os
import tempfile
import time
import re
from typing import Optional

from docx import Document  # python-docx — used to read .docx resumes

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from pypdf import PdfReader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.trustedhost import TrustedHostMiddleware
import json
from langchain_openai import ChatOpenAI

load_dotenv()

from backend.agent import generate_resume_feedback_prompt, run_agent_analysis
from backend.job_processor import (
    calculate_skill_match_score,
    extract_education,
    extract_education_field,
    extract_job_seniority,
    extract_job_title_and_seniority,
    extract_qualifications,
    extract_skills,
    map_skills_to_source,
    match_education,
    match_seniority,
)
from backend.resume_processor import (
    extract_resume_education_degree,
    extract_resume_education_field,
    extract_resume_seniority,
)
from backend.rag_system import RAGSystem

app = FastAPI(title="Job Application Helper API")
DEBUG_PRIVACY_LOGS = os.getenv("DEBUG_PRIVACY_LOGS", "").lower() == "true"

# ── Rate limiting (slowapi) ──────────────────────────────────────────────────
# Per-IP limits prevent any single user from burning through the OpenAI budget.
# This is a soft protection

def get_real_ip(request: Request):
    return get_remote_address(request)

limiter = Limiter(key_func=get_real_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: always allow the React dev servers (Vite :5173, CRA :3000) for local
# development, plus any additional origins listed in FRONTEND_URL env var
# (comma-separated). In production, set FRONTEND_URL to your deployed
# frontend's URL, e.g. "https://jobmigo.vercel.app".
_dev_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]
_prod_origins = [
    o.strip() for o in os.getenv("FRONTEND_URL", "").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_dev_origins + _prod_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

FIELD_CATEGORIES = {
    "stem": {
        "computer science",
        "computer engineering",
        "software engineering",
        "data science",
        "information technology",
        "mathematics",
        "statistics",
        "physics",
        "chemistry",
        "biology",
        "engineering",
    },
    "humanities": {
        "english",
        "history",
        "philosophy",
        "literature",
        "linguistics",
        "classics",
    },
    "arts": {
        "art",
        "design",
        "fine arts",
        "graphic design",
        "music",
        "theater",
        "film",
    },
    "business": {
        "business",
        "finance",
        "accounting",
        "economics",
        "marketing",
        "management",
    },
    "quantitative": {
        "mathematics",
        "statistics",
        "economics",
        "physics",
    },
}


def _field_category(field: str) -> str | None:
    normalized = field.lower().strip()

    for category, values in FIELD_CATEGORIES.items():
        if category in normalized:
            return category

        if any(value in normalized for value in values):
            return category

    return None

def _check_field_match(job_fields, resume_fields):
    """Mirror of streamlit_app.check_field_match.

    Returns (match_found, matched_job_field, matched_resume_field).
    """
    if not job_fields:
        return True, None, None
    if not resume_fields:
        return False, None, None
    
    for job_field in job_fields:
        for resume_field in resume_fields:
            if _field_category(job_field) and _field_category(resume_field):
                if _field_category(job_field) == _field_category(resume_field):
                    return True, job_field, resume_field

    for job_field in job_fields:
        job_words = set(job_field.lower().split())
        for resume_field in resume_fields:
            resume_words = set(resume_field.lower().split())
            overlap = job_words & resume_words
            max_len = max(len(job_words), len(resume_words))
            if len(job_words) == 1 and len(resume_words) == 1:
                if job_words == resume_words:
                    return True, job_field, resume_field
            elif max_len and len(overlap) / max_len >= 0.5:
                return True, job_field, resume_field


    return False, None, None

RESUME_SUPPLEMENTAL_SIGNALS = [
    "projects",
    "certifications",
    "summary",
    "objective",
    "work experience",
    "professional experience",
    "technical skills",
    "github",
    "linkedin",
]

name_patterns = [
    r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",                           # John Smith
    r"\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b",                  # John R. Smith
    r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",             # John Robert Smith
    r"\b[A-Z]{2,}\s+[A-Z]{2,}\b",                             # HARSHIL PAHUJA
    r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b",
    r"\b[A-Z][a-z]+(?:'[A-Z][a-z]+)?\s+[A-Z][a-z]+(?:'[A-Z][a-z]+)?\b",
]

#Check whether the user actually uploaded a valid job description and not some slop
def validate_job_description_text(text: str) -> None:
    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Please paste a job description before analyzing.",
        )

    normalized = text.lower()

    has_required_signal = "required" in normalized or "requirements" in normalized or "minimum" in normalized or "must have" in normalized or "basic" in normalized
    has_preferred_signal = "preferred" in normalized or "nice to have" in normalized
    has_responsibilities_signal = "responsibilities" in normalized or "what you will do" in normalized
    has_summary_signal = "summary" in normalized or "overview" in normalized or "about the role" in normalized
    local_signals = {
        "required_or_minimum_qualifications": has_required_signal,
        "preferred_qualifications": has_preferred_signal,
        "responsibilities": has_responsibilities_signal,
        "summary_or_overview": has_summary_signal,
    }
    local_signal_count = sum(1 for found in local_signals.values() if found)

    try:
        validator = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            timeout=30,
            max_retries=1,
        )

        response = validator.invoke(
            f"""
You are validating pasted text for a job application analysis platform.

Determine whether the text is a job posting or job description with enough
information to compare against a resume.

Return ONLY valid JSON in this format:

{{
  "is_job_description": true,
  "confidence": 0.95,
  "document_type": "job_description",
  "primary_aspects_found": {{
    "role_summary": true,
    "responsibilities": true,
    "required_qualifications": true,
    "preferred_qualifications": false
  }},
  "reason": "Brief explanation"
}}

Primary aspects to look for:
- role summary, overview, title, or team context
- responsibilities, duties, or what the candidate will do
- required/basic/minimum qualifications
- preferred/nice-to-have qualifications
- skills, technologies, education, experience, or eligibility requirements

Examples of text that are NOT job postings include, but are not limited to, resume, cover letter,
article, assignment, random notes, or any other kind of text that is too vague to identify a role.

The local keyword signals found were:
{json.dumps(local_signals)}

Text:
{text[:6000]}
"""
        )

        if DEBUG_PRIVACY_LOGS:
            print("[JOB VALIDATION RAW RESPONSE]", repr(response.content))

        raw_content = (response.content or "").strip()
        raw_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.IGNORECASE)
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if json_match:
            raw_content = json_match.group()

        result = json.loads(raw_content)
        is_job_description = bool(result.get("is_job_description"))
        confidence = float(result.get("confidence", 0))
        aspects = result.get("primary_aspects_found", {})
        aspect_count = (
            sum(1 for found in aspects.values() if found)
            if isinstance(aspects, dict)
            else 0
        )

        if is_job_description and confidence >= 0.70 and (aspect_count >= 1 or local_signal_count >= 1):
            return

        # Let obvious job postings through when the model is overly strict,
        # but keep weak/random text blocked.
        if local_signal_count >= 2:
            print(
                "[JOB VALIDATION WARNING] "
                "LLM rejected job description, but local signals were strong enough to continue."
            )
            return

    except json.JSONDecodeError as e:
        print(
            "[JOB VALIDATION WARNING] "
            "Falling back to local job-description signals because the LLM returned invalid JSON: "
            f"{e}"
        )
        if local_signal_count >= 1:
            return

    except Exception as e:
        print(
            "[JOB VALIDATION WARNING] "
            "Falling back to local job-description signals because LLM validation failed: "
            f"{type(e).__name__}: {e}"
        )
        if local_signal_count >= 1:
            return

    raise HTTPException(
        status_code=400,
        detail="Pasted text does not look like a job description. Please upload a job description with sections for summary, required and/or preferred qualifications, and try again."
    )

# Check whether the user actually uploaded a valid resume and not some other document.
def validate_resume_text(text: str) -> None:
    normalized = text.lower()

    has_education = "education" in normalized
    has_skills = "skills" in normalized or "technical skills" in normalized
    has_experience_signal = any(
        signal in normalized
        for signal in [
            "experience",
            "work experience",
            "professional experience",
            "internships",
            "projects",
            "project experience",
        ]
    )

    if not (has_education and has_skills and has_experience_signal):
        raise HTTPException(
            status_code=400,
            detail="Uploaded document does not look like a resume. Please upload a resume with sections for work experience, education, and skills at the minimum, and try again."
        )

    supplemental_matches = sum(
        1 for signal in RESUME_SUPPLEMENTAL_SIGNALS if signal in normalized
    )

    if supplemental_matches < 1:
        raise HTTPException(
            status_code=400,
            detail="Uploaded document is not detailed enough to provide useful feedback. Consider adding more information about your projects, certifications, and extracurricular activities, and try again."
        )
    
    # Required header information
    header = re.sub(r"\s+", " ", text[:500]).strip()
    has_full_name = any(re.search(pattern, header) for pattern in name_patterns)
    has_email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", header) is not None
    has_phone = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", header) is not None

    if not has_full_name:
        raise HTTPException(
            status_code=400,
            detail="Could not detect a full name in the resume. Please make sure your full name is included and try again."
        )
    
    if not has_email and not has_phone:
        raise HTTPException(
            status_code=400,
            detail="Resume must include contact information."
        )

    # If deterministic local checks pass, the document is valid enough to
    # continue. The LLM is an extra guard against cover letters/other docs,
    # but model/API/JSON-format failures should not become user-facing 500s.
    try:

        validator = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )

        response = validator.invoke(
            f"""
You are validating uploaded documents for a resume analysis platform.

Determine whether the uploaded document is a resume.

Return ONLY valid JSON in the following format:

{{
    "is_resume": true,
    "document_type": "resume",
    "confidence": 0.95,
    "reason": "Brief explanation"
}}

Valid document types:
- resume
- cv

The document has already passed:
- Full Name Present: {has_full_name}
- Email Present: {has_email}
- Phone Present: {has_phone}

A cover letter may contain:
- experience
- education
- skills
- projects

but should NOT be classified as a resume.

A random document may contain:
- full name
- email
- phone number

but should NOT be classified as a resume.

Document:

{text[:5000]}
"""
        )

        if DEBUG_PRIVACY_LOGS:
            print("[VALIDATION RAW RESPONSE]", repr(response.content))

        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as e:
            print(
                "[VALIDATION WARNING] "
                "Skipping LLM resume validation because local checks passed but "
                f"the LLM returned invalid JSON: {e}"
            )
            return

        is_resume = result.get("is_resume", False)
        document_type = result.get("document_type", "other")
        confidence = float(result.get("confidence", 0))

        print(
            f"[VALIDATION] "
            f"type={document_type}, "
            f"confidence={confidence:.2f}, "
            f"is_resume={is_resume}"
        )

        if not is_resume or confidence < 0.80:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Uploaded document appears to be a "
                    f"{document_type}, not a resume."
                )
            )

    except HTTPException:
        raise

    except Exception as e:
        print(
            "[VALIDATION WARNING] "
            "Skipping LLM resume validation because local checks passed but "
            f"the LLM validation step failed: {type(e).__name__}: {e}"
        )
        return
    

def _extract_resume_text(upload: UploadFile) -> str:
    """Read resume bytes from the upload and return decoded text.

    Supports PDF (via pypdf) and Word documents (via doc or docx). Raises HTTPException on failure.
    """
    #Extra check to make sure nothing has consumed the file pointer before we read it. This can happen if the file is read in a previous step, e.g. for logging or validation.
    try:
        upload.file.seek(0)
    except Exception:
        pass

    if (len(raw := upload.file.read()) == 0):
        raise HTTPException(status_code=400, detail="Uploaded resume file is empty.")

    if (len(raw) > 5 * 1024 * 1024):
        raise HTTPException(
            status_code=400,
            detail="Uploaded resume file is too large. Please upload a file smaller than 5 MB.",
        )

    is_pdf = (
        (upload.content_type or "").lower() == "application/pdf"
        or (upload.filename or "").lower().endswith(".pdf")
    )

    is_word = (
        (upload.content_type or "").lower() in [
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ]
        or (upload.filename or "").lower().endswith((".doc", ".docx"))
    )

    if is_pdf:
        try:
            reader = PdfReader(io.BytesIO(raw))
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n".join(pages).strip()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read resume PDF: {exc}. Try re-saving the PDF and uploading again.",
            )
    elif is_word:
        try:
            doc = Document(io.BytesIO(raw))
            text = "\n".join([p.text for p in doc.paragraphs]).strip()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read resume Word document: {exc}. Try re-saving the document and uploading again.",
            )
    else:
        text = raw.decode("utf-8", errors="ignore").strip()
    
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file does not contain any readable text.",
        )
    if is_pdf and len(text.strip()) < 300:
        raise HTTPException(
            status_code=400,
            detail=(
                "The system could not read enough text from this PDF. It may be scanned, image-based, "
                "or exported in a format it cannot parse. Please either re-format it, re-export it as a text-based PDF, "
                "or upload a .docx file."
            ),
        )
    if is_pdf and len(text.strip()) > 5_500:
        raise HTTPException(
            status_code=400,
            detail=(
                "The system could not read this resume PDF because it is too long. Please re-save it as a .docx file "
                "or split it into smaller sections and try again."
            ),
        )
    if is_word and len(text.strip()) < 300:
        raise HTTPException(
            status_code=400,
            detail=(
                "The system could not read enough text from this Word document. It may be corrupted or in a format it cannot parse. "
                "Please re-save it as a .pdf file and try again."
            ),
        )
    if is_word and len(text.strip()) > 5_500:
        raise HTTPException(
            status_code=400,
            detail=(
                "The system could not read this Word document because it is too long. Please split it into smaller sections and try again."
            ),
        )
    validate_resume_text(text)
    return text


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
@limiter.limit("3/hour")  # Adjust as needed based on expected traffic and OpenAI budget
async def analyze(
    request: Request,
    job_description: str = Form(""),
    resume: Optional[UploadFile] = File(None),
):
    """Run the full resume/job analysis pipeline and return JSON.

    Both a `resume` and `job_description` must be provided.
    """
    resume_text = ""
    resume_error = None
    job_error = None

    try:
        resume_text = _extract_resume_text(resume)
    except HTTPException as exc:
        resume_error = exc.detail

    try:
        validate_job_description_text(job_description)
    except HTTPException as exc:
        job_error = exc.detail

    if resume_error and job_error:
        raise HTTPException(
            status_code=400,
            detail="Both your resume and job description are invalid. Please upload a valid resume and paste a valid job description."
        )

    if resume_error:
        raise HTTPException(status_code=400, detail=resume_error)

    if job_error:
        raise HTTPException(status_code=400, detail=job_error)

    MAX_JD_CHARS = 15_000
    if len(job_description) > MAX_JD_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Job description is too long ({len(job_description):,} characters). Please shorten it to under {MAX_JD_CHARS:,} characters.",
        )
    # ── Job-side extraction ────────────────────────────────────────────────
    validate_job_description_text(job_description)
    if job_description:
        #job_description_text = validate_job_description_text(job_description)
        job_required, job_preferred = extract_qualifications(job_description)
        job_skills = extract_skills(job_description, context="job_posting")
        job_required_education = extract_education(job_description)
        job_required_education_fields = extract_education_field(job_description)
        # extract_job_seniority is the hybrid YoE+title pipeline (more accurate
        # for edge cases like "AI Engineer I, 4+ years"). Only fall back to the
        # title-only LLM extractor if the hybrid returns nothing.
        job_seniority = (
            extract_job_seniority(job_description)
            or extract_job_title_and_seniority(job_description)[1]
        )
    else:
        job_required, job_preferred = [], []
        job_skills = []
        job_required_education = []
        job_required_education_fields = []
        job_seniority = None

    # ── Resume-side extraction ─────────────────────────────────────────────

    rag = None
    if resume_text:
        rag = RAGSystem(
            collection_name=f"api_resume_{int(time.time())}",
            embedding_backend=os.getenv("RAG_EMBEDDING_BACKEND", "auto"),
            persist_directory=None,
        )
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
                tmp.write(resume_text)
                tmp_path = tmp.name

            chunks = rag.load_and_process_document(tmp_path, chunk_size=500, overlap=50)
            rag.create_vectorstore(chunks)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if resume_text:
        resume_skills = extract_skills(resume_text, context="resume")
    else:
        resume_skills = []

    rag_instance = rag if (resume_text and rag and rag.vectorstore) else None
    resume_education = extract_resume_education_degree(rag_instance=rag_instance, resume_text=resume_text or None)
    resume_education_fields = extract_resume_education_field(rag_instance=rag_instance, resume_text=resume_text or None)
    resume_seniority = extract_resume_seniority(resume_text=resume_text) if resume_text else None

    # ── Skill matching (required + preferred merged into one) ──────────────
    req_bullets = job_required if isinstance(job_required, list) else []
    pref_bullets = job_preferred if isinstance(job_preferred, list) else []
    req_for_match = list(dict.fromkeys(job_skills + req_bullets + pref_bullets))
    skills_match = calculate_skill_match_score(req_for_match, [], resume_skills)
    unmatched_skills = [u["job_skill"] for u in skills_match["details"]["required"].get("unmatched", [])]

    # ── Education matching ─────────────────────────────────────────────────
    education_match = match_education(
        job_required_education,
        [],
        resume_education,
        resume_education_fields,
    )
    field_match_found, matched_job_field, matched_resume_field = _check_field_match(
        job_required_education_fields, resume_education_fields
    )

    # ── Seniority matching ─────────────────────────────────────────────────
    seniority_match = (
        match_seniority(job_seniority, resume_seniority)
        if (job_seniority or resume_seniority)
        else {}
    )

    # ── Agent feedback (only if a resume was uploaded) ─────────────────────
    skills_by_source = map_skills_to_source(resume_text, resume_skills) if resume_skills else {}
    degree_match = (
        education_match.get("required_degree_matched", False)
        if job_required_education
        else None
    )

    extraction_results = {
        "skills_match": skills_match,
        "skills_by_source": skills_by_source,
        "education_match": {
            "is_match": degree_match,
            "job_required_education": job_required_education,
            "required_degree_matched": degree_match,
            "required_degree_job": job_required_education,
            "required_degree_resume": resume_education,
            "education_field_job": job_required_education_fields,
            "resume_education_fields": resume_education_fields,
            "education_field_matched": (
                education_match.get("field_matched", False) or field_match_found
            ),
            "warning": education_match.get("warning", ""),
        },
        "seniority_match": seniority_match,
        "qualifications_job_required": list(dict.fromkeys(req_bullets + pref_bullets)),
        "qualifications_job_preferred": [],
    }

    feedback_markdown = ""
    if resume_text and rag and rag.vectorstore:
        job_title = job_description.split("\n")[0].strip() if job_description else "Position"
        feedback_prompt = generate_resume_feedback_prompt(
            job_title, job_description, extraction_results, resume_text=resume_text
        )
        try:
            feedback_markdown = run_agent_analysis(feedback_prompt, rag_instance=rag)
        except Exception as exc:
            feedback_markdown = f"_Could not generate AI feedback: {exc}_"

    # ── Response payload for the React frontend ────────────────────────────
    return {
        "skills": {
            "coverage": skills_match["required_score"],
            "matched": skills_match["required_matches"],
            "unmatched": unmatched_skills,
            "job_skills": req_for_match,
            "resume_skills": resume_skills,
        },
        "education": {
            "job_required_degrees": job_required_education,
            "resume_degrees": resume_education,
            "job_required_fields": job_required_education_fields,
            "resume_fields": resume_education_fields,
            "degree_matched": degree_match,
            "field_matched": field_match_found,
            "matched_job_field": matched_job_field,
            "matched_resume_field": matched_resume_field,
        },
        "seniority": {
            "job": job_seniority,
            "resume": resume_seniority,
            "is_match": seniority_match.get("is_match"),
            "is_overqualified": seniority_match.get("is_overqualified"),
            "is_underqualified": seniority_match.get("is_underqualified"),
            "warning": seniority_match.get("warning"),
            "recommendation": seniority_match.get("recommendation"),
        },
        "qualifications": list(dict.fromkeys(req_bullets + pref_bullets)),
        "feedback_markdown": feedback_markdown,
    }
