"""
Resume-specific extraction orchestration.

These functions use resume text and, when available, RAG-retrieved resume chunks,
then delegate general extraction primitives to job_processor.
"""
import json
import re
from langchain_openai import ChatOpenAI

from backend.job_processor import extract_education, extract_education_field


def extract_resume_education_degree(rag_instance=None, resume_text=None):
    """Extract degree level(s) from a resume using RAG + LLM fallback.

    Mirrors the same 3-layer pattern as the other resume-side extractors:
      1. RAG retrieves the education section with a targeted query
      2. Regex (extract_education) runs on the retrieved chunk text
      3. LLM fills in when regex comes back empty

    Args:
        rag_instance: RAGSystem instance with loaded resume chunks (optional)
        resume_text:  Full resume text (fallback if RAG unavailable)

    Returns:
        list: Degree types found, e.g. ['Bachelor\'s', 'Master\'s']
    """
    candidate_text = ""

    # Layer 1 - RAG retrieval
    if rag_instance and rag_instance.vectorstore:
        try:
            edu_chunks = rag_instance.retrieve_relevant_chunks(
                "education degree bachelor master PhD university GPA graduation",
                top_k=3,
            )
            candidate_text = "\n".join(c.page_content for c in edu_chunks)
        except Exception:
            pass

    # Fall back to full resume text when RAG is unavailable or returned nothing
    if not candidate_text and resume_text:
        candidate_text = resume_text

    if not candidate_text:
        return []

    # Layer 2 - unified degree extractor
    extracted_result = extract_education(candidate_text)
    if extracted_result:
        return extracted_result

    # Also try on full text (catches sections not in top chunks)
    if resume_text and candidate_text != resume_text:
        extracted_full = extract_education(resume_text)
        if extracted_full:
            return extracted_full

    # Layer 3 - LLM fallback (only reached when regex found nothing)
    try:
        model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = """You are an education extraction expert reading a resume.
Extract ONLY the degree TYPE(s) this person has earned or is currently pursuing.

Return a JSON object: {"degrees": ["Bachelor's"]}

Valid degree types: Bachelor's, Master's, PhD, Associate's, Diploma, High School, GED.
Do NOT include the field/major (e.g., Computer Science) - only the degree level.
If no degree is mentioned, return {"degrees": []}."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract degree level(s) from this resume text:\n\n{candidate_text[:3000]}"),
        ]
        response = model.invoke(messages)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            degrees = [d.strip() for d in parsed.get("degrees", []) if d.strip()]
            if degrees:
                return degrees
    except Exception:
        pass

    return []


def extract_resume_education_field(rag_instance=None, resume_text=None):
    """Extract education field/major(s) from a resume using RAG + LLM + regex fallback.

    Layer order:
      1. RAG retrieves the education section with a targeted query
      2. LLM (GPT-4o) extracts the field from the retrieved chunk text - preferred
         because it handles formats like "Bachelor of Arts in Psychology" correctly
         where regex would capture "Arts" instead of "Psychology"
      3. Regex fallback when LLM is unavailable (no API key) or returns empty

    Args:
        rag_instance: RAGSystem instance with loaded resume chunks (optional)
        resume_text:  Full resume text (fallback if RAG unavailable)

    Returns:
        list: Field/major names found, e.g. ['Computer Science']
    """
    candidate_text = ""

    # Layer 1 - RAG retrieval
    if rag_instance and rag_instance.vectorstore:
        try:
            edu_chunks = rag_instance.retrieve_relevant_chunks(
                "education degree major field of study bachelor master university GPA graduation",
                top_k=3,
            )
            candidate_text = "\n".join(c.page_content for c in edu_chunks)
        except Exception:
            pass

    # Fall back to full resume text when RAG is unavailable or returned nothing
    if not candidate_text and resume_text:
        candidate_text = resume_text

    if not candidate_text:
        return []

    # Layer 2 - LLM (preferred when API is available; more accurate than regex for
    # formats like "Bachelor of Arts in Psychology" where regex captures "Arts")
    try:
        model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = """You are an education extraction expert reading a resume.
Extract ONLY the academic field(s) / major(s) this person studied.

Return a JSON object: {"fields": ["Computer Science"]}

Rules:
- Extract the field/major name only (e.g., "Computer Science", "Mechanical Engineering", "Business Administration")
- Do NOT include the degree level (Bachelor's, Master's, etc.)
- Do NOT include the university name
- Do NOT include certifications, courses, or minors (unless only a minor is present)
- If no field/major is mentioned, return {"fields": []}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract the field/major from this resume text:\n\n{candidate_text[:3000]}"),
        ]
        response = model.invoke(messages)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            fields = [f.strip() for f in parsed.get("fields", []) if f.strip() and len(f.strip()) >= 4]
            if fields:
                return fields
    except Exception:
        pass

    # Layer 3 - unified field extractor fallback (no API key or LLM returned empty)
    extracted_result = extract_education_field(candidate_text)
    if extracted_result:
        return extracted_result

    # Also try on full text (catches sections not in top RAG chunks)
    if resume_text and candidate_text != resume_text:
        extracted_full = extract_education_field(resume_text)
        if extracted_full:
            return extracted_full

    return []




def extract_resume_seniority(rag_instance=None, resume_text=None, graduation_date=None):
    """Extract seniority level from resume.

    Layer 1 - Graduation date: current students / recent grads (<=2 yrs) -> entry-level
    Layer 2 - LLM reads only job-header lines (title + company + date; no bullet points)
    Fallback - Graduation-year math for older grads whose title lines cannot be parsed

    Args:
        rag_instance: RAGSystem instance (unused, kept for API compatibility)
        resume_text: Full resume text (required)
        graduation_date: Optional graduation year (int) to override auto-detection

    Returns:
        str: One of 'entry-level', 'mid-level', 'senior', 'lead/principal', or None
    """
    CURRENT_YEAR = 2026

    # -- Auto-detect graduation year -------------------------------------------
    if graduation_date is None and resume_text:
        grad_patterns = [
            r'(?:expected|anticipated|graduating)\s*[:\-]?\s*(?:May|June|August|December|Spring|Fall|Winter)?\s*(20\d{2})',
            r'(?:B\.?S\.?|B\.?A\.?|Bachelor|Master|M\.?S\.?)\b.*?(20\d{2})',
            r'(?:May|June|August|December)\s+(20\d{2})',
        ]
        for pattern in grad_patterns:
            hits = re.findall(pattern, resume_text, re.IGNORECASE)
            if hits:
                valid = [int(y) for y in hits if 2000 <= int(y) <= 2035]
                if valid:
                    graduation_date = max(valid)
                    break

    # -- Layer 1: Graduation date ----------------------------------------------
    # Current students (future grad) and recent graduates (<=2 yrs) -> entry-level
    if graduation_date and graduation_date >= CURRENT_YEAR - 2:
        return 'entry-level'

    # -- Extract job-header lines (title + company + date; no bullet points) ---
    # Non-bullet lines that contain a month-year date or "Present"
    _MONTHS = (r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
               r'January|February|March|April|June|July|August|'
               r'September|October|November|December)')
    date_line_pat = re.compile(
        rf'^(?![\s\u2022\-\*])(.+(?:{_MONTHS}[^\n]*\d{{4}}|Present)[^\n]*)',
        re.MULTILINE | re.IGNORECASE,
    )
    job_header_lines = date_line_pat.findall(resume_text or "")

    # -- Layer 2: LLM on job-header lines -------------------------------------
    if job_header_lines:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
            lines_text = "\n".join(job_header_lines[:20])

            system_prompt = """You are a career-level expert.

Given ONLY the date-bearing lines from a resume below (each shows a role, company, or time period with a date - no bullet-point descriptions), classify the person's seniority as exactly one of:
- "entry-level"  : intern, apprentice, research assistant, teaching assistant, or 0-2 yrs total work experience
- "mid-level"    : 2-5 yrs, titles like Engineer / Developer / Analyst without a senior/lead prefix
- "senior"       : 5-10 yrs, or a title explicitly marked Senior or Sr
- "lead/principal": 10+ yrs, or Lead, Principal, Staff, Architect, Manager, Director, VP

Focus on work-role lines only; ignore education enrollment lines and project names.
Return ONLY JSON: {"seniority": "entry-level"}
No explanation."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Date-bearing lines from resume:\n\n{lines_text}"),
            ]
            response = model.invoke(messages)
            json_match = re.search(r'\{.*?\}', response.content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                level = parsed.get("seniority", "").strip().lower()
                if level in ('entry-level', 'mid-level', 'senior', 'lead/principal'):
                    return level
        except Exception:
            pass

    # -- Fallback: graduation math for older grads without parseable title lines -
    if graduation_date:
        years_since_grad = CURRENT_YEAR - graduation_date
        if years_since_grad > 10:   return 'lead/principal'
        if years_since_grad > 5:    return 'senior'
        if years_since_grad > 2:    return 'mid-level'
        return 'entry-level'

    return None


