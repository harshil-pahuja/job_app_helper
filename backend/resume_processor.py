"""
Resume-specific extraction orchestration.

These functions use resume text and, when available, RAG-retrieved resume chunks,
then delegate general extraction primitives to job_processor.
"""
import json
import os
import re
from pathlib import Path
from langchain_openai import ChatOpenAI

from backend.job_processor import (
    detect_skill_aliases,
    extract_education,
    extract_education_field,
    normalize_job_posting_text,
)


DEBUG_PRIVACY_LOGS = os.getenv("DEBUG_PRIVACY_LOGS", "").lower() == "true"


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

    Layer 1 - RAG retrieves likely education/experience chunks
    Layer 2 - Graduation date: current students / recent grads (<=2 yrs) -> entry-level
    Layer 3 - LLM reads only job-header lines (title + company + date; no bullet points)
    Fallback - Graduation-year math for older grads whose title lines cannot be parsed

    Args:
        rag_instance: RAGSystem instance with loaded resume chunks (optional)
        resume_text: Full resume text (required)
        graduation_date: Optional graduation year (int) to override auto-detection

    Returns:
        str: One of 'entry-level', 'mid-level', 'senior', 'lead/principal', or None
    """
    CURRENT_YEAR = 2026
    candidate_text = ""

    # -- Layer 1: RAG retrieval -------------------------------------------------
    if rag_instance and rag_instance.vectorstore:
        try:
            seniority_chunks = rag_instance.retrieve_relevant_chunks(
                "work experience job titles internships employment dates education graduation",
                top_k=4,
            )
            candidate_text = "\n".join(c.page_content for c in seniority_chunks)
        except Exception:
            pass

    if not candidate_text and resume_text:
        candidate_text = resume_text

    # -- Auto-detect graduation year -------------------------------------------
    if graduation_date is None and candidate_text:
        grad_patterns = [
            r'(?:expected|anticipated|graduating)\s*[:\-]?\s*(?:May|June|August|December|Spring|Fall|Winter)?\s*(20\d{2})',
            r'(?:B\.?S\.?|B\.?A\.?|Bachelor|Master|M\.?S\.?)\b.*?(20\d{2})',
            r'(?:May|June|August|December)\s+(20\d{2})',
        ]
        for pattern in grad_patterns:
            hits = re.findall(pattern, candidate_text, re.IGNORECASE)
            if hits:
                valid = [int(y) for y in hits if 2000 <= int(y) <= 2035]
                if valid:
                    graduation_date = max(valid)
                    break

    # -- Layer 1: Graduation date ----------------------------------------------
    # Current students (future grad) and recent graduates (<=2 yrs) -> entry-level
    if graduation_date and graduation_date >= CURRENT_YEAR - 2:
        print(f"[DEBUG] Graduation year {graduation_date} -> entry-level (<=2 yrs since grad)")
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
    job_header_lines = date_line_pat.findall(candidate_text or "")

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
                    print(f"[DEBUG] LLM seniority extraction -> {level}")
                    return level
        except Exception:
            pass

    # -- Fallback: graduation math for older grads without parseable title lines -
    if graduation_date:
        years_since_grad = CURRENT_YEAR - graduation_date
        print(f"[DEBUG] Graduation year {graduation_date} -> {years_since_grad} years since grad")
        if years_since_grad > 10:   return 'lead/principal'
        if years_since_grad > 5:    return 'senior'
        if years_since_grad > 2:    return 'mid-level'
        return 'entry-level'

    return None

def extract_skills_from_resume(rag_instance=None, resume_text=None):
    """Extract skills from a resume using RAG + LLM plus regex supplement.

    Args:
        rag_instance: RAGSystem instance with loaded resume chunks (optional)
        resume_text:  Full resume text (fallback if RAG unavailable)

    Returns:
        list: Skills found, e.g. ['Python', 'Project Management']
    """
    candidate_text = ""

    # Layer 1 - RAG retrieval
    if rag_instance and rag_instance.vectorstore:
        try:
            skill_chunks = rag_instance.retrieve_relevant_chunks(
                "skills technologies tools programming languages",
                top_k=3,
            )
            candidate_text = "\n".join(c.page_content for c in skill_chunks)
        except Exception:
            pass

    # Fall back to full resume text when RAG is unavailable or returned nothing
    if not candidate_text and resume_text:
        candidate_text = resume_text

    if not candidate_text:
        return []

    llm_skills = []

    # Layer 2 - LLM extraction
    try:
        model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = """You are a skills extraction expert reading a resume.
Extract ONLY the skills, technologies, and tools this person has experience with.
Technical skills, programming languages, software tools, and soft skills are all valid.
Return a JSON object: {"skills": ["Python", "Project Management"]}
Rules:
- Return only the skill names, no descriptions or context.
- If no skills are mentioned, return {"skills": []}."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract skills from this resume text:\n\n{candidate_text[:3000]}"),
        ]
        response = model.invoke(messages)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            llm_skills = [s.strip() for s in parsed.get("skills", []) if s.strip()]
    except Exception:
        pass

    # Layer 3 - targeted regex supplement for explicit resume skill/list sections.
    regex_skills = []
    skill_section_pattern = re.compile(
        r'^\s*(?:technical\s+skills|skills|programming\s+languages?|languages?|frameworks?|tools(?:\s+and\s+technologies)?|technologies|databases?|cloud|devops|software|libraries)\s*:\s*(.+)$',
        re.IGNORECASE | re.MULTILINE,
    )
    for match in skill_section_pattern.finditer(candidate_text):
        value = match.group(1)
        for item in re.split(r',|;|\||/|\u2022|\bor\b', value):
            skill = item.strip().strip("()[]{}").strip()
            if 2 <= len(skill) <= 100 and not skill.isdigit():
                regex_skills.append(skill)

    combined = set()
    for skill in llm_skills + regex_skills:
        normalized = normalize_job_posting_text(skill)
        if 2 <= len(normalized) <= 100:
            combined.add(normalized)

    return detect_skill_aliases(combined)


def extract_work_experience_companies(resume_text: str, rag_instance=None) -> list[str]:
    """
    Extract employer/company names from the resume work experience section.

    Uses RAG-retrieved work experience chunks when available, then falls back
    to section parsing and collapsed-text recovery. Makes one LLM extraction
    call, with local regex as a final fallback.
    """
    if not resume_text:
        return []

    lines = resume_text.split('\n')
    companies = []
    seen = set()
    in_work_section = False
    experience_content = []

    for line in lines:
        line_stripped = line.strip()

        if re.search(r'^\s*(Work\s+Experience(s)?|Experience(s)?|Internship(s)?|Employment|Career)\s*(?:\n|:|$)', line, re.IGNORECASE):
            in_work_section = True
            continue
        if re.search(r'^\s*(Projects|Education|Leadership|Awards|Skills|Certifications?|References)\s*(?:\n|:|$)', line, re.IGNORECASE):
            in_work_section = False
            continue

        if in_work_section:
            experience_content.append(line_stripped)

    candidate_experience_text = ""

    if rag_instance and rag_instance.vectorstore:
        try:
            experience_chunks = rag_instance.retrieve_relevant_chunks(
                "work experience employment internships employers companies roles",
                top_k=4,
            )
            candidate_experience_text = "\n".join(c.page_content for c in experience_chunks)
        except Exception:
            pass

    work_exp_text = '\n'.join(experience_content)
    if not candidate_experience_text.strip():
        candidate_experience_text = work_exp_text

    if not candidate_experience_text.strip():
        normalized_text = re.sub(r"\s+", " ", resume_text).strip()
        experience_match = re.search(
            r"\b(?:Work\s+Experience(s)?|Experience(s)?|Internship(s)?|Employment|Career)\b",
            normalized_text,
            re.IGNORECASE,
        )

        if experience_match:
            after_experience = normalized_text[experience_match.end():]
            next_section_match = re.search(
                r"\b(?:Projects|Education|Leadership|Awards|Skills|Certifications?|References)\b",
                after_experience,
                re.IGNORECASE,
            )
            chunk_end = (
                experience_match.end() + next_section_match.start()
                if next_section_match
                else min(len(normalized_text), experience_match.start() + 3500)
            )
            candidate_experience_text = normalized_text[experience_match.start():chunk_end][:3500]

    if candidate_experience_text.strip():
        try:
            validator = ChatOpenAI(model="gpt-4o", timeout=30, max_retries=1)
            response = validator.invoke(
                f"""
You are a resume parsing assistant.
Extract employer/company names from this work experience text.

Return ONLY valid JSON:
{{"companies": ["Company A", "Company B"]}}

If no company names can be confidently extracted:
{{"companies": []}}

Rules:
- Include only actual employers, labs, universities, startups, or organizations where the candidate worked.
- Do not include job titles, dates, locations, skills, bullets, projects, or section headers.

Work experience text:
"{candidate_experience_text}"
                """
            )
            if DEBUG_PRIVACY_LOGS:
                print("[COMPANY EXTRACTION RAW RESPONSE]", repr(response.content))

            result = {"companies": []}
            raw_content = (response.content or "").strip()
            raw_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.IGNORECASE)

            try:
                result = json.loads(raw_content)
            except json.JSONDecodeError as e:
                print(
                    "[VALIDATION WARNING] "
                    "Skipping LLM company extraction because "
                    f"the LLM returned invalid JSON: {e}"
                )

            companies_extracted = result.get("companies", [])
            if isinstance(companies_extracted, str):
                companies_extracted = [companies_extracted]
            if not companies_extracted and result.get("company_name"):
                companies_extracted = [result["company_name"]]

            print(f"[COMPANY EXTRACTION] count={len(companies_extracted)}")

            for company in companies_extracted:
                company = company.strip()
                if company and company.lower() != "none" and company not in seen:
                    companies.append(company)
                    seen.add(company)

        except Exception as e:
            print(
                "[VALIDATION WARNING] "
                "Skipping LLM company extraction because "
                f"the LLM validation step failed: {type(e).__name__}: {e}"
            )

    if not companies:
        regex_lines = experience_content or candidate_experience_text.splitlines()
        for line in regex_lines:
            work_exp_regex = re.search(r'([A-Z][^|]*?)\s*\|\s*([A-Za-z0-9][A-Za-z0-9\s&\-\.]+?)(?:\s{2,}|$)', line)

            if work_exp_regex and not any(keyword in line.lower() for keyword in ['required', 'preferred', 'qualifications', 'skills:']):
                left_side = work_exp_regex.group(1).strip()
                right_side = work_exp_regex.group(2).strip()
                job_title_keywords = [
                    'engineer', 'developer', 'manager', 'analyst', 'specialist',
                    'architect', 'lead', 'senior', 'junior', 'associate',
                    'director', 'executive', 'officer', 'coordinator',
                    'consultant', 'intern', 'assistant',
                ]

                left_has_title_keywords = any(keyword in left_side.lower() for keyword in job_title_keywords)
                right_has_title_keywords = any(keyword in right_side.lower() for keyword in job_title_keywords)

                if left_has_title_keywords and not right_has_title_keywords:
                    candidate = right_side
                elif right_has_title_keywords and not left_has_title_keywords:
                    candidate = left_side
                else:
                    candidate = left_side if len(left_side) < len(right_side) else right_side

                candidate_lower = candidate.lower()
                leadership_terms = [
                    'club', 'organization', 'society', 'association', 'board',
                    'committee', 'president', 'founder', 'group',
                ]

                if (
                    len(candidate) > 5
                    and candidate not in seen
                    and not any(fw in candidate_lower for fw in ['react', 'node', 'express', 'firebase', 'data science'])
                    and not any(term in candidate_lower for term in leadership_terms)
                ):
                    companies.append(candidate)
                    seen.add(candidate)

    return companies


def map_skills_to_source(resume_text, resume_skills):
    """
    Map extracted skills to their source across the entire resume.

    Args:
        resume_text: Full resume text
        resume_skills: List of extracted skills

    Returns:
        dict mapping resume sections/sources to skills found in that section
    """
    skills_by_source = {}

    lines = resume_text.split('\n')
    sections = []
    current_section_start = 0
    current_section_name = None

    i = 0
    while i < len(lines):
        major_section_match = re.search(
            r'^\s*(Work\s+Experience(s)?|Internship(s)?|Employment|Career|Projects|Education|Leadership|Awards|Skills|Certifications?|References)\s*(?:\n|:|$)',
            lines[i],
            re.IGNORECASE,
        )

        if major_section_match:
            if current_section_name:
                sections.append((current_section_start, i, current_section_name, 'section'))

            current_section_name = major_section_match.group(1).lower()
            current_section_start = i + 1
            i += 1

            if current_section_name in ['work experience', 'work experiences', 'internship', 'internships', 'employment', 'career']:
                j = i
                while j < len(lines):
                    if re.search(r'^\s*(Projects|Education|Leadership|Awards|Skills|Certifications?|References)\s*(?:\n|:|$)', lines[j], re.IGNORECASE):
                        break

                    work_exp_match = re.search(r'([A-Z][^|]*?)\s*\|\s*([A-Za-z0-9][A-Za-z0-9\s&\-\.]+?)(?:\s{2,}|$)', lines[j])

                    if work_exp_match and not any(kw in lines[j].lower() for kw in ['required', 'preferred', 'qualifications']):
                        company_or_role = work_exp_match.group(1).strip()
                        role_or_company = work_exp_match.group(2).strip()
                        company_name = company_or_role if len(company_or_role) > len(role_or_company) else role_or_company

                        if j > current_section_start:
                            sections.append((current_section_start, j, current_section_name, 'section'))

                        k = j + 1
                        while k < len(lines):
                            if re.search(r'^\s*(Projects|Education|Leadership|Awards|Skills|Certifications?|References)\s*(?:\n|:|$)', lines[k], re.IGNORECASE):
                                break

                            next_work_match = re.search(r'([A-Z][^|]*?)\s*\|\s*([A-Za-z0-9][A-Za-z0-9\s&\-\.]+?)(?:\s{2,}|$)', lines[k])
                            if next_work_match and not any(kw in lines[k].lower() for kw in ['required', 'preferred']):
                                break

                            k += 1

                        sections.append((j, k, company_name, 'work_exp'))
                        j = k
                    else:
                        j += 1

                current_section_start = j
                i = j
                current_section_name = None
            else:
                i += 1
        else:
            i += 1

    if current_section_name:
        sections.append((current_section_start, len(lines), current_section_name, 'section'))

    for start, end, section_name, section_type in sections:
        section_lines = lines[start:end]
        section_text = '\n'.join(section_lines)
        section_text_normalized = normalize_job_posting_text(section_text)

        if not section_text.strip():
            skills_by_source[section_name] = []
            continue

        matched_skills = []
        for resume_skill in resume_skills:
            normalized_skill = normalize_job_posting_text(resume_skill)
            skill_pattern = r'(?<![\w+#.])' + re.escape(normalized_skill) + r'(?![\w+#.])'
            if re.search(skill_pattern, section_text_normalized):
                matched_skills.append(resume_skill)

        skills_by_source[section_name] = matched_skills

    generic_sections = {'skills', 'experience', 'work experience', 'internships', 'career', 'employment'}
    specific_sections = {
        k for k in skills_by_source.keys()
        if k.lower() not in generic_sections and k.lower() not in {'education', 'leadership', 'awards', 'projects'}
    }

    if 'skills' in skills_by_source and specific_sections:
        specific_skills = set()
        for section in specific_sections:
            for skill in skills_by_source[section]:
                specific_skills.add(skill.lower())

        skills_by_source['skills'] = [
            skill for skill in skills_by_source['skills']
            if skill.lower() not in specific_skills
        ]

    return skills_by_source

def main():
    """Debug resume-side extraction across every resume in tests/resumes."""
    from backend.rag_system import RAGSystem

    resume_dir = Path("tests/resumes")
    if not resume_dir.exists():
        print(f"Resume folder not found: {resume_dir}")
        return

    resume_paths = sorted(
        path for path in resume_dir.iterdir()
        if path.suffix.lower() in {".pdf", ".docx", ".txt"}
    )
    if not resume_paths:
        print(f"No supported resumes found in {resume_dir}")
        return

    print(f"Testing resume extraction for {len(resume_paths)} resume(s)")
    print("=" * 80)

    for index, resume_path in enumerate(resume_paths, start=1):
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", resume_path.stem).strip("._-")
        safe_name = (safe_name[:80] or f"resume_{index}").strip("._-")

        print(f"\nResume {index}/{len(resume_paths)}: {resume_path.name}")
        try:
            rag = RAGSystem(
                collection_name=f"debug_resume_{index}_{safe_name}",
                persist_directory=None,
            )

            resume_text = rag.extract_and_read_resume_text(str(resume_path))
            chunks = rag.load_and_process_document(str(resume_path))
            rag.create_vectorstore(chunks)
            rag_instance = rag if rag.vectorstore else None

            skills = extract_skills_from_resume(rag_instance=rag_instance, resume_text=resume_text)
            skills_by_source = map_skills_to_source(resume_text, skills) if skills else {}
            education_degrees = extract_resume_education_degree(rag_instance=rag_instance, resume_text=resume_text)
            education_fields = extract_resume_education_field(rag_instance=rag_instance, resume_text=resume_text)
            seniority = extract_resume_seniority(rag_instance=rag_instance, resume_text=resume_text)
            companies = extract_work_experience_companies(resume_text, rag_instance=rag_instance)

            print(f"- extracted_text_chars: {len(resume_text)}")
            print(f"- chunks: {len(chunks)}")
            print(f"- chunk_sections: {[section for _, section in chunks]}")
            print(f"- skills_count: {len(skills)}")
            print(f"- skills: {skills}")
            print(f"- skills_by_source_counts: { {source: len(source_skills) for source, source_skills in skills_by_source.items()} }")
            print(f"- education_degrees: {education_degrees}")
            print(f"- education_fields: {education_fields}")
            print(f"- seniority: {seniority}")
            print(f"- work_experience_companies: {companies}")
        except Exception as exc:
            print(f"- ERROR: {type(exc).__name__}: {exc}")

        print("=" * 80)


if __name__ == "__main__":
    main()
