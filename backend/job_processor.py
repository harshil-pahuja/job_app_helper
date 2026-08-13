"""
Job-processing utilities for job description analysis, extraction, and matching.
"""
import re
import json
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from difflib import SequenceMatcher
from dotenv import load_dotenv  # type: ignore

load_dotenv()

generic_words = {
    "apply", "save", "submit", "click", "here", "link", "website", "company", 
    "benefits", "culture", "mission", "values", "team", "role", "position", "salary"
}

# Examples used to teach the LLM how skill aliases should be normalized.
skill_alias_examples = {
    'github': 'git',
    'github actions': 'git',
    'google cloud platform': 'gcp',
    'google cloud': 'gcp',
    'postgres': 'postgresql',
    'js': 'javascript'
}

def normalize_job_posting_text(text):
    """Normalize job posting text or extracted snippets for matching."""
    normalized = str(text).strip().lower()
    normalized = normalized.replace('_', ' ')
    normalized = re.sub(r'\s+\d+(\.\d+)*', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    normalized = re.sub(r'\s*\(.*\)$', '', normalized).strip()

    if len(normalized) > 100 or len(normalized) < 2:
        return str(text).strip().lower()
    return normalized


def detect_skill_aliases(skills):
    """Use an LLM to normalize extracted technical skill aliases."""
    cleaned_skills = sorted({
        normalize_job_posting_text(skill)
        for skill in skills
        if skill and str(skill).strip()
    })
    if not cleaned_skills:
        return []

    alias_examples = "\n".join(
        f"- {alias} -> {normalized}"
        for alias, normalized in skill_alias_examples.items()
    )

    try:
        model = ChatOpenAI(model='gpt-4o', temperature=0, timeout=60, max_retries=2)
        system_prompt = f"""You normalize extracted technical skills into normalized names.

Return ONLY valid JSON in this exact format:
{{"skills": ["normalized skill"]}}

Rules:
- Preserve every real skill from the input.
- Merge aliases and spelling variants into one normalized skill name.
- Do not invent skills that are not present.
- Do not remove a skill just because it is broad if it is explicitly present.
- Use lowercase normalized names.

Examples:
{alias_examples}"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Normalize these extracted skills:\n\n{json.dumps(cleaned_skills)}")
        ]
        response = model.invoke(messages)
        json_match = re.search(r"\{.*?\}", response.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            normalized_skills = parsed.get("skills", [])
            if isinstance(normalized_skills, list):
                return sorted({
                    normalize_job_posting_text(skill)
                    for skill in normalized_skills
                    if skill and 1 <= len(str(skill).strip()) <= 100
                })
    except Exception:
        pass

    return cleaned_skills

# Check if generic or vague skill candidates (e.g., "apply", "save", etc.) should be filtered out.
def is_generic(skill):
    skill_lower = skill.strip().lower()
    if not skill_lower:
        return True

    # Exact match (entire skill is a generic word)
    if skill_lower in generic_words:
        return True
    
    # Check if skill CONTAINS any individual generic word (word boundary match)
    for word in generic_words:
        # For multi-word generic terms, do substring match
        if ' ' in word:
            if word in skill_lower:
                return True
        else:
            # Use word boundaries to match whole words only
            if re.search(r'\b' + re.escape(word) + r'\b', skill_lower):
                return True
    
    # Compact deterministic meta-language checks. Keep these shape-based so this
    # does not become another large hardcoded vocabulary list.
    meta_patterns = [
        r'^at least\b',
        r'\bone or more\b',
        r'\b\d+\s*or\s*more\b',
        r'^prefer(?:red)?\b',
        r'\bequivalent\b',
        r'^no\s+',
        r'^\d+',
        r'\d+[a-z]*\+?',
        r'\w+-(?:scale|grade|level)\b',
        r'\b(?:inc|llc|ltd|corp|co\.?|gmbh)\b',
    ]
    if any(re.search(pattern, skill_lower) for pattern in meta_patterns):
        return True

    if len(skill.split()) > 5:
        return True

    if len(skill) <= 2 and not skill.replace('+', '').replace('#', '').isalpha():
        return True

    try:
        model = ChatOpenAI(model='gpt-4o', temperature=0, timeout=60, max_retries=2)
        system_prompt = """You are filtering extracted skill candidates from job postings and resumes.

Return ONLY valid JSON in this exact format:
{"is_generic": true}

A candidate is generic if it is vague, non-specific, UI text, company boilerplate,
or not a concrete skill, tool, technology, qualification, method, domain, or competency.

Classify these as generic unless they are part of a specific technical/domain skill:
- locations, workplace types, or geographic regions
- industries, business sectors, departments, or company categories
- dates, months, years, durations, schedules, or time references
- benefits, compensation, hiring-process, or application instructions
- company culture, mission, values, boilerplate, equal-opportunity, or about-the-company language
- aspirational statements about what the company does, believes, builds, transforms, protects, or empowers
- meta-requirements, metrics, vague descriptors, company names, or long descriptive phrases"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Determine whether this extracted skill candidate is generic:\n\n{skill_lower}")
        ]
        response = model.invoke(messages)
        json_match = re.search(r"\{.*?\}", response.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed.get("is_generic") is True
    except Exception:
        pass

    return False


def tokenize_phrase(phrase):
    """Return a set of normalized tokens for a phrase (alphanumeric tokens).

    Uses simple splitting and filtering to avoid heavy dependencies.
    """
    if not phrase:
        return set()
    # remove punctuation except + and # (for C++/C#)
    cleaned = re.sub(r"[^\w\s\+#]", " ", phrase.lower())
    tokens = {t for t in cleaned.split() if len(t) > 1}
    return tokens


def token_overlap_score(a, b):
    """Compute token overlap score between two phrases (0..1).

    Score = intersection / max(len(tokens_a), len(tokens_b)).
    """
    #Useful for catching matches like "python programming" vs "programming in python" where token sets are the same but word order differs.
    ta = tokenize_phrase(a)
    tb = tokenize_phrase(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    denom = max(len(ta), len(tb))
    return len(inter) / denom
# token overlap scoring examples retained as reference:
# token_overlap_score("python programming", "programming in python") -> 2/3
# token_overlap_score("google cloud", "google cloud platform") -> 2/3
# token_overlap_score("java", "javascript") -> 0/2

def fuzzy_ratio(a, b):
    """Return a fuzzy similarity ratio between two strings using SequenceMatcher."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _select_posting_sections(text, expected_section_keywords, include_preamble=False):
    """
    Return text from job-posting sections whose headings match expected keywords.

    If no matching section is found, return the original text so extraction still
    works for postings without clean headings.
    """
    sections = split_posting_into_sections(text, include_preamble=include_preamble)
    if not sections:
        return text

    expected = tuple(keyword.lower() for keyword in expected_section_keywords)
    selected = []
    for section_name, section_text in sections.items():
        normalized_name = section_name.lower()
        if section_name == "__preamble__" and not include_preamble:
            continue
        if any(keyword in normalized_name for keyword in expected):
            selected.append(f"{section_name}\n{section_text}".strip())

    return "\n\n".join(selected) if selected else text

def extract_skills(text):
    """
    Extract required and preferred skills from a job posting using LLM semantic
    extraction plus regex/spaCy fallback.
    Includes technical skills and soft skills so job postings with interpersonal
    requirements still get represented.
    """
    if not text or not text.strip():
        return []

    candidate_text = _select_posting_sections(
        text,
        (
            "skills",
            "qualifications",
            "requirements",
            "what we need to see",
            "ways to stand out",
            "what you will bring",
        ),
    )

    llm_skills = set()
    try:
        model = ChatOpenAI(model='gpt-4o', timeout=60, max_retries=2)

        system_prompt = """You are a skills extraction expert. Extract required and preferred skills from a job posting.

Include:
- Technical skills such as programming languages, frameworks, databases, cloud, DevOps, ML, tools, and platforms
- Soft/professional skills when the job posting explicitly requires them, such as communication, collaboration, leadership, problem-solving, adaptability, customer service, and attention to detail
- For database technologies, extract the specific database, engine, storage system, or query language required by the posting. Do not collapse specific tools into a broader generic term when the text names the specific tool.

Return a JSON object with fields 'required_skills' and 'preferred_skills' (both lists).
Example: {"required_skills": ["Python", "Communication"], "preferred_skills": ["Docker"]}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract skills from this text:\n\n{candidate_text[:4000]}")
        ]

        response = model.invoke(messages)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            extracted = parsed.get('required_skills', []) + parsed.get('preferred_skills', [])

            for skill in extracted:
                skill = str(skill).strip().lower()
                if 2 <= len(skill) <= 100 and not is_generic(skill):
                    llm_skills.add(normalize_job_posting_text(skill))
    except Exception:
        pass

    skills_expanded = set()
    
    # Supplemental soft-skill pass. The main LLM prompt already extracts soft
    # skills; this keeps deterministic coverage for common explicit phrases.
    soft_skill_patterns = {
        'Communication': r'\b(communication|communicat(ing|ion|e)|speaking|presentation|verbal|written|writing|interpersonal)\b',
        'Leadership': r'\b(leadership|leading|leader|lead\s+team|mentor(ing)?|mentorship)\b',
        'Teamwork': r'\b(teamwork|team\s+player|collaboration|collaborative|working\s+in\s+team|cross-functional)\b',
        'Problem-solving': r'\b(problem[- ]solving|problem[- ]solver|analytical|troubleshooting|critical\s+thinking)\b',
        'Time Management': r'\b(time\s+management|project\s+management|organization(al)?|priorit(y|ization)|organizational|organized)\b',
        'Adaptability': r'\b(adaptab|flexible|flexibility|agile|willingness\s+to\s+learn|quick\s+learner|self[- ]directed|self[- ]starter)\b',
        'Customer Service': r'\b(customer\s+service|customer\s+support|client\s+relations?|stakeholder\s+management)\b',
        'Decision-making': r'\b(decision[- ]making|decision\s+maker|strategic\s+thinking|strategic)\b',
        'Creativity': r'\b(creativ|innovation|innovative|design\s+thinking|think\s+outside)\b',
        'Attention to Detail': r'\b(attention\s+to\s+detail|meticulous|detail[- ]oriented|quality)\b',
    }
    text_lower = candidate_text.lower()
    for skill_name, pattern in soft_skill_patterns.items():
        if re.search(pattern, text_lower):
            skills_expanded.add(skill_name)

    skills_expanded.update(llm_skills)

    return detect_skill_aliases(skills_expanded)

def extract_experience(text):
    """Extract experience-related phrases and the strongest years signal.

    Uses an LLM as the primary extractor, with regex as a deterministic
    supplement for common years-of-experience patterns.
    """
    if not text:
        return {"phrases": [], "years": None}

    candidate_text = _select_posting_sections(
        text,
        (
            "experience",
            "qualifications",
            "requirements",
            "minimum qualifications",
            "basic qualifications",
            "required qualifications",
            "preferred qualifications",
            "what we need to see",
            "ways to stand out",
        ),
    )

    llm_phrases = []
    llm_years = None
    try:
        model = ChatOpenAI(model="gpt-4o", temperature=0, timeout=60, max_retries=2)
        system_prompt = """You extract experience requirements from job postings and resumes.

Return ONLY valid JSON in this exact format:
{"phrases": ["experience phrase"], "years": 0}

Rules:
- Extract explicit experience requirements or experience claims.
- Include phrases such as "4+ years of experience" or "3-5 years professional experience".
- For "years", return the strongest minimum years-of-experience signal as an integer.
- For ranges like "3-5 years", use 3.
- For "4+ years", use 4.
- If multiple year requirements exist, use the highest minimum requirement.
- If no years signal exists, use null.
- Do not infer years from seniority words alone, such as junior, senior, staff, or lead.
- Do not include benefits, company culture, responsibilities, or education unless tied directly to experience."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract experience from this text:\n\n{candidate_text[:5000]}")
        ]
        response = model.invoke(messages)
        raw_content = (response.content or "").strip()
        raw_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.IGNORECASE)
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            phrases = parsed.get("phrases", [])
            years = parsed.get("years")
            if isinstance(phrases, list):
                llm_phrases = [
                    str(phrase).strip()
                    for phrase in phrases
                    if phrase and 2 <= len(str(phrase).strip()) <= 200
                ]
            if isinstance(years, int) and 0 <= years <= 30:
                llm_years = years
    except Exception:
        pass

    regex_years = []
    regex_phrases = []
    for match in _YOE_PATTERN.finditer(candidate_text):
        try:
            years = int(match.group(1))
            if 0 <= years <= 30:
                regex_years.append(years)
                regex_phrases.append(match.group(0).strip())
        except ValueError:
            continue

    phrases = []
    seen = set()
    for phrase in llm_phrases + regex_phrases:
        normalized = normalize_job_posting_text(phrase)
        if normalized and normalized not in seen:
            phrases.append(phrase)
            seen.add(normalized)

    year_candidates = []
    if llm_years is not None:
        year_candidates.append(llm_years)
    year_candidates.extend(regex_years)

    return {
        "phrases": phrases,
        "years": max(year_candidates) if year_candidates else None,
    }

def extract_education(text):
    """Extract degree levels using LLM plus regex fallback/supplement."""
    if not text or not text.strip():
        return []

    candidate_text = _select_posting_sections(
        text,
        (
            "education",
            "qualifications",
            "requirements",
            "minimum qualifications",
            "basic qualifications",
            "required qualifications",
            "preferred qualifications",
            "what we need to see",
        ),
    )

    llm_degrees = []
    try:
        model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)

        system_prompt = """You are an education requirements extraction expert. Extract ONLY degree types from the text.

Degree types include:
- Bachelor's degree (B.S., B.A., Bachelor of Science, etc.)
- Master's degree (M.S., M.A., M.B.A., Master of Science, etc.)
- PhD (Ph.D., Doctorate, etc.)
- Associate's degree (A.S., A.A., Associate's, etc.)
- Diploma, GED, High School
- Generic "Degree" when no specific type is mentioned

DO NOT include:
- Education fields/majors such as Computer Science or Engineering
- Years of experience
- Certifications
- Company names

Return a JSON object with field 'degrees' containing a list of extracted degree types.
Example: {"degrees": ["Bachelor's", "Master's"]}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract degree types from this text:\n\n{candidate_text[:4000]}")
        ]

        response = model.invoke(messages)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            for degree in parsed.get('degrees', []):
                degree_clean = str(degree).strip().strip('.,;/')
                if degree_clean and 2 <= len(degree_clean) <= 50:
                    llm_degrees.append(degree_clean)
    except Exception:
        pass

    # Removed ambiguous bare 2-letter forms (BS, BA, MS, MA, AS, AA) - they match
    # common English words under re.IGNORECASE (e.g. "as", "ma").
    # Trailing \b replaced with a lookahead so dot-ending abbreviations like
    # "B.S." and "Ph.D." (which end in a non-word char) are matched correctly.
    regex = re.compile(
        r"\b(B\.S\.|B\.A\.|Bachelor'?s?|"
        r"M\.S\.|M\.A\.|M\.B\.A\.|MBA|Master'?s?|"
        r"Ph\.D\.|PhD|Doctorate|"
        r"A\.S\.|A\.A\.|Associate'?s?|"
        r"High School|GED|Diploma|"
        r"Degree)(?=[\s,/|;:\n(]|$)",
        re.IGNORECASE
    )
    # spaCy does not recognize education as a named entity
    matches = regex.findall(candidate_text)
    return list(dict.fromkeys(llm_degrees + matches))

def extract_education_field(text):
    """Extract education field/major using LLM plus regex fallback/supplement.
    
    Looks for patterns like "Bachelor's in Computer Science" OR fields after labels like "Major:", "Field:"
    Handles various resume formats to be more robust, including bare field names.
    """
    if not text or not text.strip():
        return []

    candidate_text = _select_posting_sections(
        text,
        (
            "education",
            "qualifications",
            "requirements",
            "minimum qualifications",
            "basic qualifications",
            "required qualifications",
            "preferred qualifications",
            "what we need to see",
        ),
    )

    llm_validated_fields = []
    try:
        model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)

        system_prompt = """You are a STRICT education field extraction expert. Your task is to extract ONLY explicitly required education fields/majors from text.

ONLY EXTRACT IN DIRECT DEGREE STATEMENTS:
- "Bachelor's degree in Computer Science" -> extract "Computer Science"
- "Master's in Information Technology" -> extract "Information Technology"
- "Bachelor of Science in Sociology" -> extract "Sociology"
- "B.S. in Data Science" -> extract "Data Science"
- "B.S. Political Science" -> extract "Political Science"
- "Major in Psychology" -> extract "Psychology"
- "degree with a focus on Accounting" -> extract "Accounting"
- "degree from an accredited institution in Accounting" -> extract "Accounting"

NEVER EXTRACT from job duties/skills context:
- "Deep understanding of information technology solutions" -> do NOT extract, even though IT is a valid major
- "Experience with data science techniques" -> do NOT extract "Data Science"
- "Cloud platform management" -> do NOT extract "Cloud"
- "software development experience" -> do NOT extract "Software Development"
- Extracurricular or non-degree mentions of fields, such as "volunteered in a political science club" -> do NOT extract "Political Science" or "Political Science Club"

Broad requirements:
- "A Bachelor's degree in a STEM field" -> return an empty education_fields list
- "A Bachelor's degree in a related field" -> return an empty education_fields list
- "A Bachelor's degree in a humanities field" -> return an empty education_fields list

If the text says "Bachelor's degree" with NO specific field mentioned, return an empty list.

Output format: {"education_fields": ["Field1", "Field2"]} or {"education_fields": []} if none required."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract education fields/majors from this text:\n\n{candidate_text[:4000]}")
        ]

        response = model.invoke(messages)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            fields = parsed.get('education_fields', [])

            for field in fields:
                field_clean = str(field).strip().strip('.,;/')
                if not field_clean or len(field_clean) < 2 or len(field_clean) > 100 or field_clean.isdigit():
                    continue

                field_escaped = re.escape(field_clean)
                degree_patterns = [
                    r"(?:bachelor|master|phd|b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|m\.?b\.?a\.?|degree)[^\n]*?\s+(?:in|of|with)\s+" + field_escaped,
                    r"(?:major|field|concentration|discipline)\s*:?\s*" + field_escaped,
                    r"(?:bachelor|master|phd|b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|m\.?b\.?a\.?|degree)\s+(?:in|of|with)\s+[^;\n]*\b" + field_escaped + r"\b",
                ]

                if any(re.search(pattern, candidate_text, re.IGNORECASE) for pattern in degree_patterns):
                    llm_validated_fields.append(field_clean)
    except Exception:
        pass

    fields = []
    
    # Pattern 1: Match degree names followed by field (primary pattern)
    # Handles: "Bachelor's in Physics", "B.S. in Engineering", etc.
    # Use lookahead to stop at newline/punctuation
    pattern1 = r'\b(?:B\.?S\.?|B\.?A\.?|Bachelor\'?s?|M\.?S\.?|M\.?A\.?|M\.?B\.?A\.?|Master\'?s?|Ph\.?D\.?|Doctorate|A\.?S\.?|A\.?A\.?|Associate\'?s?|High School|GED|Diploma)\.?[ \t]+(?:degree[ \t]+)?(?:of[ \t]+Science[ \t]+in|in|of|with|related[ \t]+to)[ \t]+([A-Z][a-z]*(?:[ \t]+[A-Z][a-z]*){0,3})(?=\s*[\n,\.;:\(\)-]|\s*$)'
    
    matches1 = re.findall(pattern1, candidate_text, re.IGNORECASE)
    for match in matches1:
        field = match.strip().rstrip('.,;/')
        # Only keep if it's a reasonable field name (2-100 chars, not pure numbers/stopwords)
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'related', 'field'):
            fields.append(field)

    # Pattern 1b: Degree abbreviation followed DIRECTLY by field - no 'in' connector
    # Handles: "B.S. Computer Science, Math Minor" -> "Computer Science"
    #           "B.A. English Literature (GPA: 3.8)"  -> "English Literature"
    # Excluded status words (first word) to avoid: "M.S. Student," "Ph.D. Candidate,"
    # Anchored to line-start to prevent matching "MS Azure" in mid-sentence job posting text.
    _degree_status_words = {'student', 'candidate', 'applicant', 'expected', 'thesis', 'dissertation'}
    # (?:^|\n)\s* - start of line or after newline, optional whitespace
    pattern1b = r'(?:^|\n)\s*(?:B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|Ph\.?D\.?)\s+(?!(?:in|of|degree)\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})(?=\s*[,\(\n]|\s*$)'
    for match in re.findall(pattern1b, candidate_text, re.IGNORECASE):
        field = match.strip().rstrip('.,;/')
        first_word = field.split()[0].lower()
        if first_word in _degree_status_words:
            continue
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'related', 'field'):
            fields.append(field)
    
    # Pattern 2: Match fields after labels like "Major:", "Field:", "Concentration:", etc.
    # Stop at newline, punctuation, or other structural breaks
    # NOTE: Added \b word boundaries to prevent matching "Field" inside "fields" (job posting garbage)
    pattern2 = r'\b(?:Major|Field(?:\s+of\s+study)?|Specialization|Concentration|Discipline|Subject)\b\s*:?\s*([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,3})(?=[\n,\.;:\(\)-]|$)'
    matches2 = re.findall(pattern2, candidate_text, re.IGNORECASE)
    for match in matches2:
        field = match.strip().rstrip('.,;/')
        # Only keep if it's a reasonable field name (at least 2 chars, exclude common stopwords)
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'unknown', 'other', 'general studies'):
            fields.append(field)
    
    # Pattern 3: Match fields after "Education" with degree, stopping at newline
    # This helps catch cases where the field is listed in a formal format
    pattern3 = r'(?:Education|Degree)\s*[:\-]?\s*(?:B\.?S\.?|B\.?A\.?|Bachelor\'?s?|M\.?S\.?|M\.?A\.?|M\.?B\.?A\.?|Master\'?s?|Ph\.?D\.?|Doctorate)\.?[ \t]+in[ \t]+([A-Z][a-z]*(?:[ \t]+[A-Z][a-z]*){0,2})(?=\s*[\n,\.;:\(\)]|\s*$)'
    matches3 = re.findall(pattern3, candidate_text, re.IGNORECASE)
    for match in matches3:
        field = match.strip().rstrip('.,;/')
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree'):
            fields.append(field)
    
    # Pattern 4: Match single-line field names that appear right after a degree keyword and newline
    # Handles: "Education\nMicrobiology", "Bachelor's\nMicrobiology", "Bachelor of Science\nMicrobiology"
    # Must stop at next newline or punctuation
    pattern4 = r'\b(?:Education|Bachelor\'?s?|Bachelor\s+of\s+Science|Master\'?s?|Master\s+of\s+Science|Degree|B\.?S\.?|B\.?A\.?|M\.?S\.?|Ph\.?D\.?)\b\s*[:\-]?\s*\n\s*([A-Z][a-z]*(?:[ \t]+[A-Z][a-z]*){0,2})(?=\n|,|$)'
    matches4 = re.findall(pattern4, candidate_text, re.IGNORECASE)
    for match in matches4:
        field = match.strip().rstrip('.,;/')
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'university', 'college', 'gpa', 'cum', 'laude', 'honors'):
            fields.append(field)
    
    # Pattern 5: Match capitalized field names on a new line after degree info, with bullet or dash
    # Handles: "* B.S. in\n* Microbiology", "* Bachelor of Science\n* Microbiology"
    # Only capture up to 2 words and stop at newline
    # Anchored to line-start to prevent A\.?S\.? matching "as" inside words like "areas".
    pattern5 = r'(?:^|\n)\s*(?:B\.?S\.?|B\.?A\.?|Bachelor\'?s?|Bachelor\s+of\s+Science|M\.?S\.?|M\.?A\.?|M\.?B\.?A\.?|Master\'?s?|Master\s+of\s+Science|Ph\.?D\.?|A\.?S\.?|A\.?A\.?|Associate\'?s?|Diploma|GED)(?:[ \t]+in)?\s*\n\s*[\-\*]?\s*([A-Z][a-z]*(?:[ \t]+[A-Z][a-z]*){0,2})(?=\n|,|$)'
    matches5 = re.findall(pattern5, candidate_text, re.IGNORECASE)
    for match in matches5:
        field = match.strip().rstrip('.,;/')
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'gpa', 'cum', 'laude', 'honors', 'university', 'college'):
            fields.append(field)
    
    # Pattern 6: Match fields with inline dates/multiple spaces/parentheticals after field name
    # Handles: "Bachelor of Science in Microbiology                                May 2026"
    #           "Bachelor of Science in Computer Science (GPA: 3.5/4.0) Current - May 2026"
    # Stop at optional whitespace then a paren/newline/comma, OR multiple spaces, OR end of string
    pattern6 = r'(?:Bachelor\s+of\s+Science|B\.?S\.?|Master\s+of\s+Science|M\.?S\.?)\s+in\s+([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,2})(?=\s*[\(\n,\.;]|\s{2,}|\s*$)'
    matches6 = re.findall(pattern6, candidate_text, re.IGNORECASE)
    for match in matches6:
        field = match.strip().rstrip('.,;/')
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'gpa', 'cum', 'laude', 'honors', 'university', 'college'):
            fields.append(field)
    
    # Pattern 7: Match fields in degree requirements with parenthetical content
    # Handles: "Master's or PhD degree (or equivalent experience) in Computer Science"
    # This catches job postings that have alternate/equivalent experience noted in parentheses
    pattern7 = r'(?:Master\'?s?|PhD|Bachelor\'?s?|Ph\.?D\.?)\s+(?:or\s+PhD\s+)?degree\s*(?:\([^)]*\))?\s+in\s+([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,3})(?=[,\n\.;])'
    matches7 = re.findall(pattern7, candidate_text, re.IGNORECASE)
    for match in matches7:
        field = match.strip().rstrip('.,;/')
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'gpa', 'cum', 'laude', 'honors', 'university', 'college'):
            fields.append(field)
            
            # Pattern 8: Extract additional comma/or-separated fields in the same clause
            # Handles: "in Computer Science, Computer Engineering, or Electrical Engineering"
            # Scope is limited to the current sentence only (up to . ; newline or " OR ")
            # to prevent scanning the rest of the document and picking up tool names.
            start_idx = candidate_text.find(match)
            if start_idx != -1:
                # Find end of this sentence/clause - stop at sentence-ending punctuation or " OR "
                sentence_end = re.search(r'(?:\.|;|\n|\bOR\b)', candidate_text[start_idx:], re.IGNORECASE)
                end_idx = start_idx + sentence_end.start() if sentence_end else start_idx + 300
                remaining_text = candidate_text[start_idx:end_idx]
            else:
                remaining_text = ""
            additional_fields = re.findall(r',\s*(?:or\s+)?([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,3})(?=[,\n\.;]|$)', remaining_text)
            for additional in additional_fields:
                field_clean = additional.strip().rstrip('.,;/')
                if 2 <= len(field_clean) <= 100 and not field_clean.isdigit() and field_clean.lower() not in ('science', 'degree', 'gpa', 'cum', 'laude', 'honors', 'university', 'college'):
                    fields.append(field_clean)
    
    # Final validation: reject garbage fields (< 4 chars, lowercase start, stopwords, activity names)
    valid_fields = []
    for field in fields:
        # Skip if less than 4 characters (likely garbage like "s" or "it")
        if len(field) < 4:
            continue
        # Skip if starts with lowercase (indicates corrupted capture)
        if field and field[0].islower():
            continue
        # Skip if only stopwords
        stopwords = {'and', 'of', 'in', 's', 'industries', 'or', 'to', 'the', 'a', 'an'}
        if field.lower() in stopwords:
            continue
        valid_fields.append(field)
    
    return list(dict.fromkeys(llm_validated_fields + valid_fields))  # Remove duplicates while preserving order

def match_education(job_required_education, job_preferred_education, resume_education, resume_education_fields):
    """Match education requirements from job posting against resume education.
    
    Args:
        job_required_education: list of required degree types (e.g., ['Bachelor\'s', 'B.S.'])
        job_preferred_education: list of preferred degree types
        resume_education: list of degrees found in resume
        resume_education_fields: list of education fields/majors found in resume
    
    Returns:
        dict with matched degrees and fields
    """
    result = {
        'required_degree_matched': False,
        'required_degree_job': job_required_education,
        'required_degree_resume': resume_education,
        'preferred_degree_matched': False,
        'preferred_degree_job': job_preferred_education,
        'education_field_job': [],
        'education_field_resume': resume_education_fields,
        'field_matched': False,
        'field_match_details': None,
        'details': {}
    }

    degree_rank = {
        'high_school': 0,
        'associates': 1,
        'bachelors': 2,
        'masters': 3,
        'phd': 4,
    }
    
    def normalize_degree(degree):
        """Map a degree to its general category."""
        d = normalize_job_posting_text(str(degree).replace("\u2019", "'")).strip(".,;/")

        if re.search(r"\b(ph\.?\s*d\.?|phd|doctorate|doctoral)\b", d):
            return 'phd'
        if re.search(r"\b(m\.?\s*b\.?\s*a\.?|m\.?\s*s\.?|m\.?\s*a\.?|masters?|master's)\b", d):
            return 'masters'
        if re.search(r"\b(b\.?\s*s\.?|b\.?\s*a\.?|bachelors?|bachelor's)\b", d):
            return 'bachelors'
        if re.search(r"\b(a\.?\s*s\.?|a\.?\s*a\.?|associates?|associate's)\b", d):
            return 'associates'
        if re.search(r"\b(high school|diploma|ged)\b", d):
            return 'high_school'
        return d

    def degree_match_by_level(job_degrees, resume_degrees):
        """Return True when the resume has at least the minimum required degree level.

        If a posting says "Master's or PhD degree", a Bachelor's resume should not
        match. If a posting says "Bachelor's degree", a Master's resume should match.
        Generic "degree" only means any degree when no specific level is present.
        """
        job_normalized = {normalize_degree(d) for d in job_degrees}
        resume_normalized = {normalize_degree(d) for d in resume_degrees}

        job_levels = [
            degree_rank[d]
            for d in job_normalized
            if d in degree_rank
        ]
        resume_levels = [
            degree_rank[d]
            for d in resume_normalized
            if d in degree_rank
        ]

        if not job_levels:
            return (
                bool(resume_degrees) and 'degree' in job_normalized,
                job_normalized,
                resume_normalized,
                'Generic degree requirement matched by any resume degree',
            )

        if not resume_levels:
            return False, job_normalized, resume_normalized, None

        return (
            max(resume_levels) >= min(job_levels),
            job_normalized,
            resume_normalized,
            None,
        )
    
    # Check required degrees
    if job_required_education:
        matched, job_normalized, resume_normalized, note = degree_match_by_level(
            job_required_education,
            resume_education,
        )
        result['required_degree_matched'] = matched
        result['details']['required'] = {
            'job_normalized': list(job_normalized),
            'resume_normalized': list(resume_normalized),
            'match': matched,
        }
        if note:
            result['details']['required']['note'] = note
    
    # Check preferred degrees
    if job_preferred_education:
        matched, job_normalized, resume_normalized, note = degree_match_by_level(
            job_preferred_education,
            resume_education,
        )
        result['preferred_degree_matched'] = matched
        result['details']['preferred'] = {
            'job_normalized': list(job_normalized),
            'resume_normalized': list(resume_normalized),
            'match': matched,
        }
        if note:
            result['details']['preferred']['note'] = note
    
    # Check field/major matching (if we have both job and resume fields)
    if resume_education_fields:
        result['education_field_resume'] = resume_education_fields
        # Simple fuzzy match for fields - check for common keywords
        resume_fields_lower = [f.lower() for f in resume_education_fields]
        result['field_match_details'] = {
            'resume_fields': resume_education_fields,
            'match_found': False
        }
        # If we later get job_required_education_fields, we can enhance this matching
    
    return result

def extract_qualifications(text):
    """Extract REQUIRED and PREFERRED qualifications from a job posting.

    LLM extraction is primary. The regex section parser mirrors the original
    implementation and supplements/falls back when the LLM misses items or is
    unavailable.
    """
    def _dedupe(items):
        cleaned = []
        seen = set()
        for item in items:
            if not isinstance(item, str):
                continue
            item = item.strip().strip("-*+ \t")
            item = re.sub(r"\s+", " ", item).strip()
            key = item.lower()
            if len(item) > 5 and key not in seen:
                cleaned.append(item)
                seen.add(key)
        return cleaned

    def _extract_with_regex(job_text):
        requirements = []
        preferences = []
        current_section = None
        lines = job_text.split('\n')

        for line in lines:
            line_lower = line.lower().strip()

            if (
                'required qualifications' in line_lower
                or 'basic qualifications' in line_lower
                or 'required skills' in line_lower
                or 'qualifications you must have' in line_lower
            ):
                current_section = 'required'
                continue
            if (
                'preferred qualifications' in line_lower
                or 'preferred skills' in line_lower
                or 'desired skills' in line_lower
                or 'nice to have' in line_lower
                or 'qualifications we prefer' in line_lower
            ):
                current_section = 'preferred'
                continue
            if re.search(r'(responsibilities|key responsibilities|offer|about|interview|technical skills|what we offer)', line_lower):
                current_section = None
                continue

            if current_section:
                stripped = line.strip()
                bullet_match = re.match(r'^[^\w\s]+\s+(.+)', stripped)
                if bullet_match:
                    item = bullet_match.group(1).strip()
                elif stripped and line.startswith((' ', '\t')) and not line_lower.startswith(('required', 'preferred')):
                    item = stripped
                else:
                    item = None

                if item and len(item) > 5:
                    if current_section == 'required':
                        requirements.append(item)
                    elif current_section == 'preferred':
                        preferences.append(item)

        return _dedupe(requirements), _dedupe(preferences)

    if not text or not text.strip():
        return [], []

    candidate_text = _select_posting_sections(
        text,
        (
            "qualifications",
            "requirements",
            "minimum qualifications",
            "basic qualifications",
            "required qualifications",
            "preferred qualifications",
            "skills required",
            "what we need to see",
            "ways to stand out",
        ),
    )

    regex_required, regex_preferred = _extract_with_regex(candidate_text)
    llm_required = []
    llm_preferred = []

    try:
        model = ChatOpenAI(model="gpt-4o", temperature=0, timeout=60, max_retries=2)
        system_prompt = """You are a job posting qualification extraction expert.
Extract only candidate qualifications from the job posting.

Return ONLY valid JSON:
{
  "required": ["required qualification 1"],
  "preferred": ["preferred qualification 1"]
}

Rules:
- Required/basic/minimum/must-have qualifications go in "required".
- Preferred/nice-to-have/desired qualifications go in "preferred".
- Include education, experience, eligibility, technologies, skills, certifications, and domain requirements when phrased as qualifications.
- Do not include responsibilities, benefits, compensation, company culture, interview process, or equal opportunity language.
- Preserve the job posting wording as much as possible.
- If a category is absent, return an empty list for that category."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract qualifications from this job posting:\n\n{candidate_text[:6000]}"),
        ]
        response = model.invoke(messages)
        raw_content = (response.content or "").strip()
        raw_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.IGNORECASE)
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            llm_required = parsed.get("required", [])
            llm_preferred = parsed.get("preferred", [])
    except Exception:
        pass

    requirements = _dedupe(llm_required + regex_required)
    preferences = _dedupe(llm_preferred + regex_preferred)

    return requirements, preferences

# -- Seniority helpers --------------------------------------------------------

def _rule_based_job_seniority(text):
    """Rule-based fallback for seniority extraction. Used when LLM is unavailable."""
    text_lower = text.lower()

    # 'graduate', 'fresh', 'fresher' removed - too generic; they appear in senior postings
    # ("graduate degree required", "fresh perspectives") and caused false entry-level hits.
    entry_level_keywords = {
        'entry-level', 'entry level', 'intern', 'internship',
        'no experience required', '0-2 years', '0 years', 'new college grad', 'recent grad'
    }
    senior_keywords = {
        'senior', '5+ years', '6+ years', '7+ years', '8+ years',
        'experienced', 'expert level', '8-10 years', '5-7 years', '6-10 years'
    }
    mid_level_keywords = {
        'mid-level', 'mid level', 'intermediate', '3-5 years', '3 years',
        '4 years', '5 years', 'associate', 'professional'
    }
    lead_principal_keywords = {
        'lead', 'principal', 'staff', 'architect', '10+ years', 'head of',
        'director', 'manager', '15+ years', '12+ years'
    }

    for keyword in entry_level_keywords:
        if keyword in text_lower:
            return 'entry-level'
    for keyword in senior_keywords:
        if keyword in text_lower:
            return 'senior'
    for keyword in mid_level_keywords:
        if keyword in text_lower:
            return 'mid-level'
    for keyword in lead_principal_keywords:
        if keyword in text_lower:
            return 'lead/principal'

    years = extract_experience(text)["years"]
    if years is not None:
        return _yoe_to_seniority(years, text)

    return None


# Pattern for "N years" / "N+ years" / "N-M years" of experience.
# Captures the FIRST number (the floor) since "4+" and "4-6" both imply >= 4.
_YOE_PATTERN = re.compile(
    r'(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2})?\s*(?:to\s*\d{1,2}\s*)?years?\s+(?:of\s+)?(?:professional\s+|relevant\s+|industry\s+)?(?:experience|exp)\b',
    re.IGNORECASE,
)


def _yoe_to_seniority(years, title=None):
    """Map a years-of-experience number to a seniority bucket.

    Buckets match the schema used by extract_job_seniority:
      0-1 yrs  -> entry-level
      2-5 yrs  -> mid-level (2-year boundary can remain entry-level for junior titles)
      6+ yrs  -> either senior or lead/principal depending on title keywords
    """
    if years is None:
        return None
    if years <= 1:
        return 'entry-level'
    if years == 2:
        title_seniority = _seniority_from_title(title)
        if title_seniority == 'entry-level':
            return 'entry-level'
        return 'mid-level'
    if years <= 5:
        return 'mid-level'
    if years >= 6:
        title_seniority = _seniority_from_title(title)
        if title_seniority == 'lead/principal':
            return 'lead/principal'
        return 'senior'

_TITLE_SENIORITY_KEYWORDS = {
    'entry-level': (
        'intern', 'internship', 'junior', 'entry-level', 'entry level',
        'new grad', 'new college grad', 'recent grad'
    ),
    'senior': ('senior', 'sr.'),
    'lead/principal': (
        'lead', 'principal', 'staff', 'architect', 'manager', 'director',
        'head of', 'vp', 'vice president', 'chief'
    ),
}


def _seniority_from_roman_numeral(title):
    """Map Roman numeral title levels to seniority buckets, ignoring I."""
    if not title:
        return None

    match = re.search(r"(?<![a-z0-9])(ii|iii|iv|v|vi|vii|viii|ix|x)(?![a-z0-9])", title.lower())
    if not match:
        return None

    numeral = match.group(1)
    if numeral == 'ii':
        return 'mid-level'
    if numeral in {'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x'}:
        return 'senior'


def split_posting_into_sections(text, include_preamble=False):
    """Split a job posting into sections based on common headings.

    Returns:
        dict: mapping of section name to section text.
        If include_preamble=True, includes "__preamble__" with text before the first section.
    """
    if not text:
        return {}

    # Common section headings in job postings
    section_headings = [
        'about the company', 'about us', 'company overview', 'company description',
        'full job description', 'job description', 'about the job', 'about this job',
        'about the role', 'about this role', 'role overview', 'role summary',
        'summary', 'overview', 'responsibilities', 'what you will do',
        "what you'll do", 'what you will be doing', "what you'll be doing",
        'what we need to see', 'qualifications', 'requirements', 'skills required',
        'minimum qualifications', 'basic qualifications', 'preferred qualifications',
        'required qualifications', 'benefits', 'perks', 'compensation', 'salary',
        'ways to stand out', 'how to apply', 'application process',
        'equal opportunity employer'
    ]

    # Create a regex pattern to match headings
    heading_pattern = r'^\s*(?:' + '|'.join(re.escape(h) for h in section_headings) + r')\s*[:\-]?\s*$'
    heading_regex = re.compile(heading_pattern, re.IGNORECASE | re.MULTILINE)

    sections = {}
    current_section = None
    current_text = []
    preamble_lines = []  # Text before the first recognized section heading

    for line in text.splitlines():
        if heading_regex.match(line):
            if include_preamble and current_section is None and preamble_lines:
                sections["__preamble__"] = '\n'.join(preamble_lines).strip()
            if current_section and current_text:
                sections[current_section] = '\n'.join(current_text).strip()
            current_section = line.strip().lower()
            current_text = []
        else:
            if current_section:
                current_text.append(line)
            elif include_preamble:
                preamble_lines.append(line)

    if current_section and current_text:
        sections[current_section] = '\n'.join(current_text).strip()
    elif include_preamble and preamble_lines and "__preamble__" not in sections:
        sections["__preamble__"] = '\n'.join(preamble_lines).strip()

    return sections

def _extract_job_title(text):
    """Extract a job title from pasted/OCR job-description text.

    Layer 1 removes obvious job-board/UI noise from the top of the posting.
    Layer 2 returns explicit seniority-title lines before the LLM can drop them.
    Layer 3 asks the LLM to extract the title from the cleaned top section.
    Layer 4 uses conservative title-like line heuristics as a fallback.
    """
    if not text:
        return None

    noise_keywords = {
        "about the job", "about us", "jobs", "apply", "save", "share", "send inmail",
        "linkedin", "indeed", "posted", "reposted", "followers", "applicants",
        "company", "overview", "benefits", "qualifications", "responsibilities",
        "people also viewed", "promoted", "easy apply", "see who", "show more",
        "meet the hiring team", "job activity", "report this job", "sign in",
        "create job alert", "similar jobs", "view all jobs", "employment type",
        "job function", "industries", "seniority level", "workplace type",
        "full-time", "part-time", "contract", "temporary",
        "on-site", "remote", "hybrid", "ago", "be among the first",
        "years of experience", "year of experience", "minimum qualifications",
        "preferred qualifications", "required qualifications", "basic qualifications",
        "what you will be doing", "what we need to see", "ways to stand out",
    }

    def clean_line(line):
        return re.sub(r"\s+", " ", line).strip(" -|\u2022\t")

    def is_noise_line(line):
        normalized = line.lower()
        if not normalized:
            return True
        if len(line) > 140:
            return True
        if any(noise in normalized for noise in noise_keywords):
            return True
        return False

    def has_title_shape(line):
        normalized = line.lower()
        words = line.split()
        if not words or len(words) > 12 or len(line) > 120:
            return False
        if is_noise_line(line):
            return False
        if normalized.startswith(("you will", "we are", "we're", "this role", "the role", "if you", "with ")):
            return False
        if normalized.endswith((".", ":", ";")):
            return False
        if re.search(r"\b\d+\+?\s+years?\b", normalized):
            return False
        return True

    sections = split_posting_into_sections(text, include_preamble=True)
    title_source_text = sections.get("__preamble__", text)

    lines = [clean_line(line) for line in title_source_text.splitlines()]
    lines = [line for line in lines if line]
    cleaned_top_lines = [line for line in lines[:40] if not is_noise_line(line)]
    cleaned_top_text = "\n".join(cleaned_top_lines[:20]).strip()

    # Layer 2: explicit seniority title line. This must happen before the LLM
    # because the LLM may simplify "Senior Software Engineer" to "Software Engineer".
    for line in cleaned_top_lines:
        if _seniority_from_title(line, is_noise_line=is_noise_line, validate_title_shape=True):
            return line

    # Layer 3: LLM title extraction from cleaned top content.
    if cleaned_top_text:
        try:
            model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
            system_prompt = """You extract job titles from job postings.

Return ONLY valid JSON in this exact format:
{"job_title": "Senior Software Engineer"}

Rules:
- Extract the role title only.
- Do not return company names, job-board UI text, locations, benefits, responsibilities, or paragraphs.
- If no title is present, return {"job_title": null}."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Extract the job title from this cleaned top section:\n\n{cleaned_top_text[:1200]}"),
            ]
            response = model.invoke(messages)
            json_match = re.search(r"\{.*?\}", response.content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                job_title = parsed.get("job_title")
                if isinstance(job_title, str):
                    job_title = clean_line(job_title)
                    if has_title_shape(job_title):
                        return job_title
        except Exception:
            pass

    # Layer 4: conservative fallback heuristic. Without hardcoded role terms,
    # only return short, clean preamble lines and let the LLM handle broad titles.
    for line in cleaned_top_lines:
        if has_title_shape(line):
            return line

    return None


def _seniority_from_title(title, is_noise_line=None, validate_title_shape=False):
    """Classify seniority from a job title using deterministic title signals."""
    if not title:
        return None

    normalized = title.lower()

    if validate_title_shape:
        words = title.split()
        if not words or len(words) > 12 or len(title) > 120:
            return None
        if is_noise_line and is_noise_line(title):
            return None
        if normalized.startswith(("you will", "we are", "we're", "this role", "the role", "if you", "with ")):
            return None
        if normalized.endswith((".", ":", ";")):
            return None

    for seniority, keywords in _TITLE_SENIORITY_KEYWORDS.items():
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])", normalized)
            for keyword in keywords
        ):
            return seniority

    roman_seniority = _seniority_from_roman_numeral(title)
    if roman_seniority:
        return roman_seniority

    return None


def extract_job_seniority(text):
    """Extract seniority level requirement from a job posting.

    Hybrid pipeline:
      Layer 1 - Explicit title keywords, such as "Senior Engineer", are trusted.
      Layer 2 - Non-I Roman numerals in the title, such as "Engineer II", map
        to tiered levels (II=mid-level, III/IV=senior, V+=lead/principal)
        unless a stronger explicit title keyword is present.
    Layer 3 - Years-of-experience regex is used when earlier layers do not decide.
    Layer 4 - GPT-4o classifies unclear postings using title/context.
      Layer 5 - Rule-based keyword scan is the final fallback.

    Returns:
    str: one of 'entry-level', 'mid-level', 'senior', 'lead/principal', or None
    """

    if not text or not text.strip():
        return None

    candidate_text = _select_posting_sections(
        text,
        (
            "qualifications",
            "requirements",
            "minimum qualifications",
            "basic qualifications",
            "required qualifications",
            "preferred qualifications",
            "experience",
            "seniority",
            "what we need to see",
            "ways to stand out",
        ),
        include_preamble=True,
    )

    # -- Layer 1: explicit title seniority -------------------------------------
    title = _extract_job_title(text)
    title_seniority = _seniority_from_title(title)
    if title_seniority:
        return title_seniority

    # -- Layer 3: deterministic YoE extraction + title-keyword refinement ------
    experience_data = extract_experience(candidate_text)
    yoe = experience_data["years"]
    if yoe is not None:
        bucket = _yoe_to_seniority(yoe, title)
        if bucket != 'lead/principal':
            return bucket

        if _seniority_from_title(title) == 'lead/principal':
            return 'lead/principal'
        return 'senior'

    # -- Layer 4: LLM ----------------------------------------------------------
    llm_seniority = None
    try:
        model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
        system_prompt = """You are a job posting analyst. Classify the seniority level required by the posting into exactly one of these four categories:

- "entry-level"    : 0-2 years experience, intern / graduate / junior roles
- "mid-level"      : 3-5 years, associate / intermediate roles
- "senior"         : 6-9 years, senior individual-contributor roles (NOT management)
- "lead/principal" : 10+ years AND explicit management / leadership responsibility
                     (manager, director, lead, principal, staff, architect, VP)

Classify based on the full posting, including title, level keywords, and stated
years of experience when present:
- "Junior X", "X Intern", "New Grad" -> entry-level
- "X I" (no level prefix) -> do not infer entry-level from "I" alone
- "X II", "X III", etc. (no level prefix) -> mid-level
- "Senior X", "Sr. X" -> senior
- "Lead X", "Principal X", "Staff X", "X Manager", "X Director" -> lead/principal

If the title gives no signal at all (e.g. just "Software Engineer" with no
context), default to "mid-level".

Return ONLY valid JSON: {"seniority": "<level>"}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Classify the seniority level for this job posting:\n\n{candidate_text[:3000]}")
        ]
        response = model.invoke(messages)
        json_match = re.search(r'\{.*?\}', response.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            llm_seniority = parsed.get('seniority')
            if llm_seniority not in {'entry-level', 'mid-level', 'senior', 'lead/principal'}:
                llm_seniority = None
    except Exception:
        pass

    if llm_seniority:
        return llm_seniority

    # -- Layer 5: rule-based fallback ------------------------------------------
    return _rule_based_job_seniority(candidate_text)

def match_seniority(job_seniority, resume_seniority):
    """Match resume seniority against job seniority requirement.
    
    Returns a dict with match status and recommendations.
    
    Rules:
    - Entry-level job: accepts all levels (entry, mid, senior)
    - Mid-level job: accepts mid-level and above (mid, senior, lead)
    - Senior job: accepts senior and above (senior, lead/principal)
    - Lead/Principal job: only accepts lead/principal
    
    Args:
        job_seniority: Job posting seniority level
        resume_seniority: Resume seniority level
    
    Returns:
        dict with match status, details, and warnings
    """
    seniority_order = ['entry-level', 'mid-level', 'senior', 'lead/principal']
    
    result = {
        'job_seniority': job_seniority,
        'resume_seniority': resume_seniority,
        'is_match': False,
        'is_overqualified': False,
        'is_underqualified': False,
        'warning': None,
        'recommendation': None
    }
    
    # If either is missing, can't determine a match
    if not job_seniority or not resume_seniority:
        result['is_match'] = False
        if not job_seniority:
            result['warning'] = "Could not determine job seniority level from posting"
        if not resume_seniority:
            result['warning'] = "Could not determine seniority level from resume"
        return result
    
    job_idx = seniority_order.index(job_seniority) if job_seniority in seniority_order else -1
    resume_idx = seniority_order.index(resume_seniority) if resume_seniority in seniority_order else -1
    
    if job_idx == -1 or resume_idx == -1:
        result['is_match'] = False
        return result
    
    # Check if resume meets job requirement
    if resume_idx >= job_idx:
        result['is_match'] = True
    
    # Check if overqualified (resume is significantly higher than job)
    if resume_idx > job_idx + 1:
        result['is_overqualified'] = True
        result['warning'] = f"You appear overqualified for this {job_seniority} role (your background suggests {resume_seniority})"
        result['recommendation'] = "Consider highlighting how your experience brings value, or look for senior opportunities"
    elif resume_idx < job_idx:
        result['is_underqualified'] = True
        result['warning'] = f"Your experience level ({resume_seniority}) may be below the job requirement ({job_seniority})"
        result['recommendation'] = "Consider highlighting growth, recent projects, or adjacent experience"
    elif result['is_match']:
        result['recommendation'] = f"Your {resume_seniority} experience aligns well with this {job_seniority} role"
    
    return result


def check_matched_skills(job_skills, resume_skills):
    """Check which job skills are matched by the resume skills.

    Args:
        job_skills: List of skills required/preferred by the job posting
        resume_skills: List of skills extracted from the resume

    Returns:
        dict with matched and unmatched skills, and a match percentage
    """
    job_skills_set = {
        normalized
        for skill in job_skills
        for normalized in [normalize_job_posting_text(skill)]
        if normalized
    }
    resume_skills_set = {
        normalized
        for skill in resume_skills
        for normalized in [normalize_job_posting_text(skill)]
        if normalized
    }

    matched_skills = job_skills_set.intersection(resume_skills_set)
    unmatched_skills = job_skills_set.difference(resume_skills_set)

    match_percentage = (len(matched_skills) / len(job_skills_set) * 100) if job_skills_set else 0

    return {
        'matched_skills': sorted(matched_skills),
        'unmatched_skills': sorted(unmatched_skills),
        'match_percentage': match_percentage
    }


def main():
    """Run all extraction functions across text job postings."""
    job_postings_dir = Path("tests/job_postings")
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}

    if not job_postings_dir.exists():
        print(f"Job postings folder not found: {job_postings_dir}")
        return

    posting_paths = sorted(
        path for path in job_postings_dir.iterdir()
        if path.is_file() and path.suffix.lower() not in image_extensions
    )

    if not posting_paths:
        print(f"No non-image job postings found in {job_postings_dir}")
        return

    print(f"Testing extraction functions for {len(posting_paths)} posting(s)")
    print("=" * 100)

    for index, posting_path in enumerate(posting_paths, start=1):
        print(f"\nPosting {index}/{len(posting_paths)}: {posting_path.name}")
        try:
            text = posting_path.read_text(encoding="utf-8", errors="ignore")
            results = {}

            extractor_calls = [
                ("job_title", lambda t: _extract_job_title(t)),
                ("skills", lambda t: extract_skills(t)),
                ("experience", lambda t: extract_experience(t)),
                ("education", lambda t: extract_education(t)),
                ("education_fields", lambda t: extract_education_field(t)),
                ("qualifications", lambda t: extract_qualifications(t)),
                ("seniority", lambda t: extract_job_seniority(t)),
            ]

            for name, extractor in extractor_calls:
                try:
                    results[name] = extractor(text)
                except Exception as exc:
                    results[name] = f"ERROR: {type(exc).__name__}: {exc}"

            print(f"- job_title: {results['job_title']}")

            skills = results["skills"]
            if isinstance(skills, list):
                print(f"- skills ({len(skills)}): {skills}")
            else:
                print(f"- skills: {skills}")

            print(f"- experience: {results['experience']}")

            education = results["education"]
            if isinstance(education, list):
                print(f"- education ({len(education)}): {education}")
            else:
                print(f"- education: {education}")

            education_fields = results["education_fields"]
            if isinstance(education_fields, list):
                print(f"- education_fields ({len(education_fields)}): {education_fields}")
            else:
                print(f"- education_fields: {education_fields}")

            qualifications = results["qualifications"]
            if (
                isinstance(qualifications, tuple)
                and len(qualifications) == 2
                and isinstance(qualifications[0], list)
                and isinstance(qualifications[1], list)
            ):
                required, preferred = qualifications
                print(f"- qualifications.required ({len(required)}): {required}")
                print(f"- qualifications.preferred ({len(preferred)}): {preferred}")
            else:
                print(f"- qualifications: {qualifications}")

            print(f"- seniority: {results['seniority']}")

        except Exception as exc:
            print(f"- ERROR: {type(exc).__name__}: {exc}")

        print("=" * 100)


if __name__ == "__main__":
    main()
