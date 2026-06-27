"""
NLP utilities for resume and job description analysis.
Handles text parsing, entity extraction, skill matching, etc.
"""
import re
import os
import json
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
import spacy # type: ignore
from difflib import SequenceMatcher
from dotenv import load_dotenv  # type: ignore
from langchain.chat_models import init_chat_model  # type: ignore

# Load environment variables
load_dotenv()

nlp = spacy.load("en_core_web_sm")
logger = logging.getLogger(__name__)

# Load sample text from file
sample_file = Path(__file__).parent.parent / "tests" / "sample_company.txt"
text = sample_file.read_text(encoding="utf-8") if sample_file.exists() else ""

compensation_keywords = {
    'salary', 'equity', 'bonus', 'pto', 'benefits', 
    'package', 'stipend', 'budget', 'matching', '401k',
    'compensation', '(k)', 'parental leave'
}

compensation_patterns = [
    r'^\$[\d,]+',           # $180,000
    r'unlimited\s*\w+',     # unlimited pto, unlimited vacation
    r'\d+%?\s*(match|equit)',  # 6% match, equity package
]

generic_words = {
    # Generic nouns
    'ability', 'knowledge', 'code', 'services', 'systems', 'infrastructure',
    'requirements', 'apis', 'engineering', 'ci', 'cs',
    'the', 'role', 'your', 'our', 'work', 'team', 'project',
    # Adjectives that describe quality, not skills
    'significant', 'excellent', 'strong', 'high', 'deep', 'driven',
    # HR/Growth words
    'opportunity', 'opportunities', 'growth', 'career',
    # Generic descriptors
    'equal', 'employer', 'diverse', 'inclusive',
    'internal', 'external', 'consumers', 'users', 'thousands',
    # Responsibility terms
    'support', 'production support',
    # Other filler
    'management', 'planning', 'implementation', 'technology', 'problems',
    'concepts', 'scientists', 'employers', 'portfolio', 'legacy', 'teams',
    'frameworks',
    # HR/legal/diversity language
    'consideration', 'candidates', 'belief', 'regard', 'disability', 'disabilities',
    'employment', 'family', 'gender identity', 'national origin', 'race', 'religion',
    'sex', 'sexual orientation', 'veteran status', 'all qualified applicants',
    'any other legally-protected characteristic', 'color', 'dental', 'exposure',
    'feasibility', 'proficiency', 'vision', 'this position', 'this range',
    'this reasonable accommodations form', 'department overview',
    'job description', 'people', 'property', 'places',
    'paid parental', 'pay transparency', 'their most authentic self',
    'unit', 'integration',
    # Generic verbs/adjectives (too vague for skills)
    'deployment', 'ownership', 'methodology', 'feasibility',
    'performance', 'optimization', 'automation',
    'subsystem design', 'system testing', 'audio',
    # Generic action/responsibility terms (appear in most job postings)
    'credentials', 'testing', 'validation',
    # Punctuation artifacts
    '(ooad',
    # Misc 
    'date', 'time', 'schedule', 'deadline', 'budget', 'resource',
    # REMOVED: Company-specific mission/values language (too brittle across companies)
    # Instead, using pattern-based detection in is_mission_language()
    # Generic single words that are too broad
    'analysis', 'science', 'stem', 'processes', 'process', 'development',
    'test', 'defense', 'navigation', 'displays', 'concept',
    'developing', 'developing software', 'creative', 'ideal',
    'ideal solutions', 'creative solutions', 'solutions',
    # Multi-word generic phrases
    'all different backgrounds', 'all aspects', 'other tasks',
    'other software engineers', 'software engineers',
    # Very generic software/product terms
    'software', 'software languages', 'software products', 'software applications',
    'operational needs', 'technical computing environments',
    'company', 'organization', 'business',
}

# Descriptor words that indicate meta-language, not skills
descriptor_words = {
    'concepts', 'technology', 'problems', 'approaches', 'practices',
    'experience', 'expertise', 'knowledge', 'understanding'
}

# Abstract phrases that are often too generic to count as concrete skills
abstract_skill_phrases = {
    'data pipelines',
    'data platforms',
    'real-time analytics',
    'cloud platforms',
    'database schemas',
    'machine learning workflows',
    'software design patterns',
    'solid principles',
    'event streaming platforms',
    'data science tools',
    'system architecture',
}

# Canonical aliases so UI output and matching are cleaner and less repetitive.
skill_aliases = {
    'github': 'git',
    'github actions': 'git',
    'google cloud platform': 'gcp',
    'google cloud': 'gcp',
    'postgres': 'postgresql',
    'js': 'javascript',

    # Explicit DB-term synonyms
    'postgre sql': 'postgresql',
    'postgresql db': 'postgresql',
    'postgres database': 'postgresql',
    'postgresql database': 'postgresql',
    'psql': 'postgresql',
    'my sql': 'mysql',
    'mysql db': 'mysql',
    'mysql database': 'mysql',
    'microsoft sql server': 'sql server',
    'ms sql': 'sql server',
    'mssql': 'sql server',
    'sqlserver': 'sql server',
    'sqlite3': 'sqlite',
    'mongo db': 'mongodb',
}

# Database terms used to prevent overly-permissive fuzzy matches
# (e.g., mysql incorrectly matching generic sql).
database_terms = {
    'postgresql', 'mysql', 'sql server', 'sqlite', 'mongodb', 'oracle',
    'dynamodb', 'snowflake', 'bigquery', 'redis', 'mariadb', 'cassandra',
    'cockroachdb', 'neo4j'
}

def is_database_term(skill):
    s = canonicalize_skill(skill)
    return s in database_terms

def canonicalize_skill(skill):
    """Normalize known aliases to a canonical skill token."""
    s = skill.strip().lower()
    # Normalize punctuation/spacing commonly seen in DB terms.
    s = s.replace('_', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    # Normalize variants with parenthetical noise.
    s = re.sub(r'\s*\(.*\)$', '', s).strip()
    return skill_aliases.get(s, s)

# Locations
locations = {
    'san francisco', 'new york', 'chicago', 'london', 'toronto', 'remote',
    'california', 'new york', 'chicago', 'seattle', 'boston'
}

# Industries
industries = {
    'healthcare', 'finance', 'retail', 'banking', 'technology',
    'manufacturing', 'education', 'transportation', 'real estate'
}

# Company descriptors
company_descriptors = {
    'fortune 500', 'startup', 'enterprises', 'hypergrowth', 'fast-growing'
}

# Time references
time_references = {
    'january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december',
    'week', 'quarter', 'year', 'month'
}

header_patterns = [
    # All caps headers
    r'^[A-Z\s]{5,40}$',  # "REQUIRED QUALIFICATIONS"
    
    # Title case with optional colon
    r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*:?$',  # "Required Qualifications:"
    
    # With dashes
    r'^-\s+[A-Z].*$',  # "- Required Qualifications"
    
    # Keywords that indicate headers
    r'.*(qualifications|requirements|responsibilities|skills|about|offer|interview).*',
    
    # Lines with mostly uppercase (job posting style)
    r'^[A-Z\s\-]{10,}$'
]

# Also catch patterns
generic_patterns = [
    r'^(the|a|an|your|our)\s',  # Starts with article
    r'\s(and|or)\s.*',           # Contains logical operators (filler)
    r'equal\s+opportunity',      # "equal opportunity employer"
    r'significant.*opportunit',  # "significant...opportunities"
    r'care(er|ing)\s+(growth|opportunity)',  # "career growth", "caring opportunity"
    r'\bcomputer\s+(science|science)',  # "computer science" (academic field)
    r'analytical\s+workload',    # "analytical workloads" (vague)
]

# Look for common seniority indicators
seniority_levels = {
    'intern', 'entry', 'junior', 'associate', 'mid-level', 'senior', 
    'lead', 'principal', 'manager', 'director', 'vp', 'c-level', 'new college grad', 'recent grad', 'I', 'II', 'III', 'IV', 'V', 'VI'
}

def preprocess_text(text):
    text = text.lower()

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # tokenize
    doc = nlp(text)

    tokens = []

    for token in doc:
        if not token.is_stop and not token.is_space:
            tokens.append(token.lemma_)

    return tokens

# Helper function to filter out compensation-related phrases from skill extraction
def is_compensation(skill):
    """Check if a skill phrase is compensation/benefits language."""
    skill_lower = skill.lower()
    
    # Keyword check
    if any(kw in skill_lower for kw in compensation_keywords):
        return True
    
    # Pattern check
    if any(re.search(pattern, skill_lower) for pattern in compensation_patterns):
        return True
    
    return False

def is_section_header(text):
    """Check if text matches header patterns across different job post formats."""
    skill_lower = text.lower()
    
    # Exact phrase matching (fastest)
    if skill_lower in header_patterns:
        return True
    
    # Pattern matching (catches variations)
    if any(re.search(pattern, text) for pattern in header_patterns):
        return True
    
    # Keyword matching (catches embedded headers)
    if any(kw in skill_lower for kw in ['qualifications', 'requirements']):
        return True
    
    return False

# Check if skill CONTAINS any individual generic word
def is_generic(skill):
    skill_lower = skill.lower()
    
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
    
    # Check against generic patterns
    if any(re.search(p, skill_lower) for p in generic_patterns):
        return True
    
    return False

def is_meta_requirement(skill):
    """Detect meta-requirement language like 'at least one X', 'X or more'."""
    patterns = [
        r'^at least',           # "at least one"
        r'one or more',         # "one or more X"
        r'\d+\s*or\s*more',     # "3 or more"
        r'^prefer',             # "prefer X"
        r'equivalent',          # "equivalent experience"
    ]
    return any(re.search(p, skill, re.IGNORECASE) for p in patterns)

def is_measurement(skill):
    """Detect metrics/measurements like '100m+ events', 'petabyte-scale'."""
    # Starts with number or measurement pattern
    return bool(re.match(r'^\d+', skill)) or bool(re.search(r'\d+[a-z]*\+?', skill))

def is_vague_descriptor(skill):
    """Detect vague descriptive phrases that aren't concrete skills."""
    vague_words = {
        'similar', 'comparable', 'appropriate', 'relevant', 'applicable',
        'general', 'basic', 'advanced', 'cutting', 'edge'
    }
    if any(word in skill for word in vague_words):
        return True
    
    # Patterns for size/scale metrics: "petabyte-scale", "enterprise-grade"
    if re.search(r'(\w+-scale|\w+-grade|\w+-level)', skill):
        return True
    
    # Negation patterns: "no legacy systems"
    if re.match(r'^no\s', skill, re.IGNORECASE):
        return True
    
    return False

def is_meta_language(skill):
    """Check if skill is meta/descriptor language rather than a concrete skill."""
    skill_lower = skill.lower()
    
    # NEW: Check for meta-requirement patterns
    if is_meta_requirement(skill):
        return True
    
    # NEW: Check for measurements/metrics
    if is_measurement(skill):
        return True
    
    # NEW: Check for vague descriptors
    if is_vague_descriptor(skill):
        return True

    # Filter abstract, non-concrete skill phrases
    if skill_lower in abstract_skill_phrases:
        return True
    if re.match(r'^(data|cloud|software|system|machine learning|event streaming|database)\s+(pipelines?|platforms?|analytics|workflows?|architecture|patterns?|principles|schemas|tools?)$', skill_lower):
        return True
    
    # Check if contains descriptive words
    if any(desc in skill_lower for desc in descriptor_words):
        return True
    
    # Too many words (> 5) = likely description, not skill
    if len(skill.split()) > 5:
        return True
    
    # Check locations
    if any(loc in skill_lower for loc in locations):
        return True
    
    # Check industries
    if any(ind in skill_lower for ind in industries):
        return True
    
    # Check company descriptors
    if any(comp in skill_lower for comp in company_descriptors):
        return True
    
    # Check time references
    if any(time in skill_lower for time in time_references):
        return True
    
    # Check for company suffixes: "inc", "llc", "ltd", "corp", "gmbh"
    if re.search(r'\b(inc|llc|ltd|corp|co\.?|gmbh)\b', skill_lower):
        return True
    
    # Single letter or very short abbreviations (not C++ or C#)
    if len(skill) <= 2 and not skill.replace('+', '').replace('#', '').isalpha():
        return True
    
    return False

def is_short_acronym(skill):
    """Filter out short acronyms likely to be artifacts from parenthetical text.
    
    Examples of artifacts: 'hal' (from HAL), 'sil' (from SIL)
    
    Real tech terms to keep: 'git', 'sql', 'go', 'c++', 'c#', 'api'
    
    Heuristic: Filter terms that are all-lowercase 2-3 chars BUT only if they look
    like acronym artifacts (consecutive consonants with all lowercase). 
    Real tech: git, sql, go, etc. - are actual command names, not acronyms.
    Artifacts: hil, sil, ete - look like extracted acronyms.
    """
    s = skill.lower().strip()
    
    # Don't filter if it has special chars (c++, c#) - those are real
    if '+' in s or '#' in s:
        return False
    
    # Don't filter 2-3 char terms that are known programming keywords/tools
    # These appear in real skill contexts, not from acronyms
    if s in {'git', 'go', 'sql', 'api', 'rpc', 'ftp', 'ssh', 'aws', 'gcp', 'ai', 'ml', 'iot'}:
        return False
    
    # Filter suspicious 2-3 char patterns that look like acronym artifacts
    # Pattern: mostly consonants with no vowels = likely acronym (e.g., hil, sil, ete, dsp)
    if 2 <= len(s) <= 3:
        vowels = sum(1 for c in s if c in 'aeiou')
        # If 0 or 1 vowel in 2-3 char term, likely acronym artifact
        if vowels <= 1:
            return True
    
    return False


def strip_company_culture_sections(text):
    """Remove 'About Us' and company culture/mission sections before skill extraction.
    
    These sections often contain mission language, company values, and aspirational
    language that gets incorrectly extracted as skills. This is more generalizable
    than hardcoding company-specific terms.
    
    Returns text with these sections removed.
    """
    result = text
    
    # Pattern 1: Remove major "About Us" / "Company Culture" / "Mission" sections
    # These are typically intro sections before job details
    patterns = [
        r'(?:about\s+(?:us|the\s+company)|our\s+(?:mission|values|culture|beliefs)|company\s+(?:culture|mission|values|overview)|who\s+we\s+are)\s*\n+.*?(?=\n\s*(?:requirements|qualifications|responsibilities|skills|location|apply|interview|contact|what\s+we\s+offer|job\s+description|about\s+the\s+role|the\s+role|responsibilities|key\s+responsibilities))',
    ]
    
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.DOTALL)
    
    # Pattern 2: Remove sentences with obvious mission/aspirational language patterns
    # These often contain multi-word mission language that slips through
    mission_sentence_patterns = [
        #[^.!?\n]* -> capture entire sentence without punctuation (including new lines) 
        r'[^.!?\n]*(?:inspire|empower|transform|revolutionize|innovate|deter\s+aggression|tomorrow\'?s\s+threat|today\'?s\s+mission)[^.!?\n]*[.!?\n]',
        r'[^.!?\n]*(?:world[- ]class|tremendous|cutting[- ]edge)\s+(?:engineers|team|company|people)[^.!?\n]*[.!?\n]',
        r'[^.!?\n]*(?:the\s+future|tomorrow|together|teamwork)[^.!?\n]*make[^.!?\n]*[.!?\n]',
    ]
    
    for pattern in mission_sentence_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.DOTALL)
    
    return result


def is_mission_language(phrase):
    """Detect mission/values/culture language by structural patterns, not individual words.
    
    This is generalizable across companies and industries without hardcoding
    company-specific missions, values, or aspirational language.
    
    Uses linguistic patterns that are universal to mission statements.
    """
    phrase_lower = phrase.lower().strip()
    
    # SIMPLE STRING CHECKS (most robust, handles any Unicode issues)
    # These catch obvious mission language that shouldn't be skills
    simple_mission_checks = [
        'mission',  # "today's mission", "our mission"
        'threat',    # "tomorrow's threat", "security threat"
        'aggression', # "deter aggression"
    ]
    
    # If phrase is just one of these mission keywords + maybe 1-2 modifiers
    word_count = len(phrase_lower.split())
    if 1 <= word_count <= 3:
        # Single or 2-word phrases with mission keywords are probably mission language
        if any(keyword in phrase_lower for keyword in simple_mission_checks):
            # Exception: Don't filter if it has concrete tech terms
            tech_keywords = {'software', 'data', 'cloud', 'database', 'api', 'web',
                           'python', 'java', 'c++', 'kubernetes', 'docker', 'git',
                           'hardware', 'network', 'system', 'application', 'platform', 'embedded',
                           'security', 'protocol', 'interface', 'architecture'}
            if not any(tech in phrase_lower for tech in tech_keywords):
                return True
    
    # REGEX CHECKS (fallback for more complex cases)
    direct_mission_patterns = [
        # Defense-specific mission language
        r"deter.*aggression",
        r"defend.*(?:against|threat)",
        r"secure.*(?:the|our|their)",
        # Obvious aspirational language as primary verb
        r"^(?:inspire|empower|revolutionize|innovate|lead|drive|shape|protect|deter)\s",
        r"^(?:tremendous|world.?class|cutting.?edge|best.?in.?class)\s",
        # Company names
        r'^(?:raytheon|honeywell|boeing|lockheed|general\s+dynamics|northrop)(?:\s|$)',
    ]
    
    for pattern in direct_mission_patterns:
        if re.search(pattern, phrase_lower, re.IGNORECASE):
            return True
    
    # SCORING SYSTEM for subtle cases
    aspirational_verbs = {
        'inspire', 'empower', 'transform', 'revolutionize', 'innovate', 
        'drive', 'accelerate', 'pioneer', 'shape'
    }
    
    abstract_descriptors = {
        'world-class', 'world class', 'cutting-edge', 'best-in-class', 'leading', 'premier',
        'exceptional', 'tremendous', 'great'
    }
    
    mission_patterns = [
        r'(?:member\s+of|part\s+of|committed\s+to|dedicated\s+to)',
        r'(?:together|teamwork|collaboration)\s+(?:create|build|drive)',
    ]
    
    score = 0
    
    verb_count = sum(1 for verb in aspirational_verbs 
                     if re.search(r'\b' + verb + r'\b', phrase_lower))
    if verb_count >= 1:
        score += 2
    
    descriptor_count = sum(1 for desc in abstract_descriptors 
                          if desc in phrase_lower)
    if descriptor_count >= 1:
        score += 2
    
    if any(re.search(p, phrase_lower) for p in mission_patterns):
        score += 2
    
    # Multi-word phrases with no concrete tech terms = likely description/mission
    words = phrase.split()
    if 2 <= len(words) <= 5:
        tech_keywords = {'software', 'data', 'cloud', 'database', 'api', 'web',
                        'python', 'java', 'c++', 'kubernetes', 'docker', 'git',
                        'hardware', 'network', 'system', 'application', 'platform', 'embedded'}
        has_tech = any(word.lower() in tech_keywords for word in words)
        if not has_tech and score >= 1:
            return True
    
    return score >= 2


def has_structural_problems(skill):
    """Check if skill has formatting issues (newlines, dashes, etc.)."""
    # Contains newline
    if '\n' in skill:
        return True
    
    # Starts or ends with dash
    if skill.startswith('-') or skill.endswith('-'):
        return True
    
    # Unmatched closing parenthesis
    if ')' in skill and '(' not in skill:
        return True
    
    # Multiple spaces or tabs
    if '  ' in skill or '\t' in skill:
        return True
    
    return False

def normalize_skill(skill):
    """Normalize skill text (lowercase, remove extra spaces)."""
    normalized = re.sub(r'\s+\d+(\.\d+)*', '', skill.strip().lower())
    normalized = canonicalize_skill(normalized)
    if len(normalized) > 100:
        return skill  # Don't normalize if it becomes too long (likely a description)
    if len(normalized) < 2:
        return skill  # Don't normalize if it becomes too short (likely not a skill)
    
    return normalized.capitalize()



# --- Matching helpers ---
def normalize_skills(skills):
    """Normalize an iterable of skill strings into a list of normalized strings."""
    normalized = []
    seen = set()
    for s in skills:
        if not s:
            continue
        n = normalize_skill(s)
        # Keep lowercase form for matching
        n = n.lower()
        if n in seen:
            continue
        seen.add(n)
        normalized.append(n)
    return normalized


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


def calculate_skill_match_score_advanced(job_skills, resume_source, fuzzy_threshold=0.75, token_threshold=0.5):
    """Calculate matching between a list of job skills and resume content.

    - `job_skills` should be an iterable of skill strings (from `extract_skills`).
    - `resume_source` may be either raw text (str) or an iterable of resume skill strings.

    Returns a dict with matched pairs, counts, coverage, and average similarity.
    """
    # Prepare resume skills: if raw text provided, extract skills from it
    if isinstance(resume_source, str):
        resume_skills = extract_skills(resume_source)
    else:
        resume_skills = list(resume_source)

    job_norm = normalize_skills(job_skills)
    resume_norm = normalize_skills(resume_skills)

    matched = []
    unmatched = []

    for j in job_norm:
        best = None
        best_score = 0.0
        best_method = None
        #Check for token matching betwee job skill and each resume skill, then fuzzy matching as a fallback if token overlap is low. This allows us to catch both exact matches and close variations.
        for r in resume_norm:
            # Guardrail: DB vendor/engine names should not fuzzy-match generic "sql"
            # or different DB engines. Require exact canonical match for DB terms.
            if is_database_term(j):
                if canonicalize_skill(j) != canonicalize_skill(r):
                    continue

            # token overlap first (fast, interpretable)
            to_score = token_overlap_score(j, r)
            
            # NEW: For multi-word job skills, require higher token overlap
            # Multi-word skills need more than just partial token overlap
            effective_token_threshold = token_threshold
            job_tokens = len(j.split())
            if job_tokens >= 2:
                # For 2+ word skills, require at least 0.7 token overlap (not 0.5)
                effective_token_threshold = max(0.7, token_threshold)
            
            if to_score > best_score and to_score >= effective_token_threshold: 
                best = r 
                best_score = to_score 
                best_method = 'token'

            # fuzzy fallback - but reject substring-based fuzzy matches (false positives)
            # e.g., 'git' should NOT match 'gitlab' just because 'git' is in 'gitlab'
            fr = fuzzy_ratio(j, r)
            
            # Guard against substring false positives
            # If one is a substring of the other, require MUCH higher fuzzy score
            j_lower = j.lower()
            r_lower = r.lower()
            is_substring_match = (j_lower in r_lower) or (r_lower in j_lower)
            
            if is_substring_match and fr < 0.95:
                # Reject fuzzy matches based on simple substrings unless near-identical
                # This prevents 'git' from matching 'gitlab'
                pass
            elif fr > best_score:
                best = r
                best_score = fr
                best_method = 'fuzzy'

        # Decide match by thresholds
        # allow a slightly lower fuzzy threshold for very short tokens (e.g., 'git' vs 'github')
        if best_method == 'token':
            effective_token_threshold = token_threshold
            job_tokens = len(j.split())
            if job_tokens >= 2:
                effective_token_threshold = max(0.7, token_threshold)
            is_match = best_score >= effective_token_threshold
        elif best_method == 'fuzzy':
            eff_fuzzy_threshold = fuzzy_threshold
            
            # NEW: For multi-word job skills, require much stricter fuzzy matching
            # A multi-word skill should not match a single-word resume skill via fuzzy alone
            job_tokens = len(j.split())
            if best is not None:
                resume_tokens = len(best.split())
                # If job has multiple words but resume has only 1, require very high similarity
                if job_tokens >= 2 and resume_tokens == 1:
                    eff_fuzzy_threshold = 0.95  # Basically require near-exact match
            
            # if either side is very short, relax fuzzy threshold
            if best is not None and (len(j) <= 3 or len(best) <= 3):
                eff_fuzzy_threshold = min(0.65, fuzzy_threshold)
            is_match = best_score >= eff_fuzzy_threshold
        else:
            is_match = False

        if is_match:
            matched.append({'job_skill': j, 'resume_skill': best, 'score': round(best_score, 3), 'method': best_method})
        else:
            unmatched.append({'job_skill': j, 'best_candidate': best, 'score': round(best_score, 3) if best is not None else 0.0, 'method': best_method})

    total = len(job_norm)
    matched_count = len(matched)
    # Average score of matched skills (for those that had a match above threshold)
    avg_score = round(sum(m['score'] for m in matched) / matched_count, 3) if matched_count else 0.0
    # Coverage percentage of matched skills
    coverage = round((matched_count / total) * 100, 1) if total else 0.0

    return {
        'total_job_skills': total,
        'matched_count': matched_count,
        'coverage_percent': coverage,
        'average_match_score': avg_score,
        'matched': matched,
        'unmatched': unmatched,
        'resume_skills_sample': resume_norm[:30],
    }

def extract_soft_skills(text):
    """
    Extract soft skills from text using pattern matching.
    Soft skills are interpersonal and professional competencies.
    
    Returns:
        set: Set of soft skills found in the text
    """
    soft_skills = set()
    
    # Common soft skills and their variations
    soft_skill_patterns = {
        # \b = word boundary to avoid partial matches (e.g., "communicative" shouldn't match "communication" skill)
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
    
    text_lower = text.lower()
    
    for skill_name, pattern in soft_skill_patterns.items():
        if re.search(pattern, text_lower):
            soft_skills.add(skill_name)
    
    return soft_skills

#Use spaCy's Named Entity Recognition to extract skills, experience, and other relevant information from the text.
def extract_skills(text):
    """
    Extract skills from job posting by finding noun phrases.
    Works for any domain without hardcoded skill lists.
    """
    # STEP 1: Strip company culture/mission sections to avoid extracting mission language as skills
    text = strip_company_culture_sections(text)
    
    doc = nlp(text)
    skills = set()

    # Capture explicit skill lists from common resume/job section labels.
    explicit_lines = re.findall(
        # ^\s*[-•]*\s* - start of line, optional whitespace, optional bullet
        r'^\s*[-•]*\s*(languages?|programming languages?|tools(?:\s+and\s+frameworks)?|frameworks?|databases?|cloud platforms?|devops|version control|skills(?:\s*in)?|professional skills(?:\s+and\s+interests)?)\s*:\s*(.+)$',
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    for _, value in explicit_lines:
        for p in re.split(r',|;|/|\bor\b', value):
            p = p.strip().strip('()').lower()
            if not p:
                continue
            # r'^[a-zA-Z0-9\+\#\.\s\-]{1,60}$' -> allow common skill characters, limit length to avoid descriptions
            if re.match(r'^[a-zA-Z0-9\+\#\.\s\-]{1,60}$', p) and not is_generic(p) and not is_meta_language(p):
                skills.add(canonicalize_skill(p))

    # Also handle bullet style: "Skills in Python, Java, C++"
    # ^\s*[-•]*\s* -> optional bullet at start of line, then look for "skills in X"
    skills_in_lines = re.findall(r'^\s*[-•]*\s*skills\s+in\s+(.+)$', text, flags=re.IGNORECASE | re.MULTILINE)
    for line in skills_in_lines:
        for p in re.split(r',|;|/|\bor\b', line):
            p = p.strip().strip('()').lower()
            if not p:
                continue
            if re.match(r'^[a-zA-Z0-9\+\#\.\s\-]{1,60}$', p) and not is_generic(p) and not is_meta_language(p):
                skills.add(canonicalize_skill(p))
    
    # Extract noun chunks (typically 1-4 word phrases that are good skill candidates)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip().lower()
        
        # NEW: Skip items starting/ending with punctuation (artifacts)
        if chunk_text.startswith('(') or chunk_text.startswith('[') or chunk_text.endswith(')') or chunk_text.endswith(']'):
            continue
        
        # NEW: Skip phrases starting with filler words (all, other, various, some)
        if re.match(r'^(all|other|various|some|any|many|most|one)\s+', chunk_text):
            continue
        
        # NEW: Stricter single-word filtering - reject generic single words
        words = chunk_text.split()
        if len(words) == 1:
            # Single words: only keep if they're known technical terms or acronyms
            # Don't use fuzzy matching - be strict
            if chunk_text in {'git', 'sql', 'go', 'java', 'python', 'ruby', 'php', 'rust', 'kotlin', 'scala',
                             'c', 'c++', 'c#', 'r', 'swift', 'perl', 'lua', 'groovy', 'dart',
                             'linux', 'windows', 'macos', 'android', 'ios',
                             'aws', 'gcp', 'azure', 'docker', 'kubernetes', 'kafka',
                             'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                             'react', 'vue', 'angular', 'node', 'express', 'django',
                             'tensorflow', 'pytorch', 'scikit', 'pandas', 'numpy',
                             'jira', 'git', 'gitlab', 'github', 'jenkins', 'devops'}:
                # Known tech term - keep it
                pass
            else:
                # Unknown single word - skip to reduce noise
                continue
        
        # Filter: skip compensation-related terms
        if is_compensation(chunk_text):
            continue
        
        # Filter: skip chunks with structural problems
        if has_structural_problems(chunk_text):
            continue

        # Filter: skip section headers
        if is_section_header(chunk_text):
            continue
        
        # NEW: Skip short acronyms likely extracted from parentheticals (unless known tech)
        if is_short_acronym(chunk_text):
            continue
        
        # Filter: skip generic terms that don't indicate specific skills
        if is_generic(chunk_text):
            continue
        
        # Filter: skip meta language (descriptors, locations, industries, etc.)
        if is_meta_language(chunk_text):
            continue
        
        # NEW: Skip mission/culture/values language using pattern-based detection
        if is_mission_language(chunk_text):
            continue

        # NEW: Skip chunks that are lists of HR categories (diversity language)
        # Common pattern: words separated by commas in HR sections
        if ',' in chunk_text and any(hr_term in chunk_text.lower() for hr_term in ['disability', 'religion', 'gender', 'national origin', 'sexual orientation', 'veteran']):
            continue

        # Filter: skip very short or very long chunks
        if 2 <= len(chunk_text) <= 100:
            # Skip if all words are stop words
            if not all(token.is_stop for token in chunk):
                skills.add(canonicalize_skill(chunk_text))
    
    # Also extract entities marked as ORG or PRODUCT (sometimes skills)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            ent_text = ent.text.strip().lower()
            
            # Skip if has structural problems
            if has_structural_problems(ent_text):
                continue
            
            # NEW: Skip items starting/ending with punctuation
            if ent_text.startswith('(') or ent_text.startswith('[') or ent_text.endswith(')') or ent_text.endswith(']'):
                continue
            
            # NEW: Skip entities with generic words patterns
            if re.match(r'^(all|other|various|some|any)\s+', ent_text):
                continue
            
            # NEW: Skip short acronyms
            if is_short_acronym(ent_text):
                continue
            
            # NEW: Skip if contains "software engineers" or similar HR language
            if any(hr_phrase in ent_text for hr_phrase in ['software engineers', 'team member', 'engineer', 'developer']):
                continue
            
            # NEW: Skip mission/culture/values language using pattern-based detection
            if is_mission_language(ent_text):
                continue
            
            # Skip if compensation, generic, or meta language
            if is_compensation(ent_text) or is_generic(ent_text) or is_meta_language(ent_text):
                continue
            
            if 2 <= len(ent_text) <= 100:
                skills.add(canonicalize_skill(ent_text))
    
    # Additionally, extract language/tool lists that appear inline, e.g.,
    # "programming language (Python, Java, Go, or C++)" or
    # "Programming Languages: Python, Go, Java"
    # This catches cases like C++ that noun-chunks may miss.
    prog_lang_patterns = []
    # Look for "programming language(s) (X, Y, Z)" or "programming language(s): X, Y, Z"
    prog_lang_patterns.extend(re.findall(r'programming language[s]?[^\n]*\(([^)]+)\)', text, flags=re.IGNORECASE))
    prog_lang_patterns.extend(re.findall(r'programming language[s]?:\s*([^\n]+)', text, flags=re.IGNORECASE))
    for group in prog_lang_patterns:
        # clean parentheses content and split on commas/or/slash
        parts = re.split(r',|/|\bor\b', group)
        for p in parts:
            p = p.strip().strip('()').lower()
            # ignore generic markers like 'required' or short noise
            if not p or p in ('required', 'required)', '(required'):
                continue
            # only accept plausible language/tool tokens (letters, +, #, numbers, spaces)
            if re.match(r'^[a-zA-Z0-9\+\#\s\-]{1,40}$', p):
                # remove trailing qualifiers like '(required)'
                p = re.sub(r'\(.*\)$', '', p).strip()
                if 1 <= len(p) <= 100 and not is_generic(p) and not is_meta_language(p):
                    skills.add(canonicalize_skill(p))

    # Final pass: drop known abstract phrases from output.
    skills = {s for s in skills if not is_meta_language(s)}

    # Prune single-word fragments when a clearer multi-word term already exists
    # e.g., keep "beyond compare" and drop "compare".
    pruned = set(skills)
    for s in list(skills):
        if len(s.split()) == 1:
            if any(
                other != s and re.search(r'\b' + re.escape(s) + r'\b', other)
                for other in skills
            ):
                pruned.discard(s)
    skills = pruned
    # Post-process: strip qualifiers like "preferred", "required"
    skills_cleaned = strip_skill_qualifiers(skills)
    # Post-process: split comma-separated skills
    skills_expanded = split_comma_separated_skills(skills_cleaned)
    
    # Add soft skills extracted from the text
    soft_skills = extract_soft_skills(text)
    skills_expanded.update(soft_skills)
    
    return sorted(list(skills_expanded))

def extract_skills_with_llm(text, context='resume'):
    """
    Extract skills from text using LLM (GPT-4o) semantic understanding.
    Falls back to regex-based extraction if LLM call fails.
    
    Args:
        text: Resume or job posting text
        context: 'resume' or 'job_posting' to tailor the extraction prompt
    
    Returns:
        list: Sorted list of technical skills
    """
    if not text or not text.strip():
        return []
    
    try:
        # Initialize GPT-4o model
        model = ChatOpenAI(model='gpt-4o', timeout=60, max_retries=2)  # Use deterministic output for extraction
        
        # Craft context-specific prompt
        if context == 'resume':
            system_prompt = """You are a technical skills extraction expert. Extract ONLY technical skills from the resume.
            
Technical skills include:
- Programming languages (Python, Java, C++, JavaScript, etc.)
- Frameworks & libraries (React, Django, FastAPI, etc.)
- Databases & data tools (PostgreSQL, MongoDB, Spark, etc.)
- Cloud & DevOps (AWS, Docker, Kubernetes, Terraform, etc.)
- Data science & ML (TensorFlow, PyTorch, Scikit-learn, etc.)
- Tools & platforms (Git, JIRA, GitHub, GitLab, etc.)
- Other technical domains (API design, microservices, distributed systems, etc.)

DO NOT include:
- Soft skills (leadership, communication, teamwork, etc.)
- Personal interests or hobbies (gaming, reading, etc.)
- Honors, scholarships, or certifications unless they're technical frameworks
- Generic terms (management, planning, problem-solving, etc.)
- Company names or job titles

Return a JSON object with field 'skills' containing a list of extracted technical skills.
Example: {"skills": ["Python", "React", "Docker", "AWS"]}"""
        else:  # job_posting
            system_prompt = """You are a technical skills extraction expert. Extract required and preferred technical skills from a job posting.
            
Technical skills include:
- Programming languages
- Frameworks & libraries
- Databases & data tools
- Cloud & DevOps
- Data science & ML
- Tools & platforms
- Other technical domains

Return a JSON object with fields 'required_skills' and 'preferred_skills' (both lists).
Example: {"required_skills": ["Python", "AWS"], "preferred_skills": ["Kubernetes", "Docker"]}"""
        
        # Call LLM
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract technical skills from this text:\n\n{text[:4000]}")  # Limit to 4000 chars to stay within token budget
        ]
        
        response = model.invoke(messages)
        response_text = response.content
        
        # Parse JSON response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            
            if context == 'resume':
                skills = parsed.get('skills', [])
            else:
                # For job postings, combine both required and preferred
                skills = list(set(parsed.get('required_skills', []) + parsed.get('preferred_skills', [])))
            
            # Clean up and normalize skills
            cleaned_skills = []
            for skill in skills:
                skill = skill.strip().lower()
                if 2 <= len(skill) <= 100 and skill:
                    cleaned_skills.append(canonicalize_skill(skill))
            
            return sorted(list(set(cleaned_skills)))
    except Exception:
        # Silently fall back to regex extraction
        logger.warning("LLM skill extraction failed; falling back to regex extraction")
    
    # Fallback to regex-based extraction
    return extract_skills(text)

def strip_skill_qualifiers(skills_set):
    """
    Remove qualifiers that follow skills like 'preferred', 'required', 'optional'.
    E.g., "C++ preferred" -> "C++", "Python required" -> "Python"
    
    Generalizable solution that works across any programming language or skill.
    """
    cleaned_skills = set()
    
    # Qualifiers that commonly follow skills in job postings (not structural connectors)
    qualifier_pattern = r'\s+(preferred|required|optional|recommended|desired|essential|mandatory)\b'
    
    for skill in skills_set:
        # Remove trailing qualifiers
        cleaned = re.sub(qualifier_pattern, '', skill, flags=re.IGNORECASE).strip()
        
        # Only add if something was extracted
        if cleaned and 2 <= len(cleaned) <= 100:
            cleaned_skills.add(cleaned)
    
    return cleaned_skills


def split_comma_separated_skills(skills_set):
    """
    Split comma-separated skills into individual items.
    E.g., "postgresql, mysql" becomes separate skills.
    """
    expanded_skills = set()
    
    for skill in skills_set:
        if ',' in skill:
            # Split and clean each part
            parts = [part.strip() for part in skill.split(',')]
            for part in parts:
                # Validate each part meets size requirements
                if 2 <= len(part) <= 100 and part:
                    expanded_skills.add(part)
        else:
            expanded_skills.add(skill)
    
    return expanded_skills


def extract_experience(text):
    doc = nlp(text)
    experience = []
    # Look for patterns like "3+ years experience", "at least 5 years", "one year of experience", etc.
    regex = re.compile(r'(\d+)\+?\s*(?:years?|months?)\s+(?:of\s+)?([a-z\s]+?(?:experience|expertise))', re.IGNORECASE)
    for sent in doc.sents:
        matches = regex.findall(sent.text)
        for match in matches:
            experience.append(' '.join(match).strip())

    return experience

def extract_education(text):
    # Removed ambiguous bare 2-letter forms (BS, BA, MS, MA, AS, AA) — they match
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
    matches = regex.findall(text)
    return list(set(matches))  # Remove duplicates


def extract_education_with_llm(text):
    """Extract degree levels from job posting using LLM.
    
    Uses GPT-4o to semantically understand degree requirements regardless of format.
    Falls back to regex-based extraction if LLM fails or returns empty results.
    
    Args:
        text: Job posting text
    
    Returns:
        list: List of degree types (e.g., ['Bachelor\'s', 'Master\'s', 'Degree'])
    """
    if not text or not text.strip():
        return []
    
    try:
        # Initialize GPT-4o model
        model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
        
        system_prompt = """You are an education requirements extraction expert. Extract ONLY the required and preferred degree types from a job posting.

Degree types include:
- Bachelor's degree (B.S., B.A., Bachelor of Science, etc.)
- Master's degree (M.S., M.A., M.B.A., Master of Science, etc.)
- PhD (Ph.D., Doctorate, etc.)
- Associate's degree (A.S., A.A., Associate's, etc.)
- Diploma, GED, High School
- Generic "Degree" when no specific type is mentioned

DO NOT include:
- Education fields/majors (Computer Science, Engineering, etc.)
- Years of experience requirements
- Other qualifications or certifications
- Company names

Return a JSON object with field 'degrees' containing a list of extracted degree types.
Example: {"degrees": ["Bachelor's", "Master's"]}"""
        
        # Call LLM
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract degree requirements from this job posting:\n\n{text[:4000]}")
        ]
        
        response = model.invoke(messages)
        response_text = response.content
        
        # Parse JSON response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            degrees = parsed.get('degrees', [])
            
            # Clean up and validate degrees
            cleaned_degrees = []
            for degree in degrees:
                degree_clean = degree.strip().strip('.,;/')
                if degree_clean and 2 <= len(degree_clean) <= 50:
                    cleaned_degrees.append(degree_clean)
            
            # If we got good results from LLM, combine with regex for completeness
            if cleaned_degrees:
                # Also get regex results to combine for maximum coverage
                regex_degrees = extract_education(text)
                # Combine, removing duplicates while preserving order
                combined = cleaned_degrees.copy()
                for degree in regex_degrees:
                    if degree not in combined:
                        combined.append(degree)
                return combined
    except Exception as e:
        # Silently fall back to regex extraction
        pass
    
    # Fallback to regex-based extraction if LLM fails or returns empty
    return extract_education(text)


def extract_education_field(text):
    """Extract education field/major from text (e.g., 'Computer Science', 'Electrical Engineering').
    
    Looks for patterns like "Bachelor's in Computer Science" OR fields after labels like "Major:", "Field:"
    Handles various resume formats to be more robust, including bare field names.
    """
    fields = []
    
    # Filter out common job posting phrases that aren't education fields
    excluded_phrases = {
        'support requirements analysis', 'who you are', 'desired skills',
        'required qualifications', 'preferred qualifications', 'responsibilities',
        'nice to have', 'about the role', 'what we offer', 'requirements',
        'qualifications', 'benefits', 'interview', 'about you', 'your role',
        'we are looking', 'we seek', 'we need', 'key responsibilities',
        'familiarity with', 'experience with', 'knowledge of', 'proficiency in',
        'understanding of', 'working knowledge', 'hands-on experience'
    }
    
    # University/college names and institution suffixes to exclude
    institution_keywords = {
        'university', 'college', 'institute', 'academy', 'school',
        'campus', 'polytechnic', 'tech'
    }
    
    # Degree names that should NOT be captured as fields
    degree_names = {
        'bachelor of science', 'master of science', 'bachelor of arts', 'master of arts', 'mba', 'bachelor of engineering', 'master of engineering', 'bachelor of technology', 'master of technology',
        'doctorate', 'diploma', 'certificate'
    }
    
    def is_excluded_phrase(field):
        """Check if field matches excluded job posting phrases, is a university name, or is a degree name."""
        field_lower = field.lower()
        # Check job posting phrases
        for phrase in excluded_phrases:
            if phrase in field_lower:
                return True
        # Check if it's a university/college name (contains institution keywords)
        for keyword in institution_keywords:
            if keyword in field_lower:
                return True
        # Check if it's a degree name itself
        if field_lower in degree_names:
            return True
        return False
    
    # Pattern 1: Match degree names followed by field (primary pattern)
    # Handles: "Bachelor's in Physics", "B.S. in Engineering", etc.
    # Use lookahead to stop at newline/punctuation
    pattern1 = r'\b(?:B\.?S\.?|B\.?A\.?|Bachelor\'?s?|M\.?S\.?|M\.?A\.?|M\.?B\.?A\.?|Master\'?s?|Ph\.?D\.?|Doctorate|A\.?S\.?|A\.?A\.?|Associate\'?s?|High School|GED|Diploma)\.?[ \t]+(?:degree[ \t]+)?(?:of[ \t]+Science[ \t]+in|in|of|with|related[ \t]+to)[ \t]+([A-Z][a-z]*(?:[ \t]+[A-Z][a-z]*){0,3})(?=\s*[\n,\.;:\(\)—]|\s*$)'
    
    matches1 = re.findall(pattern1, text, re.IGNORECASE)
    for match in matches1:
        field = match.strip().rstrip('.,;/')
        # Only keep if it's a reasonable field name (2-100 chars, not pure numbers/stopwords)
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'related', 'field') and not is_excluded_phrase(field):
            fields.append(field)

    # Pattern 1b: Degree abbreviation followed DIRECTLY by field — no 'in' connector
    # Handles: "B.S. Computer Science, Math Minor" → "Computer Science"
    #           "B.A. English Literature (GPA: 3.8)"  → "English Literature"
    # Excluded status words (first word) to avoid: "M.S. Student," "Ph.D. Candidate,"
    # Anchored to line-start to prevent matching "MS Azure" in mid-sentence job posting text.
    _degree_status_words = {'student', 'candidate', 'applicant', 'expected', 'thesis', 'dissertation'}
    # (?:^|\n)\s* - start of line or after newline, optional whitespace
    pattern1b = r'(?:^|\n)\s*(?:B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|Ph\.?D\.?)\s+(?!(?:in|of|degree)\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})(?=\s*[,\(\n]|\s*$)'
    for match in re.findall(pattern1b, text, re.IGNORECASE):
        field = match.strip().rstrip('.,;/')
        first_word = field.split()[0].lower()
        if first_word in _degree_status_words:
            continue
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'related', 'field') and not is_excluded_phrase(field):
            fields.append(field)
    
    # Pattern 2: Match fields after labels like "Major:", "Field:", "Concentration:", etc.
    # Stop at newline, punctuation, or other structural breaks
    # NOTE: Added \b word boundaries to prevent matching "Field" inside "fields" (job posting garbage)
    pattern2 = r'\b(?:Major|Field(?:\s+of\s+study)?|Specialization|Concentration|Discipline|Subject)\b\s*:?\s*([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,3})(?=[\n,\.;:\(\)—]|$)'
    matches2 = re.findall(pattern2, text, re.IGNORECASE)
    for match in matches2:
        field = match.strip().rstrip('.,;/')
        # Only keep if it's a reasonable field name (at least 2 chars, exclude common stopwords)
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'unknown', 'other', 'general studies') and not is_excluded_phrase(field):
            fields.append(field)
    
    # Pattern 3: Match fields after "Education" with degree, stopping at newline
    # This helps catch cases where the field is listed in a formal format
    pattern3 = r'(?:Education|Degree)\s*[:\-]?\s*(?:B\.?S\.?|B\.?A\.?|Bachelor\'?s?|M\.?S\.?|M\.?A\.?|M\.?B\.?A\.?|Master\'?s?|Ph\.?D\.?|Doctorate)\.?[ \t]+in[ \t]+([A-Z][a-z]*(?:[ \t]+[A-Z][a-z]*){0,2})(?=\s*[\n,\.;:\(\)]|\s*$)'
    matches3 = re.findall(pattern3, text, re.IGNORECASE)
    for match in matches3:
        field = match.strip().rstrip('.,;/')
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree') and not is_excluded_phrase(field):
            fields.append(field)
    
    # Pattern 4: Match single-line field names that appear right after a degree keyword and newline
    # Handles: "Education\nMicrobiology", "Bachelor's\nMicrobiology", "Bachelor of Science\nMicrobiology"
    # Must stop at next newline or punctuation
    pattern4 = r'\b(?:Education|Bachelor\'?s?|Bachelor\s+of\s+Science|Master\'?s?|Master\s+of\s+Science|Degree|B\.?S\.?|B\.?A\.?|M\.?S\.?|Ph\.?D\.?)\b\s*[:\-]?\s*\n\s*([A-Z][a-z]*(?:[ \t]+[A-Z][a-z]*){0,2})(?=\n|,|$)'
    matches4 = re.findall(pattern4, text, re.IGNORECASE)
    for match in matches4:
        field = match.strip().rstrip('.,;/')
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'university', 'college', 'gpa', 'cum', 'laude', 'honors') and not is_excluded_phrase(field):
            fields.append(field)
    
    # Pattern 5: Match capitalized field names on a new line after degree info, with bullet or dash
    # Handles: "* B.S. in\n* Microbiology", "* Bachelor of Science\n* Microbiology"
    # Only capture up to 2 words and stop at newline
    # Anchored to line-start to prevent A\.?S\.? matching "as" inside words like "areas".
    pattern5 = r'(?:^|\n)\s*(?:B\.?S\.?|B\.?A\.?|Bachelor\'?s?|Bachelor\s+of\s+Science|M\.?S\.?|M\.?A\.?|M\.?B\.?A\.?|Master\'?s?|Master\s+of\s+Science|Ph\.?D\.?|A\.?S\.?|A\.?A\.?|Associate\'?s?|Diploma|GED)(?:[ \t]+in)?\s*\n\s*[\-\*]?\s*([A-Z][a-z]*(?:[ \t]+[A-Z][a-z]*){0,2})(?=\n|,|$)'
    matches5 = re.findall(pattern5, text, re.IGNORECASE)
    for match in matches5:
        field = match.strip().rstrip('.,;/')
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'gpa', 'cum', 'laude', 'honors', 'university', 'college') and not is_excluded_phrase(field):
            fields.append(field)
    
    # Pattern 6: Match fields with inline dates/multiple spaces/parentheticals after field name
    # Handles: "Bachelor of Science in Microbiology                                May 2026"
    #           "Bachelor of Science in Computer Science (GPA: 3.5/4.0) Current – May 2026"
    # Stop at optional whitespace then a paren/newline/comma, OR multiple spaces, OR end of string
    pattern6 = r'(?:Bachelor\s+of\s+Science|B\.?S\.?|Master\s+of\s+Science|M\.?S\.?)\s+in\s+([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,2})(?=\s*[\(\n,\.;]|\s{2,}|\s*$)'
    matches6 = re.findall(pattern6, text, re.IGNORECASE)
    for match in matches6:
        field = match.strip().rstrip('.,;/')
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'gpa', 'cum', 'laude', 'honors', 'university', 'college') and not is_excluded_phrase(field):
            fields.append(field)
    
    # Pattern 7: Match fields in degree requirements with parenthetical content
    # Handles: "Master's or PhD degree (or equivalent experience) in Computer Science"
    # This catches job postings that have alternate/equivalent experience noted in parentheses
    pattern7 = r'(?:Master\'?s?|PhD|Bachelor\'?s?|Ph\.?D\.?)\s+(?:or\s+PhD\s+)?degree\s*(?:\([^)]*\))?\s+in\s+([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,3})(?=[,\n\.;])'
    matches7 = re.findall(pattern7, text, re.IGNORECASE)
    for match in matches7:
        field = match.strip().rstrip('.,;/')
        if 2 <= len(field) <= 100 and not field.isdigit() and field.lower() not in ('science', 'degree', 'gpa', 'cum', 'laude', 'honors', 'university', 'college') and not is_excluded_phrase(field):
            fields.append(field)
            
            # Pattern 8: Extract additional comma/or-separated fields in the same clause
            # Handles: "in Computer Science, Computer Engineering, or Electrical Engineering"
            # Scope is limited to the current sentence only (up to . ; newline or " OR ")
            # to prevent scanning the rest of the document and picking up tool names.
            start_idx = text.find(match)
            if start_idx != -1:
                # Find end of this sentence/clause — stop at sentence-ending punctuation or " OR "
                sentence_end = re.search(r'(?:\.|;|\n|\bOR\b)', text[start_idx:], re.IGNORECASE)
                end_idx = start_idx + sentence_end.start() if sentence_end else start_idx + 300
                remaining_text = text[start_idx:end_idx]
            else:
                remaining_text = ""
            additional_fields = re.findall(r',\s*(?:or\s+)?([A-Z][a-z]*(?:\s+[A-Z][a-z]*){0,3})(?=[,\n\.;]|$)', remaining_text)
            for additional in additional_fields:
                field_clean = additional.strip().rstrip('.,;/')
                if 2 <= len(field_clean) <= 100 and not field_clean.isdigit() and field_clean.lower() not in ('science', 'degree', 'gpa', 'cum', 'laude', 'honors', 'university', 'college') and not is_excluded_phrase(field_clean):
                    fields.append(field_clean)
    
    # Final validation: reject garbage fields (< 4 chars, lowercase start, stopwords, activity names)
    # This catches edge cases where patterns slip through
    # Activity words that, when they appear as the LAST word, indicate a non-academic group
    # (e.g. "Math Team", "Robotics Club") — intentionally excludes 'program' to avoid
    # false-negatives for legitimate fields like "Computer Science Program"
    activity_suffix_words = {'team', 'club', 'society', 'organization', 'association',
                             'group', 'chapter', 'league', 'committee', 'council'}
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
        # Skip activity/club/team names: only flag when the LAST word is an activity word
        # e.g. "Math Team" → last word "team" → excluded
        # e.g. "Computer Science Program" → last word "program" → kept
        last_word = field.lower().split()[-1]
        if last_word in activity_suffix_words:
            continue
        valid_fields.append(field)
    
    return list(dict.fromkeys(valid_fields))  # Remove duplicates while preserving order

def extract_education_field_with_llm(text):
    """Extract education fields/majors from job posting using LLM.
    
    Uses GPT-4o to semantically understand education requirements regardless of format.
    Falls back to regex-based extraction if LLM fails or returns empty results.
    
    Args:
        text: Job posting text
    
    Returns:
        list: Sorted list of education fields/majors
    """
    if not text or not text.strip():
        return []
    
    try:
        # Initialize GPT-4o model
        model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
        
        system_prompt = """You are a STRICT education field extraction expert. Your task is to extract ONLY explicitly required education fields/majors from job postings.

ONLY EXTRACT IN DIRECT DEGREE STATEMENTS:
✓ "Bachelor's degree in Computer Science" → extract "Computer Science"
✓ "Master's in Information Technology" → extract "Information Technology"
✓ "Bachelor of Science in Sociology" → extract "Sociology"
✓ "B.S. in Data Science" → extract "Data Science"
✓ "B.S. Political Science" → extract "Political Science"
✓ "Major in Psychology" → extract "Psychology"
✓ "degree with a focus on Accounting" → extract "Accounting"
✓ "degree from an accredited institution in Accounting" → extract "Accounting"

NEVER EXTRACT from job duties/skills context:
✗ "Deep understanding of information technology solutions" → do NOT extract, even though IT is a valid major
✗ "Experience with data science techniques" → do NOT extract "Data Science"
✗ "Cloud platform management" → do NOT extract "Cloud"
✗ "software development experience" → do NOT extract "Software Development"

CRITICAL CONTEXT RULE:
A field is only required education IF it appears in a sentence about DEGREE REQUIREMENTS or EDUCATION REQUIREMENTS.
If it appears in "Responsibilities", "Qualifications" or other job duty sections WITHOUT explicit degree mention, it's a job skill, not an education requirement.

Valid academic fields (extract ONLY if in degree statement):
- Computer Science, Software Engineering, Information Technology, Data Science
- Mechanical Engineering, Electrical Engineering, Chemical Engineering
- Business Administration, Accounting, Finance, Marketing, Economics
- Chemistry, Physics, Biology, Psychology, Mathematics, Statistics
- And other legitimate academic majors

GOLDEN RULE: Look for the pattern "[Degree Type] in/of/with [Field Name]"
- If you see this pattern → EXTRACT the field
- If you just see the term in job duties → IGNORE it

If the posting says "Bachelor's degree" with NO specific field mentioned, return EMPTY list.

Output format: {"education_fields": ["Field1", "Field2"]} or {"education_fields": []} if none required"""
        
        # Call LLM
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract education fields/majors from this job posting:\n\n{text}")
        ]
        
        response = model.invoke(messages)
        response_text = response.content
        
        # Parse JSON response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            fields = parsed.get('education_fields', [])
            
            # Clean up and validate fields
            cleaned_fields = []
            for field in fields:
                field_clean = field.strip().strip('.,;/')
                # Only keep if reasonable length and not pure stopwords
                if field_clean and 2 <= len(field_clean) <= 100 and not field_clean.isdigit():
                    cleaned_fields.append(field_clean)
            
            # CRITICAL VALIDATION: Check if extracted fields actually appear in degree statement patterns
            # This prevents hallucinations where LLM picks up terms from job duties
            validated_fields = []
            
            for field in cleaned_fields:
                field_escaped = re.escape(field)
                degree_patterns = [
                    # Direct: "degree in Computer Science"
                    r"(?:bachelor|master|phd|b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|m\.?b\.?a\.?|degree)[^\n]*?\s+(?:in|of|with)\s+" + field_escaped,
                    # Labeled: "Major: Computer Science"
                    r"(?:major|field|concentration|discipline)\s*:?\s*" + field_escaped,
                    # Comma-separated list: "degree in mathematics, computer science, statistics"
                    # Field appears anywhere after "degree ... in" on the same line/clause
                    r"(?:bachelor|master|phd|b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|m\.?b\.?a\.?|degree)\s+(?:in|of|with)\s+[^;\n]*\b" + field_escaped + r"\b",
                ]
                
                field_found_in_degree_statement = any(
                    re.search(p, text, re.IGNORECASE) for p in degree_patterns
                )
                
                # Only keep if found in degree statement context
                if field_found_in_degree_statement:
                    validated_fields.append(field)
            
            # If we got good results from LLM (after validation), return them and also try regex as supplement
            if validated_fields:
                # Also get regex results to combine for maximum coverage
                regex_fields = extract_education_field(text)
                # Combine, removing duplicates while preserving order
                combined = validated_fields.copy()
                for field in regex_fields:
                    if field not in combined:
                        combined.append(field)
                return combined
    except Exception as e:
        # Silently fall back to regex extraction
        pass
    
    # Fallback to regex-based extraction if LLM fails or returns empty
    return extract_education_field(text)

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
    
    # Normalize degree names for matching
    degree_groups = {
        'high_school': {'high school', 'diploma', 'ged'},
        'associates': {'associate', 'associate\'s', 'a.s.', 'a.a.', 'as', 'aa'},
        'bachelors': {'bachelor', 'bachelor\'s', 'b.s.', 'b.a.', 'bs', 'ba'},
        'masters': {'master', 'master\'s', 'm.s.', 'm.a.', 'ms', 'ma', 'm.b.a.', 'mba'},
        'phd': {'ph.d.', 'phd', 'doctorate', 'doctoral'},
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
        d = degree.lower().replace("’", "'").strip().strip(".,;/")
        d = re.sub(r'\s+', ' ', d)

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

# Extract required and preferred qualifications
def extract_qualifications(text):
    """Extract both REQUIRED and PREFERRED qualifications from job posting.
    
    Handles various bullet formats: -, •, *, ◆, and indented text.
    """
    requirements = []
    preferences = []
    current_section = None
    # Split text into lines for easier processing
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Detect section headers
        if 'required qualifications' in line_lower or 'basic qualifications' in line_lower or 'required skills' in line_lower or 'qualifications you must have' in line_lower:
            current_section = 'required'
            continue
        elif 'preferred qualifications' in line_lower or 'preferred skills' in line_lower or 'desired skills' in line_lower or 'nice to have' in line_lower or 'qualifications we prefer' in line_lower:
            current_section = 'preferred'
            continue
        elif re.search(r'(responsibilities|key responsibilities|offer|about|interview|technical skills|what we offer)', line_lower):
            # Hit a different section, stop collecting qualifications
            current_section = None
            continue
        
        # Extract list items - handle multiple bullet formats: -, •, *, ◆, etc.
        # Also handle indented text (starts with whitespace)
        if current_section:
            stripped = line.strip()
            # Check for common bullet characters and list markers
            bullet_match = re.match(r'^[\-•\*◆\+\◊\▪\▫\○]+\s+(.+)', stripped)
            
            if bullet_match:
                # Bullet format: "- item" or "• item"
                item = bullet_match.group(1).strip()
            elif stripped and line.startswith((' ', '\t')) and not line_lower.startswith(('required', 'preferred')):
                # Indented line without explicit bullet
                item = stripped
            else:
                item = None
            
            if item and len(item) > 5:  # Skip very short items
                if current_section == 'required':
                    requirements.append(item)
                elif current_section == 'preferred':
                    preferences.append(item)
    
    return requirements, preferences

def extract_resume_skills(rag_instance=None, resume_text=None):
    """Extract skills from resume using RAG to retrieve relevant skill sections.
    
    If RAG instance is provided, retrieves relevant chunks first for more focused extraction.
    Falls back to full text only if RAG returns very few results.
    
    Args:
        rag_instance: RAGSystem instance with loaded resume chunks (optional)
        resume_text: Full resume text (required if rag_instance is None)
    
    Returns:
        list: Sorted list of unique skills extracted from resume
    """
    # If RAG instance provided, retrieve relevant chunks (preferred - more focused)
    rag_skills = []
    if rag_instance and rag_instance.vectorstore:
        try:
            skills_chunks = rag_instance.retrieve_relevant_chunks(
                "skills technical abilities programming languages tools frameworks experience projects C++ Java Python", 
                top_k=5
            )
            text = "\n".join([chunk.page_content for chunk in skills_chunks])
            rag_skills = extract_skills(text)
        except Exception:
            # Silently continue to fallback
            pass
    
    # Fallback to full text ONLY if RAG returned very few results
    # This catches cases where RAG misses important skills
    if len(rag_skills) < 5 and resume_text:
        full_text_skills = extract_skills(resume_text)
        # Take full text results only if significantly better than RAG
        if len(full_text_skills) > len(rag_skills) * 1.5:
            return full_text_skills
    
    return rag_skills if rag_skills else []


def extract_resume_education_degree(rag_instance=None, resume_text=None):
    """Extract degree level(s) from a resume using RAG + LLM fallback.

    Mirrors the same 3-layer pattern as extract_resume_skills / extract_resume_seniority:
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

    # Layer 1 – RAG retrieval
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

    # Layer 2 – fast regex pass
    regex_result = extract_education(candidate_text)
    if regex_result:
        return regex_result

    # Also try on full text before going to LLM (catches sections not in top chunks)
    if resume_text and candidate_text != resume_text:
        regex_full = extract_education(resume_text)
        if regex_full:
            return regex_full

    # Layer 3 – LLM fallback (only reached when regex found nothing)
    try:
        model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = """You are an education extraction expert reading a resume.
Extract ONLY the degree TYPE(s) this person has earned or is currently pursuing.

Return a JSON object: {"degrees": ["Bachelor's"]}

Valid degree types: Bachelor's, Master's, PhD, Associate's, Diploma, High School, GED.
Do NOT include the field/major (e.g., Computer Science) – only the degree level.
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
      2. LLM (GPT-4o) extracts the field from the retrieved chunk text — preferred
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

    # Layer 1 – RAG retrieval
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

    # Layer 2 – LLM (preferred when API is available; more accurate than regex for
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

    # Layer 3 – regex fallback (no API key or LLM returned empty)
    regex_result = extract_education_field(candidate_text)
    if regex_result:
        return regex_result

    # Also try on full text (catches sections not in top RAG chunks)
    if resume_text and candidate_text != resume_text:
        regex_full = extract_education_field(resume_text)
        if regex_full:
            return regex_full

    return []


# ── Seniority helpers ────────────────────────────────────────────────────────

def _rule_based_job_seniority(text):
    """Rule-based fallback for seniority extraction. Used when LLM is unavailable."""
    text_lower = text.lower()

    # 'graduate', 'fresh', 'fresher' removed — too generic; they appear in senior postings
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

    years = _extract_years_of_experience(text)
    if years is not None:
        return _yoe_to_seniority(years)

    return None


# Pattern for "N years" / "N+ years" / "N-M years" of experience.
# Captures the FIRST number (the floor) since "4+" and "4-6" both imply ≥ 4.
_YOE_PATTERN = re.compile(
    r'(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2})?\s*(?:to\s*\d{1,2}\s*)?years?\s+(?:of\s+)?(?:professional\s+|relevant\s+|industry\s+)?(?:experience|exp)\b',
    re.IGNORECASE,
)


def _extract_years_of_experience(text):
    """Deterministically extract the minimum years of experience required by a job posting.

    Scans for phrases like '4+ years experience', '5-7 years of experience',
    '10 years of professional experience'. Returns the MINIMUM (floor) value,
    since "4+" and "4-6" both mean "at least 4 years."

    Returns:
        int: years of experience required, or None if no signal found.
    """
    if not text:
        return None

    candidates = []
    for match in _YOE_PATTERN.finditer(text):
        try:
            n = int(match.group(1))
            if 0 <= n <= 30:  # sanity bounds; ignore "175 year history" etc.
                candidates.append(n)
        except ValueError:
            continue

    if not candidates:
        return None

    # Use the MAX of the floors — if a posting says "2+ years preferred,
    # 5+ years required", the binding requirement is 5.
    return max(candidates)


def _yoe_to_seniority(years):
    """Map a years-of-experience number to a seniority bucket.

    Buckets match the schema used by extract_job_seniority:
      0-2 yrs  → entry-level
      3-5 yrs  → mid-level
      6-9 yrs  → senior
      10+ yrs  → lead/principal
    """
    if years is None:
        return None
    if years <= 2:
        return 'entry-level'
    if years <= 5:
        return 'mid-level'
    if years <= 9:
        return 'senior'
    return 'lead/principal'


# Title-keyword set used to refine the YoE-based bucket.
# Lead/principal requires BOTH high YoE AND a leadership keyword in the title;
# without the keyword, classification caps at 'senior'.
_LEADERSHIP_TITLE_KEYWORDS = (
    'lead', 'principal', 'staff', 'architect', 'manager', 'director',
    'head of', 'vp', 'vice president', 'chief',
)


def _title_has_keyword(title, keywords):
    """Word-boundary-ish match: keyword appears as its own token in the title."""
    if not title:
        return False
    t = f" {title.lower()} "
    return any(f" {kw} " in t or t.startswith(f"{kw} ") or t.endswith(f" {kw}") for kw in keywords)


# Extract job title and seniority level
def extract_job_title_and_seniority(text):
    """Extract job title and seniority from a job posting.

    Layer 1 — GPT-4o: reads the top of the posting and returns structured JSON.
    Layer 2 — heuristic fallback: first-line title + seniority keyword scan.
    """
    import re

    _level_to_seniority = {
        'intern': 'entry-level', 'entry': 'entry-level', 'junior': 'entry-level',
        'new college grad': 'entry-level', 'recent grad': 'entry-level', 'I': 'entry-level',
        'associate': 'mid-level', 'mid-level': 'mid-level', 'II': 'mid-level',
        'III': 'senior', 'senior': 'senior', 'IV': 'senior',
        'V': 'lead/principal', 'lead': 'lead/principal', 'principal': 'lead/principal',
        'staff': 'lead/principal', 'VI': 'lead/principal', 'manager': 'lead/principal',
        'director': 'lead/principal', 'vp': 'lead/principal', 'c-level': 'lead/principal',
    }
    _valid_levels = {'entry-level', 'mid-level', 'senior', 'lead/principal'}

    # ── Layer 1: LLM ──────────────────────────────────────────────────────────
    try:
        model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
        system_prompt = """You are a job posting analyst. Extract the job title and seniority level from the top of the posting.

Seniority levels:
- "entry-level" : 0-2 years experience, intern / graduate / junior roles
- "mid-level"   : 3-5 years, associate / intermediate roles
- "senior"      : 5-8 years, senior individual-contributor roles
- "lead/principal": 9+ years OR management / leadership responsibility (manager, director, lead, principal, staff, architect, VP)

Return ONLY valid JSON: {"job_title": "<title>", "seniority": "<level>"}
Set seniority to null if it cannot be determined from the title alone."""

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract job title and seniority:\n\n{text[:600]}")
        ]
        response = model.invoke(messages)
        json_match = re.search(r'\{.*?\}', response.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            job_title = parsed.get('job_title') or None
            seniority = parsed.get('seniority') or None
            if seniority not in _valid_levels:
                seniority = None
            if job_title:
                return job_title, seniority
    except Exception:
        pass

    # ── Layer 2: heuristic fallback ───────────────────────────────────────────
    job_title = None
    seniority = None
    first_line = text.split('\n')[0].strip()
    if len(first_line) < 100:
        job_title = first_line
        for level in sorted(seniority_levels, key=len, reverse=True):
            pattern = r'\b' + re.escape(level) + r'\b'
            if re.search(pattern, job_title, re.IGNORECASE):
                seniority = _level_to_seniority.get(level, level)
                break
    return job_title, seniority


def extract_job_seniority(text):
    """Extract seniority level requirement from a job posting.

    Hybrid pipeline (ordered by reliability):
      Layer 1 — Years-of-experience regex (DETERMINISTIC, primary signal).
                If the posting says "4+ years", it's mid-level regardless of
                what the title says. This handles edge cases like
                "AI Engineer I, 4+ years required" which the LLM gets wrong
                because it over-weights the "I" in the title.
      Layer 2 — GPT-4o (only when YoE is absent). Reads title + level keywords
                to classify postings like "Senior Engineer" with no stated YoE.
      Layer 3 — Rule-based keyword scan (final fallback if LLM unavailable).

    Returns:
        str: One of 'entry-level', 'mid-level', 'senior', 'lead/principal', or None
    """
    if not text or not text.strip():
        return None

    # ── Layer 0: Cheap intern-title short-circuit ─────────────────────────────
    # "Intern" in the title is unambiguous — always entry-level, regardless of
    # YoE phrasing elsewhere in the posting (e.g. "0-2 years preferred"). We
    # grab the first non-empty line directly to avoid the LLM call inside
    # extract_job_title_and_seniority.
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if re.search(r'\bintern(ship)?\b', first_line, re.IGNORECASE):
        return 'entry-level'

    # ── Layer 1: Deterministic YoE extraction + title-keyword refinement ──────
    # YoE is the primary signal (ground truth when stated). We only need to
    # consult the title to verify the high-end edge case:
    #   • Roman-numeral suffixes (I, II, III) are paygrades — IGNORED.
    #   • A 'lead/principal' YoE bucket REQUIRES a leadership keyword in the
    #     title (lead, principal, staff, architect, manager, director, VP,
    #     head of, chief). Otherwise we cap at 'senior' — e.g. an Anthropic
    #     "Software Engineer, Safeguards" posting asking for "5-10+ years" is
    #     a senior IC role, not a lead/principal one.
    #
    # OPTIMIZATION: only call extract_job_title_and_seniority (which itself
    # makes an LLM call) when we land in the lead/principal bucket. For
    # entry/mid/senior buckets, YoE alone is sufficient and we skip the call.
    yoe = _extract_years_of_experience(text)
    if yoe is not None:
        bucket = _yoe_to_seniority(yoe)
        if bucket != 'lead/principal':
            return bucket

        title, _ = extract_job_title_and_seniority(text)
        if _title_has_keyword(title, _LEADERSHIP_TITLE_KEYWORDS):
            return 'lead/principal'
        return 'senior'

    # ── Layer 2: LLM (only when YoE is absent) ────────────────────────────────
    try:
        model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
        system_prompt = """You are a job posting analyst. Classify the seniority level required by the posting into exactly one of these four categories:

- "entry-level"    : 0-2 years experience, intern / graduate / junior roles
- "mid-level"      : 3-5 years, associate / intermediate roles
- "senior"         : 6-9 years, senior individual-contributor roles (NOT management)
- "lead/principal" : 10+ years AND explicit management / leadership responsibility
                     (manager, director, lead, principal, staff, architect, VP)

This posting does NOT state explicit years of experience, so classify based on
title and level keywords only:
- "Junior X", "X Intern", "New Grad" → entry-level
- "X" or "X II" (no level prefix) → mid-level
- "Senior X", "Sr. X" → senior
- "Lead X", "Principal X", "Staff X", "X Manager", "X Director" → lead/principal

If the title gives no signal at all (e.g. just "Software Engineer" with no
context), default to "mid-level".

Return ONLY valid JSON: {"seniority": "<level>"}"""

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Classify the seniority level for this job posting:\n\n{text[:3000]}")
        ]
        response = model.invoke(messages)
        json_match = re.search(r'\{.*?\}', response.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            llm_seniority = parsed.get('seniority')
            if llm_seniority in {'entry-level', 'mid-level', 'senior', 'lead/principal'}:
                return llm_seniority
    except Exception:
        pass

    # ── Layer 3: rule-based fallback ──────────────────────────────────────────
    return _rule_based_job_seniority(text)

def extract_resume_seniority(rag_instance=None, resume_text=None, graduation_date=None):
    """Extract seniority level from resume.

    Layer 1 — Graduation date: current students / recent grads (≤2 yrs) → entry-level
    Layer 2 — LLM reads only job-header lines (title + company + date; no bullet points)
    Fallback — Graduation-year math for older grads whose title lines cannot be parsed

    Args:
        rag_instance: RAGSystem instance (unused, kept for API compatibility)
        resume_text: Full resume text (required)
        graduation_date: Optional graduation year (int) to override auto-detection

    Returns:
        str: One of 'entry-level', 'mid-level', 'senior', 'lead/principal', or None
    """
    CURRENT_YEAR = 2026

    # ── Auto-detect graduation year ───────────────────────────────────────────
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

    # ── Layer 1: Graduation date ──────────────────────────────────────────────
    # Current students (future grad) and recent graduates (≤2 yrs) → entry-level
    if graduation_date and graduation_date >= CURRENT_YEAR - 2:
        return 'entry-level'

    # ── Extract job-header lines (title + company + date; no bullet points) ───
    # Non-bullet lines that contain a month–year date or "Present"
    _MONTHS = (r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
               r'January|February|March|April|June|July|August|'
               r'September|October|November|December)')
    date_line_pat = re.compile(
        rf'^(?![\s•\-\*])(.+(?:{_MONTHS}[^\n]*\d{{4}}|Present)[^\n]*)',
        re.MULTILINE | re.IGNORECASE,
    )
    job_header_lines = date_line_pat.findall(resume_text or "")

    # ── Layer 2: LLM on job-header lines ─────────────────────────────────────
    if job_header_lines:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
            lines_text = "\n".join(job_header_lines[:20])

            system_prompt = """You are a career-level expert.

Given ONLY the date-bearing lines from a resume below (each shows a role, company, or time period with a date — no bullet-point descriptions), classify the person's seniority as exactly one of:
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

    # ── Fallback: graduation math for older grads without parseable title lines ─
    if graduation_date:
        years_since_grad = CURRENT_YEAR - graduation_date
        if years_since_grad > 10:   return 'lead/principal'
        if years_since_grad > 5:    return 'senior'
        if years_since_grad > 2:    return 'mid-level'
        return 'entry-level'

    return None


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


# Calculate skill match score between resume and job description
def calculate_skill_match_score(job_required_skills, job_preferred_skills, resume_skills):
    """
    Compare job skills against resume skills.
    
    Args:
        job_required_skills: list of required skills from job posting
        job_preferred_skills: list of preferred skills from job posting
        resume_skills: list of skills extracted/retrieved from resume
    
    Returns:
        dict with match scores and details
    """
    # Helper: ensure the job requirements/prefs are lists of items
    def ensure_skill_list(x):
        # If already an iterable list/tuple, return as list
        if isinstance(x, (list, tuple, set)):
            return list(x)
        # If it's a string, try to split on common separators (newlines, bullets, dashes, commas)
        if isinstance(x, str):
            # Prefer newline-split and dash/list indicators
            parts = []
            # split on newlines first
            for line in x.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # remove leading bullets or dashes
                line = re.sub(r'^[\-\u2022\*\s]+', '', line)
                # if a line contains commas, split further
                if ',' in line and len(line) > 30:
                    subparts = [p.strip() for p in line.split(',') if p.strip()]
                    parts.extend(subparts)
                else:
                    parts.append(line)

            # If splitting produced nothing useful, fallback to extracting skills via noun-chunking
            if not parts:
                return extract_skills(x)

            # If ANY part is very long (>100), it's likely a concatenated mess. Extract skills from the whole string.
            if any(len(p) > 100 for p in parts):
                return extract_skills(x)

            return parts

    req_list = ensure_skill_list(job_required_skills)
    pref_list = ensure_skill_list(job_preferred_skills)

    # (no debug prints)

    # Use the advanced matcher for required and preferred separately
    req_result = calculate_skill_match_score_advanced(req_list, resume_skills)
    pref_result = calculate_skill_match_score_advanced(pref_list, resume_skills)

    required_matches = [m['job_skill'] for m in req_result['matched']]
    preferred_matches = [m['job_skill'] for m in pref_result['matched']]

    # (no debug prints)

    return {
        "required_matches": required_matches,
        "preferred_matches": preferred_matches,
        # keep score format as fraction (0..1)
        "required_score": req_result['coverage_percent'] / 100.0,
        "preferred_score": pref_result['coverage_percent'] / 100.0,
        "details": {"required": req_result, "preferred": pref_result},
    }


def main():
    """Placeholder CLI entrypoint for local manual checks."""
    logger.info("nlp_processor module loaded. No standalone debug action configured.")


def map_skills_to_source(resume_text, resume_skills):
    """
    Map extracted skills to their source across the entire resume.
    This helps the agent understand which skills come from which section/company/project.
    
    Args:
        resume_text: Full resume text
        resume_skills: List of extracted skills
    
    Returns:
        dict mapping resume sections/sources to skills found in that section
        {
            'Dream Team Engineering': ['drizzle', 'next.js', 'postgresql'],
            'L3Harris Technologies': ['c++', 'coverity'],
            'Projects': ['react.js', 'firebase', 'google maps api'],
            'Skills': ['python', 'java', 'sql'],
            'Leadership': [],
            'Education': [],
        }
    """
    skills_by_source = {}
    
    # Split resume into proper sections with clean boundaries
    lines = resume_text.split('\n')
    
    # First pass: identify all major section boundaries and work experience entries
    sections = []  # List of (start_line, end_line, section_name, section_type)
    current_section_start = 0
    current_section_name = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this is a major section header
        major_section_match = re.search(r'^\s*(Work\s+Experience|Internships?|Employment|Career|Projects|Education|Leadership|Awards|Skills|Certifications?|References)\s*(?:\n|:|$)', lines[i], re.IGNORECASE)
        
        if major_section_match:
            # Save previous section if exists
            if current_section_name:
                sections.append((current_section_start, i, current_section_name, 'section'))
            
            current_section_name = major_section_match.group(1).lower()
            current_section_start = i + 1  # Start after the header
            i += 1
            
            # For work experience sections, look ahead for individual work entries
            if current_section_name in ['work experience', 'internships', 'employment', 'career']:
                j = i
                while j < len(lines):
                    # Check for next major section header
                    if re.search(r'^\s*(Projects|Education|Leadership|Awards|Skills|Certifications?|References)\s*(?:\n|:|$)', lines[j], re.IGNORECASE):
                        # We've hit the next section
                        break
                    
                    # Check if this line is a work experience entry
                    work_exp_match = re.search(r'([A-Z][^|]*?)\s*\|\s*([A-Za-z0-9][A-Za-z0-9\s&\-\.]+?)(?:\s{2,}|$)', lines[j])
                    
                    if work_exp_match and not any(kw in lines[j].lower() for kw in ['required', 'preferred', 'qualifications']):
                        company_or_role = work_exp_match.group(1).strip()
                        role_or_company = work_exp_match.group(2).strip()
                        
                        # Heuristic: longer one is company name
                        company_name = company_or_role if len(company_or_role) > len(role_or_company) else role_or_company
                        
                        # Save current section end at this work entry start
                        if j > current_section_start:
                            sections.append((current_section_start, j, current_section_name, 'section'))
                        
                        # Now find where THIS work entry ends
                        k = j + 1
                        while k < len(lines):
                            # Work entry ends when we see next work experience entry or next major section
                            if re.search(r'^\s*(Projects|Education|Leadership|Awards|Skills|Certifications?|References)\s*(?:\n|:|$)', lines[k], re.IGNORECASE):
                                break
                            
                            next_work_match = re.search(r'([A-Z][^|]*?)\s*\|\s*([A-Za-z][A-Za-z\s&]+?)(?:\s{2,}|$)', lines[k])
                            if next_work_match and not any(kw in lines[k].lower() for kw in ['required', 'preferred']):
                                break
                            
                            k += 1
                        
                        # Add this work experience as its own section
                        sections.append((j, k, company_name, 'work_exp'))
                        
                        j = k
                    else:
                        j += 1
                
                # If we exited the loop, mark remaining as end of work section
                current_section_start = j
                i = j
                current_section_name = None
            else:
                i += 1
        else:
            i += 1
    
    # Add final section if exists
    if current_section_name:
        sections.append((current_section_start, len(lines), current_section_name, 'section'))
    
    # Attribute already-extracted resume skills to sections without another LLM
    # call per section. This keeps one analysis from creating a burst of API
    # requests just to build source labels.
    for start, end, section_name, section_type in sections:
        section_lines = lines[start:end]
        section_text = '\n'.join(section_lines)
        section_text_normalized = canonicalize_skill(section_text)
        
        if not section_text.strip():
            skills_by_source[section_name] = []
            continue
        
        matched_skills = []
        for resume_skill in resume_skills:
            normalized_skill = canonicalize_skill(resume_skill)
            skill_pattern = r'(?<![\w+#.])' + re.escape(normalized_skill) + r'(?![\w+#.])'
            if re.search(skill_pattern, section_text_normalized):
                matched_skills.append(resume_skill)
        
        skills_by_source[section_name] = matched_skills
    
    # DE-DUPLICATION: Remove generic section skills if also appear in specific sources
    # Priority: Company/Project > Skills section > Education/Leadership
    generic_sections = {'skills', 'experience', 'work experience', 'internships', 'career', 'employment'}
    specific_sections = {k for k in skills_by_source.keys() 
                        if k.lower() not in generic_sections and k.lower() not in {'education', 'leadership', 'awards', 'projects'}}
    
    if 'skills' in skills_by_source and specific_sections:
        # Get all skills in specific sources
        specific_skills = set()
        for section in specific_sections:
            for skill in skills_by_source[section]:
                specific_skills.add(skill.lower())
        
        # Remove from generic 'skills' section if also in specific sources
        skills_by_source['skills'] = [s for s in skills_by_source['skills'] 
                                      if s.lower() not in specific_skills]
    
    return skills_by_source


if __name__ == "__main__":
    main()
