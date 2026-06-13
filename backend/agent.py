"""
LangChain agent logic for resume and job analysis.
Handles tool calling, agentic workflows, and response generation.
"""
import os
from pathlib import Path
from dotenv import load_dotenv  # type: ignore
from langchain_openai import ChatOpenAI
from pypdf import PdfReader  # type: ignore

from langchain.chat_models import init_chat_model  # type: ignore
from langchain.agents import create_agent  # type: ignore
from langchain_core.messages import HumanMessage  # type: ignore
from langchain_core.tools import tool  # type: ignore
from langgraph.checkpoint.memory import InMemorySaver

from backend.rag_system import RAGSystem, create_retrieve_resume_tool  # type: ignore

# Load environment variables from .env file
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Set OPENAI_API_KEY before running this script.")


saver = InMemorySaver()
print("agent.py loaded")

@tool
def read_project_readme() -> str:
    """Return the project's README so the agent can describe the app."""
    return (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")


def create_job_agent(rag_instance: 'RAGSystem' | None = None):
    """Initialize and return a configured LangChain agent."""
    model = ChatOpenAI(model="gpt-4o", timeout=60, max_retries=2)
    tools = [read_project_readme]
    if rag_instance:
        retrieve_tool = create_retrieve_resume_tool(rag_instance)
        tools.append(retrieve_tool)
    agent = create_agent(model=model, tools=tools, checkpointer=saver)
    return agent

print("Agent creation function defined")

# Helper constants for detecting weak bullets
WEAK_ACTION_VERBS = [
    'assisted', 'helped', 'participated', 'involved', 'worked on', 'responsible for',
    'did', 'handled', 'used', 'learned', 'got', 'put together', 'took care of'
]

NO_METRICS_PATTERNS = [
    r'improved.*(?!.*\d)',  # "improved" without a number
    r'increased.*(?!.*\d)',  # "increased" without a number
    r'reduced.*(?!.*\d)',    # "reduced" without a number
    r'accelerated.*(?!.*\d)', # "accelerated" without a number
]

def extract_weak_bullet_example(resume_text: str) -> dict | None:
    """
    Find a bullet point with weak action verb or missing metrics from resume text.
    Returns dict with original bullet and identified weakness, or None if not found.
    """
    import re
    
    if not resume_text:
        return None
    
    # Split into lines and find bullets
    lines = resume_text.split('\n')
    bullets = [line.strip() for line in lines if line.strip().startswith(('-', '•', '*'))]
    
    # Check each bullet for weak verbs
    for bullet in bullets:
        clean_bullet = bullet.lstrip('-•* ').strip()
        
        # Check for weak action verbs
        for weak_verb in WEAK_ACTION_VERBS:
            if clean_bullet.lower().startswith(weak_verb):
                return {
                    'original': clean_bullet,
                    'issue': 'weak_action_verb',
                    'weak_verb': weak_verb
                }
        
        # Check for missing metrics (has impact words but no numbers)
        for pattern in NO_METRICS_PATTERNS:
            if re.search(pattern, clean_bullet, re.IGNORECASE) and not re.search(r'\d+', clean_bullet):
                return {
                    'original': clean_bullet,
                    'issue': 'missing_metrics',
                    'impact_word': re.search(pattern, clean_bullet, re.IGNORECASE).group().split()[0]
                }
    
    return None

def extract_work_experience_companies(resume_text: str) -> list[str]:
    """
    Extract company names from ONLY the Work Experiences section of resume.
    Uses the same robust parsing as map_skills_to_source.
    Returns list of company names in order of appearance.
    """
    import re
    
    if not resume_text:
        return []
    
    lines = resume_text.split('\n')
    companies = []
    seen = set()
    in_work_section = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Check for section headers that indicate work experience
        if re.search(r'^\s*(Work\s+Experience|Internships?|Employment|Career)\s*(?:\n|:|$)', line, re.IGNORECASE):
            in_work_section = True
            continue
        # Exit work section if we hit other major sections
        elif re.search(r'^\s*(Projects|Education|Leadership|Awards|Skills|Certifications?|References)\s*(?:\n|:|$)', line, re.IGNORECASE):
            in_work_section = False
            continue
        
        if in_work_section:
            # Check if this line is a work experience entry: "Title | Company" or "Company | Title"
            work_exp_match = re.search(r'([A-Z][^|]*?)\s*\|\s*([A-Za-z0-9][A-Za-z0-9\s&\-\.]+?)(?:\s{2,}|$)', line)
            
            if work_exp_match and not any(keyword in line_stripped.lower() for keyword in ['required', 'preferred', 'qualifications', 'skills:']):
                left_side = work_exp_match.group(1).strip()
                right_side = work_exp_match.group(2).strip()
                
                # Determine which is company vs role with better heuristics
                # Job title keywords suggest it's a role, not a company
                job_title_keywords = ['engineer', 'developer', 'manager', 'analyst', 'specialist', 'architect',
                                     'lead', 'senior', 'junior', 'associate', 'director', 'executive', 'officer',
                                     'coordinator', 'consultant', 'intern', 'assistant']
                
                left_has_title_keywords = any(keyword in left_side.lower() for keyword in job_title_keywords)
                right_has_title_keywords = any(keyword in right_side.lower() for keyword in job_title_keywords)
                
                # If only one side has title keywords, that's the role
                if left_has_title_keywords and not right_has_title_keywords:
                    candidate = right_side  # Right side is company
                elif right_has_title_keywords and not left_has_title_keywords:
                    candidate = left_side  # Left side is company
                else:
                    # Fallback: pick the shorter one as company name (companies are typically shorter)
                    candidate = left_side if len(left_side) < len(right_side) else right_side
                
                # Filter out frameworks and short terms
                candidate_lower = candidate.lower()
                
                # Filter out leadership/organizational terms
                leadership_terms = ['club', 'organization', 'society', 'association', 
                                   'board', 'committee', 'president', 'founder', 'group']
                
                if (len(candidate) > 5 and 
                    candidate not in seen and
                    not any(fw in candidate_lower for fw in ['react', 'node', 'express', 'firebase', 'data science']) and
                    not any(term in candidate_lower for term in leadership_terms)):
                    companies.append(candidate)
                    seen.add(candidate)
    
    return companies



def generate_resume_feedback_prompt(job_title: str, job_description: str, extraction_results: dict, resume_text: str = "") -> str:
    """
    Format extraction results into a structured prompt for the AI agent. Utilize the matching results to provide specific feedback.
    
    Args:
        job_title: Title of the job position
        job_description: Job posting description (context)
        extraction_results: Dict containing:
            - skills_match: dict with required_score, preferred_score, required_matches, preferred_matches
            - education_match: dict with is_match, job_required, resume_fields, warnings
            - seniority_match: dict with is_match, job_seniority, resume_seniority, recommendation, warning
            - qualifications_job: list of qualifications (required + preferred combined)
    
    Returns:
        Formatted prompt string for the agent
    """
    skills = extraction_results.get('skills_match', {})
    education = extraction_results.get('education_match', {})
    seniority = extraction_results.get('seniority_match', {})
    req_quals = extraction_results.get('qualifications_job_required', [])
    
    # Get skills-by-source mapping to show where each skill comes from
    skills_by_source = extraction_results.get('skills_by_source', {})
    
    # Extract company names from resume for context
    companies = extract_work_experience_companies(resume_text)
    company_context = f"companies: {', '.join(companies)}" if companies else "no companies identified"
    
    skills_context = ""
    if skills_by_source:
        skills_context = "\n\nSKILLS ATTRIBUTED TO EACH RESUME SOURCE:\n"
        for source, skills_list in skills_by_source.items():
            if skills_list:
                skills_context += f"- {source}: {', '.join(skills_list)}\n"
    
    # Build base prompt
    prompt = f"""
Please provide personalized resume feedback for a job application to: {job_title}

IMPORTANT: The candidate's WORK EXPERIENCE companies are: {', '.join(companies) if companies else 'Not identified'}
Do NOT mention any companies, non-profit organizations, or leadership positions that are not listed above when discussing work experience.
Only reference the actual employers listed above.

**CANDIDATE WORK EXPERIENCE CONTEXT:**
Work experience at: {company_context}{skills_context}

**SKILLS ANALYSIS:**
- Skill Coverage: {skills.get('required_score', 0):.0%}
- Matched Skills: {', '.join(skills.get('required_matches', [])) or 'None'}"""
    
    # Extract unmatched skills from details
    details = skills.get('details', {})
    req_details = details.get('required', {})
    
    unmatched_required = req_details.get('unmatched', [])
    
    # Format unmatched skills
    unmatched_req_str = ', '.join([u['job_skill'] for u in unmatched_required]) if unmatched_required else 'None'
    
    # Build source attribution for matched skills to guide agent
    source_attribution = ""
    if skills_by_source:
        source_attribution = "\n\nSOURCE ATTRIBUTION FOR MATCHED SKILLS (use this to ensure accurate resume references):\n"
        for source, source_skills in skills_by_source.items():
            matched_in_source = [s for s in source_skills if s.lower() in [m.lower() for m in skills.get('required_matches', [])]]
            if matched_in_source:
                source_attribution += f"- {source}: {', '.join(matched_in_source)}\n"
    
    prompt += f"""
- Missing Skills: {unmatched_req_str}{source_attribution}

**EDUCATION ANALYSIS:**
- Match Status: {'✓ Match' if education.get('is_match') else '✗ Mismatch'}
- Degree Level Match: {'✓ Match' if education.get('required_degree_matched') else '✗ Mismatch'}
- Job Requires Degree: {', '.join(education.get('required_degree_job', [])) or 'Not specified'}
- Your Degree: {', '.join(education.get('required_degree_resume', [])) or 'Not found'}
- Job Requires Field: {', '.join(education.get('education_field_job', [])) or 'Not specified'}
- Your Field: {', '.join(education.get('resume_education_fields', [])) or 'Not extracted'}
- Field Match: {'✓ Match' if education.get('education_field_matched') else '✗ Mismatch'}
- Issues: {education.get('warning') or 'None'}

**SENIORITY LEVEL ANALYSIS:**
- Job Level: {seniority.get('job_seniority') or 'Not specified'}
- Your Level: {seniority.get('resume_seniority') or 'Not extracted'}
- Match: {'✓ Match' if seniority.get('is_match') else '✗ Mismatch'}
- Concern: {seniority.get('warning') or 'None'}

**JOB QUALIFICATIONS:**
{chr(10).join(f'- {q}' for q in req_quals) if req_quals else '- None extracted'}
"""
    
    # Add special case logic
    special_cases = []
    
    # SPECIAL CASE: Weak action verbs in resume bullets
    weak_bullet = extract_weak_bullet_example(resume_text)
    if weak_bullet and weak_bullet['issue'] == 'weak_action_verb':
        company_mention = f" at {companies[0]}" if companies else ""
        special_cases.append(f"""
**RECOMMENDATION: Strengthen Action Verbs**
Your resume contains bullets with weak action verbs like "{weak_bullet['weak_verb']}", which diminish your impact.

Example from your resume{company_mention}:
- Original: "{weak_bullet['original']}"

For this role, consider rewriting to use stronger action verbs like: "architected", "developed", "led", "implemented", "optimized", "spearheaded", "delivered", or "pioneered". 

Please suggest a rewritten version of this bullet that uses a stronger action verb and better highlights your contribution to this specific role.
""")
    
    # SPECIAL CASE: Repetitive use of any action verb more than once across bullets
    _bullet_starts = ('•', '▪', '‣', '⁃', '◦', '-', '*')
    _all_bullets = [
        line.strip() for line in resume_text.split('\n')
        if line.strip().startswith(_bullet_starts)
    ]
    # Count first words (action verbs) across all bullets
    from collections import Counter
    _first_words = [
        b.lstrip('•▪‣⁃◦-* ').split()[0].rstrip('.,;:').lower()
        for b in _all_bullets
        if b.lstrip('•▪‣⁃◦-* ').split()
    ]
    _verb_counts = Counter(_first_words)
    _repeated = {v: c for v, c in _verb_counts.items() if c > 1}
    if _repeated:
        # Pick the most-repeated verb
        _top_verb, _top_count = max(_repeated.items(), key=lambda x: x[1])
        # Find the first bullet that uses it as an example for the LLM
        _example_bullet = next(
            (b for b in _all_bullets
             if b.lstrip('•▪‣⁃◦-* ').lower().startswith(_top_verb)),
            None
        )
        company_mention = f" at {companies[0]}" if companies else ""
        example_line = f'\n- Original: "{_example_bullet}"' if _example_bullet else ""
        all_repeated = ", ".join(f'"{v}" ({c}x)' for v, c in sorted(_repeated.items(), key=lambda x: -x[1]))
        special_cases.append(f"""**RECOMMENDATION: Vary Action Verbs**
Your resume uses the verb "{_top_verb}" to start {_top_count} bullet points, which makes your contributions seem repetitive. Other repeated verbs: {all_repeated}.

Example bullet{company_mention}:{example_line}

Consider what this bullet is actually accomplishing and rewrite it with a stronger, more specific verb that reflects your ownership and impact in the context of this role.
Please suggest a rewritten version of the example bullet above using a different, more precise action verb suited to the job description.""")



    # SPECIAL CASE: Missing metrics in achievements
    if weak_bullet and weak_bullet['issue'] == 'missing_metrics':
        company_mention = f" at {companies[0]}" if companies else ""
        special_cases.append(f"""
**RECOMMENDATION: Quantify Your Achievements**
Your resume contains bullets that describe impact but lack quantifiable metrics, weakening their effectiveness.

Example from your resume{company_mention}:
- Original: "{weak_bullet['original']}"

Employers want to see concrete numbers: percentages, dollar amounts, response time reductions, user acquisition, performance improvements, etc.

Please suggest a rewritten version of this bullet that includes specific metrics or quantifiable results relevant to this role.
""")
    
    # SPECIAL CASE 1: Severely overqualified
    if seniority.get('is_overqualified') and seniority.get('resume_seniority') == 'lead/principal':
        if seniority.get('job_seniority') in ['entry-level', 'mid-level']:
            company_mention = f" with {', '.join(companies)}" if companies else ""
            special_cases.append(f"""
**SPECIAL CONSIDERATION - OVERQUALIFICATION:**
Your experience{company_mention} indicates you are significantly overqualified for this role. Address this directly:
1. Explain your career trajectory and why you're interested in this specific role
2. Emphasize what you bring beyond seniority (mentorship, stability, proven track record)
3. Consider whether this is truly a good fit for your growth trajectory
4. If applying anyway, frame it as a lateral move or intentional career pivot
""")
    
    # SPECIAL CASE 2: Severely underqualified
    if seniority.get('is_underqualified') and seniority.get('resume_seniority') == 'entry-level':
        if seniority.get('job_seniority') in ['senior', 'lead/principal']:
            company_mention = f" at {', '.join(companies)}" if companies else ""
            special_cases.append(f"""**SPECIAL CONSIDERATION - UNDERQUALIFICATION:**
Your experience{company_mention} shows you are significantly underqualified for this role. Consider the following:
1. This role may require significant ramp-up time and may be a stretch
2. Gain more years of experience or consider intermediate roles to build up to this level
3. Focus on transferable skills and your ability to learn quickly
""")
    
    # SPECIAL CASE 3: No skill matches at all
    if skills.get('required_score', 0) < 0.1:
        special_cases.append("""
**SPECIAL CONSIDERATION - SKILL GAP:**
You have very few matching required skills. Before applying:
1. This role may require significant ramp-up time
2. Focus your resume on transferable skills and learning ability
3. Consider if there are prerequisites or courses that would help
4. Be prepared to discuss how you'll quickly acquire the necessary skills
""")
    
    # SPECIAL CASE 4: Partial skill match
    if skills.get('required_matches') and skills.get('required_score', 0) < 0.5:
        special_cases.append("""**SPECIAL CONSIDERATION - PARTIAL SKILL MATCH:**
You have some required skills but are missing others. To address this:  
1. Emphasize the skills you do have and how they are relevant
2. Consider adding a 'Skills in Progress' section to show you're actively developing missing skills
3. Be prepared to discuss how your existing skills will help you quickly learn the missing ones
4. If the missing skills are critical, consider if this role is the right fit
""")
    
    # SPECIAL CASE 5: Field mismatch (even if degree matches)
    field_mismatch = education.get('education_field_job') and education.get('resume_education_fields') and not education.get('education_field_matched')
    if field_mismatch and education.get('required_degree_matched'):
        # Degree matches but field doesn't
        job_fields = ", ".join(education.get('education_field_job', []))
        resume_fields = ", ".join(education.get('resume_education_fields', []))
        
        if skills.get('required_score', 0) <= 0.7:
            special_cases.append(f"""
**SPECIAL CONSIDERATION - EDUCATION FIELD MISMATCH:**
While your degree level matches, your field ({resume_fields}) differs from the job requirement ({job_fields}).

You can still be competitive by:
1. Highlighting relevant coursework, projects, or self-directed learning aligned with the required field
2. Demonstrating how concepts from your field ({resume_fields}) transfer to {job_fields}
3. Showing willingness to rapidly develop domain knowledge in the required field
4. Emphasizing other strengths (experience, technical skills, proven ability to learn quickly)
""")
    
    # SPECIAL CASE 6: Degree mismatch
    degree_mismatch = not education.get('required_degree_matched')
    if degree_mismatch:
        job_degrees = ", ".join(education.get('required_degree_job', []))
        resume_degrees = ", ".join(education.get('required_degree_resume', []))
        
        if skills.get('required_score', 0) <= 0.7:
            special_cases.append(f"""
**SPECIAL CONSIDERATION - EDUCATION DEGREE MISMATCH:**
Degree mismatch: Job requires {job_degrees}, your resume shows {resume_degrees}.

Despite the education gap, you can still be competitive by:
1. Highlighting relevant coursework, projects, or self-directed learning aligned with the job requirements
2. Demonstrating transferable knowledge and skills
3. Showing willingness to pursue relevant certifications or further education
4. Emphasizing other strengths (experience, technical skills, proven ability to learn quickly)
""")
    
    # SPECIAL CASE 7: Education mismatches (degree or field) with strong skills
    if (not education.get('is_match') or degree_mismatch or field_mismatch) and skills.get('required_score', 0) > 0.7:
        company_mention = f" at {', '.join(companies)}" if companies else ""
        special_cases.append(f"""
**SPECIAL CONSIDERATION - EDUCATION MISMATCH WITH STRONG SKILLS:**
Despite education gaps, your experience{company_mention} and skills are strong. Leverage this:
1. Emphasize that your practical experience compensates for the degree/field mismatch
2. Highlight relevant projects and certifications
3. Show evidence of self-learning in the required field
4. Many companies value demonstrated ability over credentials for technical roles
""")
    
    # Append special cases to prompt
    if special_cases:
        prompt += "\n" + "\n".join(special_cases)
    
    # Add base recommendations
    prompt += """

**CRITICAL FEEDBACK REQUIREMENTS:**
If any warnings, red flags, or mismatches are listed above (seniority mismatch, education mismatch, low skill coverage, partial qualifications), you MUST directly address them in your feedback below. Do not omit or minimize these concerns. Include the specific recommendations provided.

**IMPORTANT - SKILL SOURCE ATTRIBUTION:**
When referencing matched skills in your feedback, you MUST use the "SOURCE ATTRIBUTION FOR MATCHED SKILLS" section above to determine where each skill comes from (company, project, skills section, etc.). Never infer or guess the source of a skill - use the attribution provided. For example, if it says "L3Harris Technologies: C++", always reference C++ as coming from L3Harris, not any other company.

Based on this analysis, please provide:
1. **Gap Analysis - Skills and Qualifications**
   Specifically reference the missing required and preferred skills listed above. For each critical gap, suggest:
   - Why it matters for this role
   - How to acquire or credibly position existing experience to address it
   If there are 3+ missing required skills, prioritize the top 3 most critical.
   For ALL mismatches, warnings, or red flags noted above (seniority level, education field/degree, skill gaps, qualifications), provide direct and honest assessment with specific recommendations.

2. **Strengths of your current resume**
   Reference the specific matched skills and qualifications from above. Select 2-3 that are most relevant to this role. Explain how each one differentiates the candidate.
   **CRITICAL: When mentioning any matched skill, reference the source (company, project, section) from the SOURCE ATTRIBUTION FOR MATCHED SKILLS section. This ensures you're attributing skills to the correct source.**
   If you reference work experience, you must mention the specific company name from the attribution, and only reference companies identified in the resume's Work Experience section. Do NOT mention any companies, non-profit organizations, or leadership positions that are not listed above when discussing work experience.
   Talk about what the resume did well in showcasing these strengths.
   
3. **Gap-Bridging Strategy - How to Use Your Strengths to Reach For Missing Skills**
   Your matched skills are foundational. For the most critical missing skills:
   - Identify which of your matched skills are most transferable to learning the missing ones
   - Suggest specific projects, certifications, or examples from your resume that can bridge the gap
   - Propose concrete actions (how to learn, what to highlight in application materials, conversation points for interviews)
"""
    
    # DEBUG: Print the full prompt being sent to agent
    print("\n" + "="*80)
    print("[DEBUG] FINAL PROMPT BEING SENT TO AGENT:")
    print("="*80)
    print(prompt.strip())
    print("="*80 + "\n")
    
    return prompt.strip()

def run_agent_analysis(prompt: str, rag_instance: 'RAGSystem' | None = None) -> str:

    """Run agent analysis with given prompt and RAG instance, return response."""
    agent = create_job_agent(rag_instance=rag_instance)
    run_config = {"configurable": {"thread_id": "job-app-helper-session"}}
    
    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=run_config,
    )
    
    return result["messages"][-1].content
