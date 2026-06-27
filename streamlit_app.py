import os
import tempfile
import time
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

from backend.agent import run_agent_analysis, generate_resume_feedback_prompt
from backend.rag_system import RAGSystem
from backend.nlp_processor import (
    extract_qualifications, extract_skills, extract_skills_with_llm, extract_resume_skills, extract_job_title_and_seniority,
    calculate_skill_match_score, extract_education, extract_education_with_llm, extract_education_field, extract_education_field_with_llm,
    extract_resume_education_degree, extract_resume_education_field,
    match_education, extract_job_seniority, extract_resume_seniority, match_seniority
)

# TODO: Import requests when backend API is ready
# import requests

def check_field_match(job_fields, resume_fields):
    """Check if any resume field matches any job field (fuzzy match).
    
    Args:
        job_fields: List of required education fields from job posting
        resume_fields: List of education fields from resume
    
    Returns:
        Tuple of (match_found: bool, matched_job: str, matched_resume: str)
    """
    # If job posting doesn't specify a required field, any field is acceptable (no mismatch)
    if not job_fields:
        return True, None, None
    
    # If job requires a field but resume has none, no match found
    if not resume_fields:
        return False, None, None
    
    # Strict keyword matching - check if any keywords overlap
    # For strict academic field matching, require meaningful keyword overlap
    for job_field in job_fields:
        job_words = set(job_field.lower().split())
        for resume_field in resume_fields:
            resume_words = set(resume_field.lower().split())
            # Check if there's meaningful overlap
            overlap = job_words & resume_words
            max_len = max(len(job_words), len(resume_words))
            # Require at least 50% overlap for multi-word fields, or exact match for single words
            if len(job_words) == 1 and len(resume_words) == 1:
                # Single word fields must match exactly
                if job_words == resume_words:
                    return True, job_field, resume_field
            elif len(overlap) / max_len >= 0.5:
                return True, job_field, resume_field
    
    return False, None, None

# Page configuration
st.set_page_config(
    page_title="Job Application Helper",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<div class="main-title">💼 Job Application Helper</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Enhance your resume and prepare for your dream job</div>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("About This App")
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")
        st.markdown(readme_content)
    st.divider()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Resume Upload")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF or TXT)",
        type=["pdf", "txt"],
        help="Upload a PDF or TXT file of your resume",
    )
    if uploaded_file:
        st.success(f"✓ Uploaded: {uploaded_file.name}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.header("🎯 Job Information")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste the job title, job description, and requirements. The more details you provide, the better the feedback!",
        height=200,
        placeholder="Paste here",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Analysis section
st.divider()
st.header("🤖 AI Analysis")

if st.button("Analyze & Generate Suggestions", type="primary", use_container_width=True):
    if not job_description and not uploaded_file:
        st.warning("Please provide either a resume or job description")
    else:
        with st.spinner("🔄 Analyzing your information..."):
            try:
                resume_text = ""
                # Extract job-side fields early so we can compute matches whether or not a resume file is uploaded
                job_required = extract_qualifications(job_description)[0] if job_description else ""
                job_preferred = extract_qualifications(job_description)[1] if job_description else ""
                job_skills = extract_skills_with_llm(job_description, context='job_posting') if job_description else []
                expected_education = extract_education(job_description) if job_description else []
                job_level = extract_job_title_and_seniority(job_description)[1] if job_description else None

                if uploaded_file:
                    is_pdf = (
                        uploaded_file.type == "application/pdf"
                        or uploaded_file.name.lower().endswith(".pdf")
                    )
                    if is_pdf:
                        try:
                            pdf_reader = PdfReader(uploaded_file)
                            pages_text = [page.extract_text() or "" for page in pdf_reader.pages]
                            resume_text = "\n".join(pages_text).strip()
                        except Exception as e:
                            st.error(f"Could not read PDF: {e}. Try saving the PDF with a different PDF reader and re-uploading.")
                    else:
                        resume_text = uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()

                if uploaded_file and not resume_text:
                    st.error("Could not extract text from the uploaded resume.")
                else:
                    embedding_backend = os.getenv("RAG_EMBEDDING_BACKEND", "auto")
                    rag = RAGSystem(
                        collection_name=f"streamlit_resume_{int(time.time())}",
                        embedding_backend=embedding_backend,
                        persist_directory=None,
                    )

                    # Load resume into RAG for chunk retrieval (do this first so all extractions can use RAG)
                    if resume_text:
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
                            tmp.write(resume_text)
                            tmp_path = tmp.name
                        try:
                            chunks = rag.load_and_process_document(tmp_path, chunk_size=500, overlap=50)
                            rag.create_vectorstore(chunks)
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

                    # Compute and display skill-match results (works even without an uploaded PDF)
                    # Use LLM-based extraction for more accurate skill identification
                    resume_skills = None
                    if resume_text:
                        try:
                            resume_skills = extract_skills_with_llm(resume_text, context='resume')
                        except Exception as e:
                            # Fallback to direct extraction if LLM fails
                            resume_skills = extract_resume_skills(resume_text=resume_text)
                    else:
                        resume_skills = []
                    
                    # Combine explicit qualifications (required + preferred) and parsed job skills into a single skills list.
                    # Treats required and preferred as one unified "skills" criteria.
                    req_bullets = job_required if isinstance(job_required, list) else []
                    pref_bullets = job_preferred if isinstance(job_preferred, list) else []
                    req_for_match = list(dict.fromkeys(job_skills + req_bullets + pref_bullets))

                    match = calculate_skill_match_score(req_for_match, [], resume_skills)
                    st.write(f"**Skill coverage:** {match['required_score']:.0%}")
                    st.write("**Matched skills:** " + (", ".join(match['required_matches']) if match['required_matches'] else "None"))

                    # Extract and match education
                    st.divider()
                    st.subheader("📚 Education Analysis")
                    
                    # Use LLM-based extraction for better coverage of degree types
                    job_required_education = extract_education_with_llm(job_description) if job_description else []
                    # Use LLM-based extraction for better coverage of education fields
                    job_required_education_fields = extract_education_field_with_llm(job_description) if job_description else []

                    # Use dedicated RAG+LLM functions for resume-side education extraction
                    rag_instance = rag if (resume_text and rag.vectorstore) else None
                    resume_education = extract_resume_education_degree(rag_instance=rag_instance, resume_text=resume_text if resume_text else None)
                    resume_education_fields = extract_resume_education_field(rag_instance=rag_instance, resume_text=resume_text if resume_text else None)
                    
                    education_match = match_education(
                        job_required_education, 
                        [],  # No preferred education in this simple case
                        resume_education,
                        resume_education_fields
                    )
                    
                    # Check field matching
                    field_match_found, matched_job_field, matched_resume_field = check_field_match(
                        job_required_education_fields, 
                        resume_education_fields
                    )
                    
                    
                    # Display education matching results
                    st.write(f"**Job requires:** {', '.join(job_required_education) if job_required_education else 'Not specified'}")
                    st.write(f"**Your education:** {', '.join(resume_education) if resume_education else 'Not found'}")
                    
                    # Display field matching - PROMINENT AND REQUIRED
                    st.markdown("---")
                    st.write("**Field/Major Matching:**")
                    
                    if job_required_education_fields:
                        st.write(f"  • **Job requires:** {', '.join(job_required_education_fields)}")
                    else:
                        st.write(f"  • **Job requires:** Not specified")
                    
                    if resume_education_fields:
                        st.write(f"  • **Your major:** {', '.join(resume_education_fields)}")
                    else:
                        st.write(f"  • **Your major:** Not extracted from resume")
                    
                    # Display field match result - THIS IS CRITICAL
                    if not job_required_education_fields:
                        # Job doesn't specify a field requirement - no field mismatch possible
                        st.info("ℹ️ Job doesn't require a specific field/major. Any relevant degree is acceptable.")
                    elif job_required_education_fields and not resume_education_fields:
                        st.error(f"🚨 **CRITICAL:** Job requires a specific field {job_required_education_fields}, but your major could not be extracted from your resume. Update your resume to explicitly state your major (e.g., 'Major: Your Field') or degree format (e.g., 'Bachelor of Science in Your Field')")
                    elif field_match_found and matched_job_field and matched_resume_field:
                        st.success(f"✓ Your major ({matched_resume_field}) matches the job requirement ({matched_job_field})!")
                    elif job_required_education_fields and resume_education_fields and not field_match_found:
                        # This is the case where both exist but don't match - CLEARLY INDICATE MISMATCH
                        st.error(f"🚨 **MISMATCH:** Job requires major in {', '.join(job_required_education_fields)}, but your resume shows {', '.join(resume_education_fields)}. These fields do NOT match. Consider if you have relevant transferable skills or if your resume needs to highlight coursework related to {', '.join(job_required_education_fields)}.")
                    
                    # Display degree type match result (secondary)
                    st.markdown("---")
                    st.write("**Degree Level Matching:**")
                    if education_match['required_degree_matched']:
                        st.success("✓ Your degree level matches the job requirement")
                    elif job_required_education and resume_education:
                        st.warning("⚠️ Your degree level may not match the job requirement")
                    elif job_required_education and not resume_education:
                        st.info("ℹ️ Could not extract degree level from resume")
                    
                    st.divider()

                    # Expandable help section for education extraction
                    if not resume_education_fields and resume_text:
                        with st.expander("💡 Tip: Why wasn't your major extracted?"):
                            lines = resume_text.split('\n')
                            education_lines = []
                            in_education = False
                            for i, line in enumerate(lines):
                                if 'education' in line.lower():
                                    in_education = True
                                    education_lines = lines[max(0, i-1):min(len(lines), i+10)]
                                    break

                            if education_lines:
                                st.write("**Found in your resume:**")
                                st.code('\n'.join(education_lines), language="text")

                            st.write("**Supported degree + major formats:**")
                            st.markdown("""
- `Bachelor of Science in Computer Science`
- `B.S. in Computer Science`
- `Bachelor's in Computer Science`
- `Major: Computer Science`
- `Field: Computer Science`
                            """)

                    st.divider()

                    # Seniority Level Analysis
                    st.subheader("💼 Seniority Level Analysis")
                    
                    # Extract job seniority - prioritize title extraction, fall back to full posting
                    job_seniority = None
                    if job_description:
                        # First, try to extract from job title (more reliable)
                        _, job_seniority_from_title = extract_job_title_and_seniority(job_description)
                        
                        # If title extraction found seniority, use that; otherwise search full posting
                        if job_seniority_from_title:
                            job_seniority = job_seniority_from_title
                        else:
                            job_seniority = extract_job_seniority(job_description)
                    
                    # Extract seniority from resume (graduation-date check → LLM on job-header lines → grad-year math)
                    resume_seniority = extract_resume_seniority(resume_text=resume_text) if resume_text else None
                    
                    # Display extraction results
                    st.write(f"**Job requires:** {job_seniority if job_seniority else 'Not specified'}")
                    st.write(f"**Your level:** {resume_seniority if resume_seniority else 'Not extracted'}")
                    
                    # Match seniority
                    if job_seniority or resume_seniority:
                        seniority_match = match_seniority(job_seniority, resume_seniority)
                        
                        st.markdown("---")
                        st.write("**Seniority Matching:**")
                        
                        if seniority_match['is_overqualified']:
                            st.warning(f"⚠️ {seniority_match['warning']}")
                            st.info(f"💡 {seniority_match['recommendation']}")
                        elif seniority_match['is_underqualified']:
                            st.warning(f"⚠️ {seniority_match['warning']}")
                            if seniority_match['recommendation']:
                                st.info(f"💡 {seniority_match['recommendation']}")
                        elif seniority_match['is_match']:
                            st.success(f"✓ Your {resume_seniority} background matches this {job_seniority} role!")
                            if seniority_match['recommendation']:
                                st.info(f"💡 {seniority_match['recommendation']}")
                        elif seniority_match['warning']:
                            st.warning(f"⚠️ {seniority_match['warning']}")
                    else:
                        st.info("ℹ️ Could not determine seniority levels from job posting or resume")
                    
                    with st.expander("ℹ️ How seniority is determined"):
                        st.markdown("""
**From the job posting:**  keywords like *entry-level*, *junior*, *senior*, *lead*, or years-of-experience ranges.

**From your resume:**  job title keywords (*Senior*, *Jr*, *Lead*, *Manager*), years-of-experience statements, or graduation date.
                        """)

                    st.divider()
                    
                    # Compile extraction results for AI feedback
                    # Degree level is the primary education match criteria
                    # Field/major matching is secondary and informational
                    degree_match = education_match.get('required_degree_matched', False)
                    field_match = education_match.get('field_matched', False) or field_match_found
                    # is_match is true if degree matches (primary criterion)
                    # Field mismatch is noted but doesn't fail overall match
                    overall_education_match = degree_match
                    
                    # Map skills to their source (work experience, projects, skills section, etc.)
                    from backend.nlp_processor import map_skills_to_source
                    skills_by_source = map_skills_to_source(resume_text, resume_skills) if resume_skills else {}
                    
                    extraction_results = {
                        'skills_match': match,
                        'skills_by_source': skills_by_source,
                        'education_match': {
                            'is_match': overall_education_match,
                            'job_required_education': job_required_education,
                            'required_degree_matched': degree_match,
                            'required_degree_job': job_required_education,
                            'required_degree_resume': resume_education,
                            'education_field_job': job_required_education_fields,
                            'resume_education_fields': resume_education_fields,
                            'education_field_matched': field_match,
                            'warning': education_match.get('warning', '')
                        },
                        'seniority_match': seniority_match if (job_seniority or resume_seniority) else {},
                        # Combine required + preferred qualifications into one unified list
                        'qualifications_job_required': list(dict.fromkeys(
                            (job_required if isinstance(job_required, list) else [])
                            + (job_preferred if isinstance(job_preferred, list) else [])
                        )),
                        'qualifications_job_preferred': []
                    }
                    
                    # Get job title from first line of job posting if available
                    job_title = job_description.split('\n')[0].strip() if job_description else "Position"
                    
                    # Generate structured prompt with extraction results and resume text for bullet rewrites
                    feedback_prompt = generate_resume_feedback_prompt(job_title, job_description, extraction_results, resume_text=resume_text)
                    
                    # Run agent with structured feedback
                    st.subheader("Resume Feedback")
                    if resume_text and rag.vectorstore:
                        st.success("Analysis complete - Generating personalized feedback...")
                        try:
                            response = run_agent_analysis(feedback_prompt, rag_instance=rag)
                            st.markdown(response)
                        except Exception as e:
                            st.error(f"Failed to generate feedback: {e}")
                    else:
                        st.info("Upload a resume to receive personalized AI feedback for this position")

                    # ── Collapsed debug panel ──────────────────────────────────────
                    with st.expander("🔬 Debug Info", expanded=False):
                        import json as _json
                        st.markdown("**Skills**")
                        st.code(
                            f"Job skills (LLM):       {job_skills}\n"
                            f"Resume skills:          {resume_skills}\n"
                            f"req_for_match count:    {len(req_for_match)}\n"
                            f"Matched skills:         {match['required_matches']}\n"
                            f"Unmatched skills:       {[u['job_skill'] for u in match['details']['required'].get('unmatched', [])]}\n"
                            f"Skill coverage:         {match['required_score']:.0%}",
                            language="text",
                        )
                        st.markdown("**Education**")
                        st.code(
                            f"Job degree (LLM):       {job_required_education}\n"
                            f"Job field  (LLM):       {job_required_education_fields}\n"
                            f"Resume degree (RAG+LLM):{resume_education}\n"
                            f"Resume field  (RAG+LLM):{resume_education_fields}\n"
                            f"Field match:            {field_match_found}  job={matched_job_field}  resume={matched_resume_field}",
                            language="text",
                        )
                        st.markdown("**Seniority**")
                        st.code(
                            f"Job seniority:          {job_seniority}\n"
                            f"Resume seniority:       {resume_seniority}\n"
                            f"Is match:               {seniority_match.get('is_match') if (job_seniority or resume_seniority) else 'N/A'}\n"
                            f"Is overqualified:       {seniority_match.get('is_overqualified') if (job_seniority or resume_seniority) else 'N/A'}\n"
                            f"Is underqualified:      {seniority_match.get('is_underqualified') if (job_seniority or resume_seniority) else 'N/A'}",
                            language="text",
                        )
                        st.markdown("**extraction_results passed to agent**")
                        st.json(extraction_results)


            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

# Footer
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
    Made with ❤️ for job seekers | Powered by LangChain & OpenAI
    </div>
    """,
    unsafe_allow_html=True,
)
