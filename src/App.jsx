// React.js frontend for Job Application Helper.
// Uploads a PDF resume + job description to the FastAPI backend and displays AI analysis.

import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

// Staged messages shown while /analyze is running. The frontend doesn't know
// the backend's real progress, but cycling through these makes the wait feel
// purposeful instead of frozen.
const LOADING_STAGES = [
  'Reading your resume…',
  'Parsing the job description…',
  'Extracting required skills…',
  'Matching your experience…',
  'Checking education & seniority…',
  'Generating personalized feedback…',
  'Almost done…',
];
const STAGE_INTERVAL_MS = 4500;

function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescription, setJobDescription] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [stage, setStage] = useState(LOADING_STAGES[0]);

  // Cycle through staged loading messages while a request is in flight.
  // Stops on the final message so it doesn't loop back to "Reading your resume…".
  useEffect(() => {
    if (!loading) return;
    setStage(LOADING_STAGES[0]);
    let i = 0;
    const id = setInterval(() => {
      i = Math.min(i + 1, LOADING_STAGES.length - 1);
      setStage(LOADING_STAGES[i]);
    }, STAGE_INTERVAL_MS);
    return () => clearInterval(id);
  }, [loading]);

  // async function - function that won't stop the rest of the code from running while it's executing
  //submit the resume and job description to the backend for analysis while showing a loading state and handling errors
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);

    if (!resumeFile && !jobDescription.trim()) {
      setError('Please upload a resume PDF or paste a job description.');
      return;
    }

    //Validate resume file size (max 1MB) to prevent excessively large uploads that could strain the backend or exceed OpenAI limits.
    if (resumeFile && resumeFile.size > 1 * 1024 * 1024) {
      setError('Resume file size must be under 1MB. Please upload a smaller file or convert it to text format.');
      return;
    }

    setLoading(true);
    const controller = new AbortController();
    const {signal} = controller;
    const timeoutId = setTimeout(() => controller.abort(), 1.5 * 60 * 1000); // 1.5 minute timeout
    try {
      //Uncontrolled form is used here so that user can use this app any time without worrying about the form state. FormData is used to handle file uploads and text data together in a single request.
      const formData = new FormData();
      formData.append('job_description', jobDescription);
      if (resumeFile) formData.append('resume', resumeFile);

      // NOTE: do NOT set Content-Type manually — the browser will add the
      // correct multipart/form-data boundary automatically.
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
        signal,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Request failed (${res.status})`);
      }

      setResult(await res.json());
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Please try again.');
      } else {
        setError(err.message || 'Something went wrong.');
      }
    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <style>{`@keyframes jah-spin { to { transform: rotate(360deg); } }`}</style>
      <header style={styles.header}>
        <h1 style={{ margin: 0 }}>Welcome to Jobmigo!</h1>
        <p style={styles.subtitle}>
          Jobmigo is your AI-powered assistant for maximizing your chances of getting your dream internships and jobs! Upload your resume and paste a job description to get personalized feedback on how well you match the role, along with actionable tips to improve your resume.
        </p>
      </header>

      <form onSubmit={handleSubmit} style={styles.form}>
        <label style={styles.label}>
          Resume (PDF or Word)
          <input
            type="file"
            accept=".pdf,.doc,.docx"
            onChange={(e) => setResumeFile(e.target.files[0] || null)}
            style={styles.fileInput}
          />
          {resumeFile && <small style={styles.fileName}>Selected: {resumeFile.name}</small>}
        </label>

        <label style={styles.label}>
          Job Description
          <textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste the job title, description, and requirements..."
            rows={10}
            style={styles.textarea}
          />
        </label>

        <button type="submit" disabled={loading} style={styles.button}>
          {loading ? (
            <span style={styles.buttonLoading}>
              <span style={styles.spinner} aria-hidden="true" />
              {stage}
            </span>
          ) : (
            'Analyze'
          )}
        </button>

        {error && <div style={styles.error}>{error}</div>}
      </form>

      {result && <Results result={result} />}
    </div>
  );
}

// ── Results display ─────────────────────────────────────────────────────────

function Results({ result }) {
  const { skills, education, seniority, qualifications, feedback_markdown } = result;

  return (
    <section style={styles.results}>
      <h2>Results</h2>

      <div style={styles.card}>
        <h3>Skills</h3>
        <p><strong>Coverage:</strong> {(skills.coverage * 100).toFixed(0)}%</p>
        <p><strong>Matched:</strong> {skills.matched.length ? skills.matched.join(', ') : 'None'}</p>
        <p><strong>Missing:</strong> {skills.unmatched.length ? skills.unmatched.join(', ') : 'None'}</p>
      </div>

      <div style={styles.card}>
        <h3>Education</h3>
        <p><strong>Job requires (degree):</strong> {education.job_required_degrees.join(', ') || 'Not specified'}</p>
        <p><strong>Your degree:</strong> {education.resume_degrees.join(', ') || 'Not found'}</p>
        <p><strong>Job requires (field):</strong> {education.job_required_fields.join(', ') || 'Not specified'}</p>
        <p><strong>Your field:</strong> {education.resume_fields.join(', ') || 'Not extracted'}</p>
        <p>
          <strong>Degree match:</strong>{' '}
          <span style={badge(education.degree_matched)}>{education.degree_matched ? '✓ Yes' : '✗ No'}</span>
        </p>
        <p>
          <strong>Field match:</strong>{' '}
          <span style={badge(education.field_matched)}>{education.field_matched ? '✓ Yes' : '✗ No'}</span>
        </p>
      </div>

      <div style={styles.card}>
        <h3>Seniority</h3>
        <p><strong>Job:</strong> {seniority.job || 'Not specified'}</p>
        <p><strong>You:</strong> {seniority.resume || 'Not extracted'}</p>
        {seniority.warning && (
          <p style={{ color: '#b45309' }}><strong>Note:</strong> {seniority.warning}</p>
        )}
        {seniority.recommendation && (
          <p style={{ color: '#0369a1' }}><strong>Tip:</strong> {seniority.recommendation}</p>
        )}
      </div>

      {qualifications && qualifications.length > 0 && (
        <div style={styles.card}>
          <h3>Job Qualifications</h3>
          <ul>{qualifications.map((q, i) => <li key={i}>{q}</li>)}</ul>
        </div>
      )}

      {feedback_markdown && (
        <div style={styles.card}>
          <h3>AI Feedback</h3>
          <ReactMarkdown>{feedback_markdown}</ReactMarkdown>
        </div>
      )}
    </section>
  );
}

// ── Inline styles (swap for CSS modules later if desired) ───────────────────

const styles = {
  page: {
    maxWidth: 800,
    margin: '0 auto',
    padding: '2rem 1rem',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: '#1f2937',
  },
  header: { marginBottom: '2rem' },
  subtitle: { color: '#6b7280', marginTop: '0.25rem' },
  form: { display: 'flex', flexDirection: 'column', gap: '1.25rem' },
  label: { display: 'flex', flexDirection: 'column', fontWeight: 600, gap: '0.5rem' },
  fileInput: { fontWeight: 400 },
  fileName: { color: '#059669', fontWeight: 400 },
  textarea: {
    fontFamily: 'inherit',
    fontSize: '0.95rem',
    padding: '0.75rem',
    border: '1px solid #d1d5db',
    borderRadius: 6,
    resize: 'vertical',
    fontWeight: 400,
  },
  button: {
    padding: '0.75rem 1.25rem',
    background: '#2563eb',
    color: 'white',
    border: 'none',
    borderRadius: 6,
    fontSize: '1rem',
    fontWeight: 600,
    cursor: 'pointer',
  },
  buttonLoading: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.6rem',
    justifyContent: 'center',
  },
  spinner: {
    display: 'inline-block',
    width: 14,
    height: 14,
    border: '2px solid rgba(255,255,255,0.4)',
    borderTopColor: '#fff',
    borderRadius: '50%',
    animation: 'jah-spin 0.8s linear infinite',
  },
  error: {
    padding: '0.75rem',
    background: '#fee2e2',
    color: '#991b1b',
    borderRadius: 6,
  },
  results: { marginTop: '2.5rem' },
  card: {
    background: '#f9fafb',
    border: '1px solid #e5e7eb',
    borderRadius: 8,
    padding: '1.25rem',
    marginBottom: '1rem',
  },
};

const badge = (ok) => ({
  display: 'inline-block',
  padding: '0.15rem 0.5rem',
  borderRadius: 4,
  background: ok ? '#d1fae5' : '#fee2e2',
  color: ok ? '#065f46' : '#991b1b',
  fontWeight: 600,
});

export default App;