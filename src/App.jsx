import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

const LOADING_STAGES = [
  'Reading your resume...',
  'Parsing the job description...',
  'Extracting required skills...',
  'Matching your experience...',
  'Checking education and seniority...',
  'Generating personalized feedback...',
  'Almost done...',
];
const STAGE_INTERVAL_MS = 4500;

function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescription, setJobDescription] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [stage, setStage] = useState(LOADING_STAGES[0]);

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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);

    if (!resumeFile) {
      setError('Please upload a resume document (PDF or Word).');
      return;
    }

    if (!jobDescription.trim()) {
      setError('Please paste a job description to analyze against.');
      return;
    }

    if (resumeFile.size > 1 * 1024 * 1024) {
      setError('Resume file size must be under 1MB. Please upload a smaller file or convert it to text format.');
      return;
    }

    setLoading(true);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1.5 * 60 * 1000);

    try {
      const formData = new FormData();
      formData.append('job_description', jobDescription);
      formData.append('resume', resumeFile);

      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || errBody.error || errBody.message || `Request failed (${res.status})`);
      }

      setResult(await res.json());
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Please try again.');
      } else {
        setError(err.message || 'Something went wrong. Please reupload your information and try again.');
      }
    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  return (
    <div className="app-page">
      <header className="app-header">
        <h1>Welcome to Jobmigo!</h1>
        <p>
          Jobmigo is your AI-powered assistant for maximizing your chances of getting your dream internships and jobs!
        </p>
        <p>
          Upload your resume and paste the description of the job you are applying for to get personalized feedback on how well you match the role,
          along with actionable tips to improve your resume!
        </p>
        <p>
          Do not upload sensitive information you do not want AI to process.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="analysis-form">
        <label className="field-label">
          Resume (PDF or Word)
          <input
            type="file"
            accept=".pdf,.doc,.docx"
            onChange={(e) => setResumeFile(e.target.files[0] || null)}
            className="file-input"
          />
          {resumeFile && <small className="file-name">Selected: {resumeFile.name}</small>}
        </label>

        <label className="field-label">
          Job Description
          <textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste the job title, description, and requirements..."
            rows={10}
            className="job-textarea"
          />
        </label>

        <button type="submit" disabled={loading} className="submit-button">
          {loading ? (
            <span className="button-loading">
              <span className="spinner" aria-hidden="true" />
              {stage}
            </span>
          ) : (
            'Analyze'
          )}
        </button>

        {error && <div className="error-message">{error}</div>}
      </form>

      {result && <Results result={result} />}
    </div>
  );
}

function Results({ result }) {
  const { feedback_markdown } = result;

  return (
    <section className="results-section">
      <h2>AI Feedback</h2>
      <div className = "feedback-warning">
        <p>
            Warning: Jobmigo uses AI-generated feedback, which can contain inaccuracies. Additionally, while it is a free tool, it is not intended to replace professional career advisors. Evaluate the feedback critically and use your own best judgment when utilizing this system.
        </p>
      </div>
      <div className="result-card">
        {feedback_markdown ? (
          <ReactMarkdown>{feedback_markdown}</ReactMarkdown>
        ) : (
          <p>No AI feedback was generated. Please try again.</p>
        )}
      </div>
    </section>
  );
}

export default App;
