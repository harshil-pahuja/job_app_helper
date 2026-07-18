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
  const [showJobTextBox, setShowJobTextBox] = useState(false);
  const [showJobImageUpload, setShowJobImageUpload] = useState(false);
  const [jobImages, setJobImages] = useState([]);
  const [jobDescription, setJobDescription] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [stage, setStage] = useState(LOADING_STAGES[0]);
  const [clearJobDescription, setClearJobDescription] = useState(false);

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
    const hasTextJobDescription = jobDescription.trim().length > 0;
    const hasImageJobDescription = jobImages.length > 0;

    if (!hasTextJobDescription && !hasImageJobDescription) {
      setError('Please provide a job description to analyze against, either by pasting text or uploading images.');
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
      if (hasTextJobDescription) {
        formData.append('job_description_text', jobDescription);
      } else if (hasImageJobDescription) {
        jobImages.forEach((file) => {
          formData.append('job_description_image', file);
        });
      }
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
  // Async function - this can happen any time the user clicks the "Insert Sample Job Description" button or the "Upload Job Description Image" button. It will fetch a sample job description from the backend and set it in the jobDescription state.
  async function inputJobDescriptionText(isImageUpload, isTextInput, isClear = false) {

    if (isTextInput) {
        setShowJobTextBox(true);   // Tell React to display it
        setShowJobImageUpload(false);   // Hide the image upload option
        setJobImages([]);   // Clear any previously uploaded images
    }

    if (isImageUpload) {
        setShowJobImageUpload(true);   // Tell React to display it
        setShowJobTextBox(false);   // Hide the text input option
        setJobDescription('');   // Clear any previously pasted text
    }
    if (isClear) {
        setJobDescription('');
        setShowJobTextBox(false);
        setShowJobImageUpload(false);
        setClearJobDescription(true);
        setJobImages([]);
    }
}

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
          Do not upload sensitive information you do not want AI to access.
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
          <span>Job Description</span>
          <small className="field-hint">
            Choose whether to paste the job description text or upload images of the job description. 
          </small>
          <small className="field-hint">
            Only one method can be used at a time. If you paste text, the image upload will be ignored. If you upload images, the pasted text will be ignored.
          </small>
          <small className="field-hint">
            If you choose to upload images, the system will only process up to 4 images per request.
          </small>
          <small className="field-hint">
            For maximum accuracy, include the job title, description, and requirements.
          </small>
          <button type="button" onClick={() => inputJobDescriptionText(false, true)} className="insert-text-button">
            Insert Sample Job Description
          </button>
          {showJobTextBox && (
            <textarea
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the job title, description, and requirements..."
              rows={10}
              className="job-textarea"
            />
          )}
          <button type="button" onClick={() => inputJobDescriptionText(true, false)} className="image-upload-button">
            Upload Job Description Image
          </button>
          {showJobImageUpload && (
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => {
                const files = Array.from(e.target.files);
                if (files.length > 4) {
                  setError('You can only upload up to 4 images for the job description.');
                  return;
                }
                // Here you would handle the image files, e.g., send them to the backend for processing.
                // For now, we just log them.
                setJobImages((prevImages) => [...prevImages, ...files]);
                setError(''); // Clear any previous error
                e.target.value = ''; // Reset the input so the same file can be selected again if needed
              }}
              className="image-input"
            />
          )}
          <div className="image-preview-container">
            {jobImages.map((file, index) => (
              <div
                key={`${file.name}-${file.lastModified}-${index}`}
                className="image-preview"
              >
                <img
                  src={URL.createObjectURL(file)}
                  alt={`Uploaded job description ${index + 1}`}
                />

                <p>{file.name}</p>

                <button
                  type="button"
                  onClick={() => {
                    setJobImages((prevImages) =>
                      prevImages.filter((_, imageIndex) => imageIndex !== index)
                    );
                  }}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
          <button type="button" onClick={() => inputJobDescriptionText(false, false, true)} className="clear-button">
            Clear
          </button>
          {clearJobDescription}
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
