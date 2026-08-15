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
  const [showJobImageUpload, setShowJobImageUpload] = useState(true);
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

    if (!hasTextJobDescription) {
      setError('Please upload job description images, extract the text, review it, and then analyze.');
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
      formData.append('job_description_text', jobDescription);
      formData.append('resume', resumeFile);

      console.log('[analyze debug] request', {
        apiBase: API_BASE,
        endpoint: `${API_BASE}/analyze`,
        hasTextJobDescription,
        jobImageCount: jobImages.length,
        resumeName: resumeFile.name,
        resumeSize: resumeFile.size,
      });

      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      console.log('[analyze debug] response status', {
        ok: res.ok,
        status: res.status,
        statusText: res.statusText,
      });

      const responseText = await res.text();
      console.log('[analyze debug] raw response preview', responseText.slice(0, 1000));

      let responseBody = {};
      try {
        responseBody = responseText ? JSON.parse(responseText) : {};
      } catch (parseErr) {
        console.error('[analyze debug] failed to parse JSON response', parseErr);
        throw new Error('Backend returned an invalid response.');
      }

      console.log('[analyze debug] parsed response', responseBody);

      if (!res.ok) {
        throw new Error(responseBody.detail || responseBody.error || responseBody.message || `Request failed (${res.status})`);
      }

      setResult(responseBody);
    } catch (err) {
      console.error('[analyze debug] request failed', err);
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
  // Reset the job description input flow.
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
        setShowJobImageUpload(true);
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
          Upload your resume and job description images to get personalized feedback on how well you match the role,
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
            Upload images of the job description, click the "Extract Text from Uploaded images" button, then review the extracted text before analyzing.
          </small>
          <small className="field-hint">
            The system will only process up to 4 images per request.
          </small>
          <small className="field-hint">
            For maximum accuracy, include the job title, description, and requirements.
          </small>
          {showJobImageUpload && (
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => {
                const files = Array.from(e.target.files);
                if (jobImages.length + files.length > 4) {
                  setError('You can only upload up to 4 images for the job description. Press the Clear button to reset and try again.');
                  e.target.value = '';
                  return;
                }
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
          {jobImages.length > 0 && (
            <div className="verify-ocr-extracted-text">
              <p>
                Please verify that the extracted text from the uploaded images is accurate.
              </p>
              <p>
                If the extracted text is incorrect, you can either remove the images and re-upload them, or manually edit the text in the box below.
              </p>
              <button
                type="button"
                className="ocr-extract-button"
                onClick={async () => {
                  try {
                    const formData = new FormData();
                    jobImages.forEach((file) => {
                      formData.append('image_file', file);
                    });

                    const res = await fetch(`${API_BASE}/extract-image-text`, {
                      method: 'POST',
                      body: formData,
                    });

                    if (!res.ok) {
                      throw new Error(`Failed to extract text from images. Status: ${res.status}`);
                    }

                    const data = await res.json();
                    setJobDescription(data.extracted_text || '');
                    setShowJobTextBox(true);
                    setShowJobImageUpload(true);
                  } catch (err) {
                    console.error('Error extracting text from images:', err);
                    setError('Failed to extract text from images. Please try again.');
                  }
                }}
              >
                Extract Text from Uploaded Images
              </button>
              {showJobTextBox && (
                <textarea
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                  placeholder="Review and edit the extracted job description text before analyzing..."
                  rows={10}
                  className="job-textarea"
                />
              )}
            </div>
          )}
          <div className="flex-container">
            <button type="button" onClick={() => inputJobDescriptionText(false, false, true)} className="clear-button">
              Clear
            </button>
          </div>
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
