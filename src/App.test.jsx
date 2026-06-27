import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App.jsx';

function makeAnalyzeResponse(overrides = {}) {
  return {
    skills: {
      coverage: 0.5,
      matched: ['python'],
      unmatched: ['docker'],
      job_skills: ['python', 'docker'],
      resume_skills: ['python'],
    },
    education: {
      job_required_degrees: [],
      resume_degrees: [],
      job_required_fields: [],
      resume_fields: [],
      degree_matched: false,
      field_matched: false,
    },
    seniority: {
      job: null,
      resume: null,
    },
    qualifications: [],
    feedback_markdown: '',
    ...overrides,
  };
}

function fillRequiredFormFields() {
  const resume = new File(['resume text'], 'resume.pdf', { type: 'application/pdf' });

  fireEvent.change(screen.getByLabelText(/Resume/i), {
    target: { files: [resume] },
  });

  fireEvent.change(
    screen.getByPlaceholderText('Paste the job title, description, and requirements...'),
    { target: { value: 'Some job description text' } }
  );
}

describe('App error handling', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('shows server detail error from non-2xx response', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'Provide a resume, a job description, or both.' }),
    });

    render(<App />);

    fillRequiredFormFields();
    fireEvent.click(screen.getByRole('button', { name: 'Analyze' }));

    await screen.findByText('Provide a resume, a job description, or both.');
  });

  test('shows server error field when detail is absent', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: false,
      status: 429,
      json: async () => ({ error: 'Rate limit exceeded: 10 per hour. Please try again later.' }),
    });

    render(<App />);

    fillRequiredFormFields();
    fireEvent.click(screen.getByRole('button', { name: 'Analyze' }));

    await screen.findByText('Rate limit exceeded: 10 per hour. Please try again later.');
  });

  test('shows timeout message for aborted request', async () => {
    const abortErr = new Error('aborted');
    abortErr.name = 'AbortError';

    vi.spyOn(global, 'fetch').mockRejectedValue(abortErr);

    render(<App />);

    fillRequiredFormFields();
    fireEvent.click(screen.getByRole('button', { name: 'Analyze' }));

    await screen.findByText('Request timed out. Please try again.');
  });

  test('renders AI feedback content from successful response', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      status: 200,
      json: async () =>
        makeAnalyzeResponse({
          feedback_markdown: '_Could not generate AI feedback: AI feedback is taking too long right now._',
        }),
    });

    render(<App />);

    fillRequiredFormFields();
    fireEvent.click(screen.getByRole('button', { name: 'Analyze' }));

    await waitFor(() => {
      expect(screen.getByText('AI Feedback')).toBeInTheDocument();
      expect(
        screen.getByText('Could not generate AI feedback: AI feedback is taking too long right now.')
      ).toBeInTheDocument();
    });
  });
});
