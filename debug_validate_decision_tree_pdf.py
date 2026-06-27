r"""Debug backend.api.validate_resume_text against resume-validation cases.

Usage:
    .\.venv\Scripts\python.exe debug_validate_decision_tree_pdf.py

What this tests:
    1. The real ACS resume PDF fixture:
       tests/resumes/ACS_Resume - Chuang Saladin, Andrew R..pdf
    2. Each HTTPException branch inside validate_resume_text using synthetic text.

The branch tests mock ChatOpenAI so they do not spend OpenAI API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types

from fastapi import HTTPException
from pypdf import PdfReader


def _install_api_import_stubs() -> None:
    """Stub unrelated backend modules so importing backend.api stays lightweight.

    This script only tests validate_resume_text. backend.api imports the full
    analysis pipeline at module import time, which can load RAG/torch/model
    dependencies that are irrelevant for these validation checks.
    """
    agent_stub = types.ModuleType("backend.agent")
    agent_stub.generate_resume_feedback_prompt = lambda *args, **kwargs: ""
    agent_stub.run_agent_analysis = lambda *args, **kwargs: ""

    nlp_stub = types.ModuleType("backend.nlp_processor")
    for name in [
        "calculate_skill_match_score",
        "extract_education_field_with_llm",
        "extract_education_with_llm",
        "extract_job_seniority",
        "extract_job_title_and_seniority",
        "extract_qualifications",
        "extract_resume_education_degree",
        "extract_resume_education_field",
        "extract_resume_seniority",
        "extract_resume_skills",
        "extract_skills_with_llm",
        "map_skills_to_source",
        "match_education",
        "match_seniority",
    ]:
        setattr(nlp_stub, name, lambda *args, **kwargs: [] if "extract" in name else {})

    rag_stub = types.ModuleType("backend.rag_system")

    class _StubRAGSystem:
        vectorstore = None

        def __init__(self, *args, **kwargs):
            pass

        def load_and_process_document(self, *args, **kwargs):
            return []

        def create_vectorstore(self, *args, **kwargs):
            self.vectorstore = None

    rag_stub.RAGSystem = _StubRAGSystem

    sys.modules.setdefault("backend.agent", agent_stub)
    sys.modules.setdefault("backend.nlp_processor", nlp_stub)
    sys.modules.setdefault("backend.rag_system", rag_stub)


_install_api_import_stubs()

import backend.api as api_module
from backend.api import (
    RESUME_OPTIONAL_SIGNALS,
    RESUME_REQUIRED_SECTIONS,
    name_patterns,
    re,
    validate_resume_text,
)


PDF_PATH = Path("tests/resumes/ACS_Resume - Chuang Saladin, Andrew R..pdf")


class _FakeValidationResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeAcceptingChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt):
        return _FakeValidationResponse(
            """
{
  "is_resume": true,
  "document_type": "resume",
  "confidence": 0.95,
  "reason": "Mocked resume validation response."
}
"""
        )


class _FakeRejectingChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt):
        return _FakeValidationResponse(
            """
{
  "is_resume": false,
  "document_type": "research_paper",
  "confidence": 0.98,
  "reason": "Mocked non-resume validation response."
}
"""
        )


@dataclass
class ValidationCase:
    name: str
    text: str
    expected_detail_contains: str
    fake_llm: type = _FakeAcceptingChatOpenAI


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def build_text(
    *,
    include_required: bool = True,
    include_optional: bool = True,
    include_name: bool = True,
    include_contact: bool = True,
) -> str:
    lines = []

    if include_name:
        lines.append("John Smith")
    else:
        lines.append("candidate profile")

    if include_contact:
        lines.append("john.smith@example.com")

    if include_required:
        lines.extend(RESUME_REQUIRED_SECTIONS)
    else:
        lines.extend(section for section in RESUME_REQUIRED_SECTIONS if section != "education")

    if include_optional:
        lines.extend(RESUME_OPTIONAL_SIGNALS)

    lines.extend(
        [
            "built python applications for backend services",
            "created frontend interfaces with javascript and react",
            "managed data workflows and collaborated with engineering teams",
            "documented technical decisions and delivered production features",
        ]
        * 8
    )

    return "\n".join(lines)


def print_local_signals(text: str) -> None:
    normalized = text.lower()
    header = text[:500]

    required_sections_found = [
        section for section in RESUME_REQUIRED_SECTIONS if section in normalized
    ]
    optional_signals_found = [
        signal for signal in RESUME_OPTIONAL_SIGNALS if signal in normalized
    ]
    missing_required_sections = [
        section
        for section in RESUME_REQUIRED_SECTIONS
        if section not in normalized
    ]
    missing_optional_signals = [
        signal
        for signal in RESUME_OPTIONAL_SIGNALS
        if signal not in normalized
    ]

    has_full_name = any(re.search(pattern, header) for pattern in name_patterns)
    has_email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", header) is not None
    has_phone = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", header) is not None

    print("[LOCAL SIGNALS]")
    print(f"- text_length: {len(normalized.strip())}")
    print(f"- required_sections_found: {required_sections_found}")
    print(f"- missing_required_sections: {missing_required_sections}")
    print(f"- optional_signals_found: {optional_signals_found}")
    print(f"- missing_optional_signals: {missing_optional_signals}")
    print(f"- has_full_name: {has_full_name}")
    print(f"- has_email: {has_email}")
    print(f"- has_phone: {has_phone}")


def run_validation_case(case: ValidationCase) -> bool:
    original_chat_openai = api_module.ChatOpenAI
    api_module.ChatOpenAI = case.fake_llm

    try:
        validate_resume_text(case.text)
    except HTTPException as exc:
        matched = case.expected_detail_contains in str(exc.detail)
        status = "PASS" if matched else "FAIL"
        print(f"\n[{status}] {case.name}")
        print(f"- status_code: {exc.status_code}")
        print(f"- detail: {exc.detail}")
        print(f"- expected_detail_contains: {case.expected_detail_contains!r}")
        return matched
    except Exception as exc:
        print(f"\n[FAIL] {case.name}")
        print(f"- unexpected_type: {type(exc).__name__}")
        print(f"- unexpected_detail: {exc}")
        return False
    finally:
        api_module.ChatOpenAI = original_chat_openai

    print(f"\n[FAIL] {case.name}")
    print("- validate_resume_text accepted text that should be rejected.")
    return False


def run_exception_branch_tests() -> bool:
    print("\n=== Synthetic Exception Branch Tests ===")

    cases = [
        ValidationCase(
            name="short document",
            text="John Smith\njohn.smith@example.com\nskills",
            expected_detail_contains="does not look like a resume",
        ),
        ValidationCase(
            name="missing required resume section",
            text=build_text(include_required=False),
            expected_detail_contains="does not look like a resume",
        ),
        ValidationCase(
            name="missing optional resume signals",
            text=build_text(include_optional=False),
            expected_detail_contains="not detailed enough",
        ),
        ValidationCase(
            name="missing full name",
            text=build_text(include_name=False),
            expected_detail_contains="Could not detect a full name",
        ),
        ValidationCase(
            name="missing contact info",
            text=build_text(include_contact=False),
            expected_detail_contains="Resume must include contact information",
        ),
        ValidationCase(
            name="llm rejects locally valid document",
            text=build_text(),
            expected_detail_contains="Uploaded document appears to be a research_paper",
            fake_llm=_FakeRejectingChatOpenAI,
        ),
    ]

    results = [run_validation_case(case) for case in cases]
    return all(results)


def run_real_pdf_fixture_test() -> bool:
    print("=== ACS Resume PDF Fixture Test ===")

    if not PDF_PATH.exists():
        print(f"[SETUP ERROR] PDF not found: {PDF_PATH}")
        return False

    print(f"Testing validate_resume_text on: {PDF_PATH}")

    try:
        text = extract_pdf_text(PDF_PATH)
    except Exception as exc:
        print(f"[EXTRACTION ERROR] {type(exc).__name__}: {exc}")
        return False

    if not text:
        print("[EXTRACTION ERROR] Extracted text is empty.")
        return False

    print("\n[EXTRACTION]")
    print(f"- pages_text_length: {len(text)}")
    print(f"- first_200_chars: {text[:200]!r}")
    print()
    print_local_signals(text)

    original_chat_openai = api_module.ChatOpenAI
    api_module.ChatOpenAI = _FakeAcceptingChatOpenAI

    try:
        validate_resume_text(text)
    except HTTPException as exc:
        print("\n[FAIL] ACS resume fixture was rejected.")
        print(f"- status_code: {exc.status_code}")
        print(f"- detail: {exc.detail}")
        return False
    except Exception as exc:
        print("\n[FAIL] Unexpected exception while validating PDF.")
        print(f"- type: {type(exc).__name__}")
        print(f"- detail: {exc}")
        return False
    finally:
        api_module.ChatOpenAI = original_chat_openai

    print("\n[PASS] ACS resume fixture was accepted as a resume.")
    return True


def main() -> int:
    pdf_ok = run_real_pdf_fixture_test()
    branches_ok = run_exception_branch_tests()

    print("\n=== Summary ===")
    print(f"- acs_resume_fixture: {'PASS' if pdf_ok else 'FAIL'}")
    print(f"- exception_branches: {'PASS' if branches_ok else 'FAIL'}")

    return 0 if pdf_ok and branches_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
