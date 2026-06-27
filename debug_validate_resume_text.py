"""Debug utility for testing backend.api.validate_resume_text on a resume file.

Usage:
    python debug_validate_resume_text.py --resume "C:/path/to/Harshil_Pahuja_Resume.pdf"
    python debug_validate_resume_text.py --resume "C:/path/to/Harshil_Pahuja_Resume.pdf" --mock-llm

If --resume is omitted, the script tries to auto-find a file that contains
"harshil" or "pahuja" in common project locations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from docx import Document
from fastapi import HTTPException
from pypdf import PdfReader

import backend.api as api_module
from backend.api import (
    RESUME_OPTIONAL_SIGNALS,
    RESUME_REQUIRED_SECTIONS,
    name_patterns,
    re,
    validate_resume_text,
)


class _FakeResumeValidationResponse:
    content = """
{
  "is_resume": true,
  "document_type": "resume",
  "confidence": 0.95,
  "reason": "Mocked validation response for local debugging."
}
"""


class _FakeChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt):
        return _FakeResumeValidationResponse()


def _iter_candidate_paths() -> Iterable[Path]:
    roots = [Path("."), Path("tests"), Path("tests/resumes")]
    exts = {".pdf", ".doc", ".docx", ".txt"}

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            name = path.name.lower()
            if "harshil" in name or "pahuja" in name:
                yield path


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages).strip()

    if suffix in {".doc", ".docx"}:
        doc = Document(str(path))
        return "\n".join((p.text or "") for p in doc.paragraphs).strip()

    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _print_local_signals(text: str) -> None:
    normalized = text.lower()
    header = text[:500]

    required_matches = [s for s in RESUME_REQUIRED_SECTIONS if s in normalized]
    optional_matches = [s for s in RESUME_OPTIONAL_SIGNALS if s in normalized]

    has_full_name = any(re.search(pattern, header) for pattern in name_patterns)
    has_email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", header) is not None
    has_phone = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", header) is not None

    print("\n[LOCAL SIGNALS]")
    print(f"- text_length: {len(normalized.strip())}")
    print(f"- required_sections_found: {required_matches}")
    print(f"- optional_signals_found: {optional_matches}")
    print(f"- has_full_name: {has_full_name}")
    print(f"- has_email: {has_email}")
    print(f"- has_phone: {has_phone}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug validate_resume_text against a resume document."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to resume file (.pdf, .doc, .docx, .txt).",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Mock the ChatOpenAI validator so this script tests local validation without calling OpenAI.",
    )
    args = parser.parse_args()

    if args.mock_llm:
        api_module.ChatOpenAI = _FakeChatOpenAI
        print("Using mocked ChatOpenAI resume validation response.")

    if args.resume:
        resume_path = Path(args.resume).expanduser()
    else:
        resume_path = next(_iter_candidate_paths(), None)
        if resume_path is None:
            print("Could not auto-find Harshil resume.")
            print("Run with: python debug_validate_resume_text.py --resume \"<path_to_resume>\"")
            return 1

    if not resume_path.exists():
        print(f"Resume path not found: {resume_path}")
        return 1

    print(f"Testing validate_resume_text on: {resume_path}")

    try:
        text = _extract_text(resume_path)
    except Exception as exc:
        print(f"Failed to extract text from file: {exc}")
        return 1

    if not text:
        print("Extracted text is empty.")
        return 1

    _print_local_signals(text)

    try:
        validate_resume_text(text)
        print("\n[RESULT] PASS: Document validated as resume.")
        return 0
    except HTTPException as exc:
        print("\n[RESULT] FAIL: validate_resume_text raised HTTPException")
        print(f"- status_code: {exc.status_code}")
        print(f"- detail: {exc.detail}")
        if exc.__cause__ is not None:
            print(f"- cause_type: {type(exc.__cause__).__name__}")
            print(f"- cause_detail: {exc.__cause__}")
        return 2
    except Exception as exc:
        print("\n[RESULT] FAIL: Unexpected exception")
        print(f"- type: {type(exc).__name__}")
        print(f"- detail: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
