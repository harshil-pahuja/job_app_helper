"""Debug education degree extraction for the Full Stack React AI Engineer role.

Usage:
    .\.venv\Scripts\python.exe debug_full_stack_react_education.py

This script checks what degree levels are extracted from:
    tests/job_postings/Full Stack React AI Engineer.txt
"""

from __future__ import annotations

from pathlib import Path

from backend.nlp_processor import extract_education, extract_education_with_llm


JOB_POSTING_PATH = Path("tests/job_postings/Full Stack React AI Engineer.txt")
EDUCATION_KEYWORDS = [
    "degree",
    "bachelor",
    "bachelor's",
    "master",
    "master's",
    "phd",
    "ph.d",
    "doctorate",
    "associate",
    "diploma",
    "ged",
    "high school",
    "education",
    "university",
    "college",
]


def print_keyword_lines(text: str) -> None:
    print("\n[EDUCATION KEYWORD LINES]")
    found = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in EDUCATION_KEYWORDS):
            found = True
            print(f"{line_number:>3}: {line}")

    if not found:
        print("- No education-related keyword lines found.")


def main() -> int:
    if not JOB_POSTING_PATH.exists():
        print(f"[SETUP ERROR] Job posting not found: {JOB_POSTING_PATH}")
        return 1

    text = JOB_POSTING_PATH.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        print("[SETUP ERROR] Job posting file is empty.")
        return 1

    print(f"Testing education extraction on: {JOB_POSTING_PATH}")
    print(f"- text_length: {len(text)}")
    print(f"- first_line: {text.splitlines()[0] if text.splitlines() else ''}")

    print_keyword_lines(text)

    regex_degrees = extract_education(text)
    print("\n[REGEX DEGREE EXTRACTION]")
    print(f"- degrees: {regex_degrees}")

    llm_degrees = extract_education_with_llm(text)
    print("\n[LLM DEGREE EXTRACTION]")
    print(f"- degrees: {llm_degrees}")

    if not llm_degrees:
        print("\n[RESULT] No degree level was extracted from this role.")
    else:
        print("\n[RESULT] Degree level(s) extracted.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
