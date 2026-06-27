"""
RAG (Retrieval-Augmented Generation) system for resume and job description processing.
Handles embeddings, vector storage, and document retrieval.
"""
import os
import logging

from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_community.document_loaders import TextLoader  # type: ignore
from langchain_community.vectorstores import Chroma  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool  # type: ignore

logger = logging.getLogger(__name__)
DEBUG_PRIVACY_LOGS = os.getenv("DEBUG_PRIVACY_LOGS", "").lower() == "true"


class RAGSystem:
    def __init__(
        self,
        collection_name: str = "job_app_helper_collection",
        embedding_backend: str = "auto",
        persist_directory: str | None = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = self._build_embeddings(embedding_backend)
        self.vectorstore = None

    def _build_embeddings(self, embedding_backend: str):
        """Build embedding model. Auto mode prefers OpenAI if API key exists."""
        backend = embedding_backend.lower().strip()
        if backend == "auto":
            backend = "openai" if os.getenv("OPENAI_API_KEY") else "huggingface"

        if backend == "openai":
            return OpenAIEmbeddings()

        if backend == "huggingface":
            # Local embedding model for offline/low-cost testing.
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        raise ValueError("embedding_backend must be one of: auto, openai, huggingface")

    def load_and_process_document(
        self,
        file_path: str,
        chunk_size: int = 150,
        overlap: int = 0,
    ):
        """Load a document, split it into chunks with section context."""
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        logger.debug(
            "Loaded %d document(s), total length: %d chars",
            len(documents),
            len(documents[0].page_content),
        )

        chunks = []
        chunk_index = 0
        for document in documents:
            text_chunks = self._split_text_into_chunks_with_sections(
                document.page_content, 
                chunk_size=chunk_size, 
                overlap=overlap
            )
            for chunk_text, section in text_chunks:
                doc = Document(
                    page_content=chunk_text, 
                    metadata={
                        "chunk_index": chunk_index,
                        "section": section
                    }
                )
                chunks.append(doc)
                chunk_index += 1

        return chunks
    
    def _split_text_into_chunks_with_sections(self, text: str, chunk_size: int = 150, overlap: int = 0):
        """Split text into chunks while tracking which resume section each chunk belongs to."""
        import re
        
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        # Broad section header list to handle diverse resume formats
        section_headers = [
            'Education', 'Academic Background', 'Academic Qualifications',
            'Skills', 'Technical Skills', 'Core Competencies', 'Competencies',
            'Technical Expertise', 'Areas of Expertise', 'Key Skills',
            'Work Experience', 'Work Experiences', 'Professional Experience',
            'Experience', 'Employment', 'Employment History', 'Work History',
            'Career History', 'Relevant Experience',
            'Projects', 'Personal Projects', 'Academic Projects', 'Key Projects',
            'Leadership', 'Leadership Experience', 'Activities',
            'Certifications', 'Licenses', 'Certificates',
            'Awards', 'Honors', 'Achievements', 'Accomplishments',
            'Summary', 'Professional Summary', 'Executive Summary',
            'Objective', 'Career Objective', 'Profile',
            'Contact', 'References', 'Publications', 'Volunteer',
        ]
        
        # Find section boundaries: (position_in_text, section_name)
        section_positions = []
        for header in section_headers:
            # Match header on its own line (with optional surrounding whitespace/punctuation)
            for match in re.finditer(rf'(?:^|\n)\s*{re.escape(header)}\s*(?:\n|$)', text, re.IGNORECASE):
                section_positions.append((match.start(), header))
        
        # Sort by position
        section_positions.sort(key=lambda x: x[0])
        
        # Create a mapping: character position -> section name
        current_section = "Resume"
        section_map = {}
        section_idx = 0
        
        for char_pos in range(len(text)):
            while section_idx < len(section_positions) and char_pos >= section_positions[section_idx][0]:
                current_section = section_positions[section_idx][1]
                section_idx += 1
            section_map[char_pos] = current_section
        
        # Split text into chunks at word boundaries (not mid-word/mid-sentence)
        step = chunk_size - overlap
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Extend to next word boundary to avoid cutting mid-word
            if end < text_len and text[end] not in (' ', '\n', '\t'):
                boundary = text.find(' ', end)
                newline = text.find('\n', end)
                # Pick the nearest whitespace boundary within 50 chars
                candidates = [b for b in [boundary, newline] if 0 < b <= end + 50]
                if candidates:
                    end = min(candidates)
            
            chunk_text = text[start:end]
            section = section_map.get(start, "Resume")
            
            chunk_with_header = f"[{section}]\n{chunk_text}"
            chunks.append((chunk_with_header, section))
            
            if end >= text_len:
                break
            start += step
        
        return chunks

    def create_vectorstore(self, chunks):
        """Create a vector store from the document chunks."""
        # Filter out empty or whitespace-only chunks
        valid_chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
        logger.debug("Valid chunks: %d", len(valid_chunks))

        if not valid_chunks:
            raise ValueError("No valid chunks to create vectorstore (all chunks are empty)")
        
        self.vectorstore = Chroma.from_documents(
            valid_chunks,
            self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5):
        """Retrieve relevant document chunks based on a query, sorted by document order."""
        if not self.vectorstore:
            raise ValueError("Vector store not created. Please load and process a document first.")
        
        relevant_chunks = self.vectorstore.similarity_search(query, k=top_k)
        # Sort by chunk_index to return in original document order
        relevant_chunks.sort(key=lambda doc: doc.metadata.get("chunk_index", 0))
        return relevant_chunks

    def ingest_file(self, file_path: str):
        """Convenience method for tests: load, split, and index a file in one call."""
        chunks = self.load_and_process_document(file_path)
        self.create_vectorstore(chunks)

def create_retrieve_resume_tool(rag_instance: RAGSystem):
    """Factory function to create a retrieval tool bound to a specific RAG instance.
    
    Args:
        rag_instance: The RAGSystem instance to retrieve from
        
    Returns:
        A LangChain tool function the agent can call
    """
    @tool
    def retrieve_resume_context(query: str) -> str:
        """Retrieve relevant resume sections for the given query."""
        chunks = rag_instance.retrieve_relevant_chunks(query, top_k=5)
        if not chunks:
            return "No relevant information found in the resume."
        return "\n\n".join([chunk.page_content for chunk in chunks])
    
    return retrieve_resume_context


def main():
    """Test skill extraction from resume to check if C++ is detected."""
    from backend.nlp_processor import extract_skills
    import sys
    from pathlib import Path
    
    # Use provided file or prompt user
    if len(sys.argv) > 1:
        resume_path = sys.argv[1]
    else:
        # Check for common resume locations
        possible_paths = [
            "resume.txt",
            "tests/resume.txt",
            "tests/sample_resume.txt",
        ]
        resume_path = None
        for path in possible_paths:
            if Path(path).exists():
                resume_path = path
                break
        
        if not resume_path:
            logger.info("No resume found. Usage: python -m backend.rag_system <path_to_resume>")
            return
    
    # Read resume. Do not log the path by default because filenames can contain
    # names or other personal details.
    logger.info("Loading resume for local RAG debug run")
    with open(resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()
    
    # Extract skills
    logger.info("Extracting skills from resume")
    skills = extract_skills(resume_text)
    
    # Display only aggregate output by default. Extracted skills may reveal
    # private resume contents, so list them only when privacy debug logging is on.
    logger.info("Extracted %d skills from resume", len(skills))
    if DEBUG_PRIVACY_LOGS:
        logger.info("="*70)
        logger.info("EXTRACTED SKILLS (%d total):", len(skills))
        logger.info("="*70)
        for i, skill in enumerate(skills, 1):
            logger.info("%2d. %s", i, skill)
    
    # Check for C++
    logger.info("\n" + "="*70)
    cpp_found = any(s.lower() == "c++" for s in skills)
    
    if cpp_found:
        logger.info("✓ SUCCESS: C++ was extracted from resume")
    else:
        logger.warning("C++ was NOT extracted from resume")
        # Check if there are any C-related skills
        c_related = [s for s in skills if "c" in s.lower() and ("+" in s or "sharp" in s)]
        if DEBUG_PRIVACY_LOGS and c_related:
            logger.info("Similar skills found: %s", ", ".join(c_related))
    
    logger.info("="*70)


if __name__ == "__main__":
    main()
