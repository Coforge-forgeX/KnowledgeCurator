"""
Reference Parser - Single Responsibility Principle

Parses document references from RAG responses.
Validates citation format and extracts file paths.
"""
import re
from typing import List

from src.core.logging import get_logger
from src.models.rag_models import DocumentReference

logger = get_logger(__name__)


class ReferenceParser:
    """
    Extracts and validates document references from RAG responses.

    Design:
    - Single Responsibility: Only parses references
    - No side effects: Pure function approach
    - Clear error handling: Invalid references logged, not raised
    """

    # Pattern to match references like [1] path/to/file.ext
    REFERENCE_PATTERN = re.compile(
        r'\[(\d+)\]\s+([A-Za-z0-9_\-/\\]+\.(?:pdf|docx?|txt|xlsx?|pptx?|csv|json|xml|html?))',
        re.IGNORECASE
    )

    # Invalid patterns to exclude (entity names, URLs, etc.)
    INVALID_PATTERNS = [
        r'https?://',  # URLs
        r'www\.',  # Web addresses
        r'\.com|\.net|\.org',  # Domain extensions
        r'^[A-Z][a-z]+\s',  # Starts with capital (likely entity name)
    ]

    def parse(self, response: str) -> List[DocumentReference]:
        """
        Parse document references from RAG response.

        Args:
            response: RAG response text

        Returns:
            List of validated DocumentReference objects

        Example:
            >>> parser = ReferenceParser()
            >>> refs = parser.parse(response)
            >>> for ref in refs:
            ...     print(f"{ref.citation_number}: {ref.file_path}")
        """
        if not response:
            logger.warning("Empty response provided to reference parser")
            return []

        # Extract references section if present
        references_section = self._extract_references_section(response)
        if not references_section:
            logger.info("No references section found in response")
            return []

        # Find all matches
        matches = self.REFERENCE_PATTERN.findall(references_section)
        if not matches:
            logger.warning("No valid references found in references section")
            return []

        # Convert to DocumentReference objects with validation
        references = []
        for number, file_path in matches:
            if self._is_valid_reference(file_path):
                try:
                    ref = DocumentReference(
                        citation_number=f"[{number}]",
                        file_path=file_path.strip(),
                        file_name=self._extract_filename(file_path)
                    )
                    references.append(ref)
                    logger.debug(f"Parsed reference: {ref.citation_number} -> {ref.file_path}")
                except ValueError as e:
                    logger.warning(f"Invalid reference: {e}")
            else:
                logger.debug(f"Filtered invalid reference: {file_path}")

        logger.info(f"Parsed {len(references)} valid references from response")
        return references

    def _extract_references_section(self, response: str) -> str:
        """
        Extract the ### References section from response.

        Returns empty string if no references section found.
        """
        # Case-insensitive match for References header
        pattern = re.compile(r'###?\s*References\s*\n(.*)', re.IGNORECASE | re.DOTALL)
        match = pattern.search(response)

        if match:
            return match.group(1)
        return ""

    def _is_valid_reference(self, file_path: str) -> bool:
        """
        Validate that file_path is a real file, not an entity name.

        Returns:
            True if valid file path, False if entity name/URL/invalid
        """
        # Check against invalid patterns
        for pattern in self.INVALID_PATTERNS:
            if re.search(pattern, file_path):
                return False

        # Must have a file extension
        if '.' not in file_path.split('/')[-1]:
            return False

        # Must not be just a domain/company name
        if file_path.count('/') == 0 and ' ' in file_path:
            return False

        return True

    def _extract_filename(self, file_path: str) -> str:
        """Extract filename from path"""
        # Handle both forward and back slashes
        path_normalized = file_path.replace('\\', '/')
        return path_normalized.split('/')[-1]


class ResponseCleaner:
    """
    Removes reference sections from RAG responses.

    Design: Single Responsibility - only cleans responses
    """

    @staticmethod
    def remove_references_section(response: str) -> str:
        """
        Remove the ### References section from response.

        Args:
            response: Original response with references

        Returns:
            Cleaned response without references section

        Example:
            >>> cleaner = ResponseCleaner()
            >>> clean = cleaner.remove_references_section(response)
        """
        if not response:
            return ""

        # Remove everything from ### References onwards (case-insensitive)
        pattern = re.compile(r'(?i)\n*##+\s*References[\s\S]*$')
        cleaned = pattern.sub('', response).strip()

        logger.debug(
            f"Removed references section: "
            f"original={len(response)} chars, cleaned={len(cleaned)} chars"
        )

        return cleaned


def parse_references(response: str) -> List[DocumentReference]:
    """
    Convenience function for parsing references.

    Args:
        response: RAG response text

    Returns:
        List of DocumentReference objects
    """
    parser = ReferenceParser()
    return parser.parse(response)


def clean_response(response: str) -> str:
    """
    Convenience function for cleaning response.

    Args:
        response: RAG response with references

    Returns:
        Cleaned response without references section
    """
    cleaner = ResponseCleaner()
    return cleaner.remove_references_section(response)
