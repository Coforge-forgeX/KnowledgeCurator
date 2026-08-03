"""
Centralized file validation for document upload

Only file types with complete processing logic are supported.
"""

# Supported file extensions
# Only include types that have complete text extraction and indexing logic
SUPPORTED_FILE_EXTENSIONS = [
    ".pdf",   # Azure Document Intelligence
    ".docx",  # python-docx
    ".doc",   # python-docx with conversion
    ".txt",   # Direct UTF-8 decode
    ".md",    # Markdown files (text processor)
]

# File extension to MIME content type mapping
FILE_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


def validate_file_extension(file_name: str) -> str:
    """
    Validate that file has a supported extension.

    Centralized validation ensures consistency across all upload endpoints.
    Only allows file types that have complete processing logic implemented.

    Args:
        file_name: File name to validate

    Returns:
        Validated file name (unchanged)

    Raises:
        ValueError: If file extension is not supported

    Example:
        >>> validate_file_extension("document.pdf")
        'document.pdf'
        >>> validate_file_extension("spreadsheet.xlsx")
        ValueError: File extension not supported...
    """
    if not file_name or "." not in file_name:
        raise ValueError("File name must have an extension")

    file_lower = file_name.lower()

    if not any(file_lower.endswith(ext) for ext in SUPPORTED_FILE_EXTENSIONS):
        supported_list = ", ".join(SUPPORTED_FILE_EXTENSIONS)
        raise ValueError(
            f"File extension not supported. "
            f"Supported types: {supported_list}. "
            f"File: {file_name}"
        )

    return file_name


def get_content_type(file_name: str) -> str:
    """
    Get MIME content type for a file based on its extension.

    Args:
        file_name: File name

    Returns:
        MIME content type string

    Example:
        >>> get_content_type("document.pdf")
        'application/pdf'
    """
    import os
    ext = os.path.splitext(file_name)[1].lower()
    return FILE_CONTENT_TYPES.get(ext, "application/octet-stream")
