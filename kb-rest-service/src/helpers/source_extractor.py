"""
Source extraction for RAG responses.

Blob-verified citations on `RAGQueryResult.sources` are authoritative when
present, keeping each response source aligned with a citation in the final
answer. Retrieved chunks are used only as a fallback when the model emitted
no usable reference mapping.

Shared by `query_rag` and `message_gpt` so both endpoints report the same
sources for the same query.
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from src.core.logging import get_logger

logger = get_logger(__name__)

# Extensions we accept as real documents. Chunk `file_path` occasionally holds
# a KB/graph label rather than a file, which must not surface as a source.
_DOCUMENT_EXTENSIONS = {
    "pdf", "doc", "docx", "txt", "xls", "xlsx", "ppt", "pptx",
    "csv", "json", "xml", "html", "htm", "md", "rtf",
}


@dataclass
class SourceRef:
    """One source document, storage-addressable and optionally cited."""

    file_name: str
    blob_path: str
    container_name: str = ""
    citation: str = ""
    download_url: str = ""


def _normalize_path(raw: Any) -> str:
    return str(raw or "").strip().replace("\\", "/").strip("/")


def _is_document_path(path: str) -> bool:
    tail = path.split("/")[-1]
    if "." not in tail:
        return False
    return tail.rsplit(".", 1)[-1].lower() in _DOCUMENT_EXTENSIONS


def _chunk_file_path(chunk: Any) -> str:
    """Read a chunk's file path from either a RetrievedChunk or a serialized dict."""
    if isinstance(chunk, dict):
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        return _normalize_path(chunk.get("file_path") or metadata.get("file_path") or chunk.get("source"))

    metadata = getattr(chunk, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    return _normalize_path(metadata.get("file_path") or getattr(chunk, "source", ""))


def extract_sources(
    result: Any,
    extra_paths: Optional[Iterable[Any]] = None,
    default_container: str = "",
) -> List[SourceRef]:
    """
    Build the source list for a `RAGQueryResult`.

    Args:
        result: RAGQueryResult (or anything exposing `retrieved_chunks` / `sources`).
        extra_paths: Additional file paths to consider (e.g. graph-derived
            chunks assembled by the caller), each a str, RetrievedChunk or dict.
        default_container: Storage container to report for chunk-derived
            sources; overridden by the container of a matching cited source.

    Returns:
        Cited sources in citation order when available; otherwise deduplicated
        retrieved sources in retrieval order as a fallback.
    """
    cited: List[SourceRef] = []
    for src in getattr(result, "sources", None) or []:
        blob_path = _normalize_path(getattr(src, "blob_path", ""))
        file_name = str(
            getattr(src, "download_name", "") or getattr(src, "file_name", "") or ""
        ).strip()
        if not file_name:
            file_name = blob_path.split("/")[-1]
        if not file_name:
            continue
        cited.append(
            SourceRef(
                file_name=file_name,
                blob_path=blob_path,
                container_name=str(getattr(src, "container_name", "") or "").strip() or default_container,
                citation=str(getattr(src, "citation", "") or ""),
                download_url=str(getattr(src, "download_url", "") or ""),
            )
        )

    if cited:
        def _citation_order(ref: SourceRef) -> tuple:
            digits = "".join(ch for ch in ref.citation if ch.isdigit())
            return (int(digits) if digits else 0, ref.file_name.lower())

        unique_cited = {
            (ref.container_name.lower(), ref.blob_path.lower(), ref.citation): ref
            for ref in cited
        }
        return sorted(unique_cited.values(), key=_citation_order)

    candidates: List[Any] = list(getattr(result, "retrieved_chunks", None) or [])
    if extra_paths:
        candidates.extend(extra_paths)

    sources: List[SourceRef] = []
    seen_paths: set = set()
    seen_names: set = set()

    for candidate in candidates:
        path = candidate if isinstance(candidate, str) else _chunk_file_path(candidate)
        path = _normalize_path(path)
        if not path or not _is_document_path(path) or path in seen_paths:
            continue

        file_name = path.split("/")[-1]
        resolved = SourceRef(
            file_name=file_name,
            blob_path=path,
            container_name=default_container,
        )

        if resolved.file_name.lower() in seen_names:
            continue

        seen_paths.add(path)
        seen_names.add(resolved.file_name.lower())
        sources.append(resolved)

    logger.debug(
        "Extracted sources from retrieved chunks",
        source_count=len(sources),
        cited_count=len(cited),
    )
    return sources
