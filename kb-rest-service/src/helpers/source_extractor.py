"""
Source extraction for RAG responses.

Sources are derived from the **retrieved chunks** — the documents the
retriever actually returned — and never from parsing the LLM answer text.
Answer parsing (`core.reference_parser`, which only recognises a markdown
`## References` heading) is far too brittle for a critical path: any answer
that formats or omits its reference list produces zero sources even though
documents were retrieved.

The LLM-cited, blob-verified sources on `RAGQueryResult.sources` are still
used, but only as *enrichment*: they contribute citation numbers and a
storage-verified container/blob path for the chunks they match. A cited
source with no matching chunk is appended so nothing is ever lost.

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
        Deduplicated sources in retrieval order, cited ones first.
    """
    # Citation/verified-path index, keyed by file name.
    cited: Dict[str, SourceRef] = {}
    for src in getattr(result, "sources", None) or []:
        blob_path = _normalize_path(getattr(src, "blob_path", ""))
        file_name = str(
            getattr(src, "download_name", "") or getattr(src, "file_name", "") or ""
        ).strip()
        if not file_name:
            file_name = blob_path.split("/")[-1]
        if not file_name:
            continue
        cited[file_name.lower()] = SourceRef(
            file_name=file_name,
            blob_path=blob_path,
            container_name=str(getattr(src, "container_name", "") or "").strip() or default_container,
            citation=str(getattr(src, "citation", "") or ""),
            download_url=str(getattr(src, "download_url", "") or ""),
        )

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
        match = cited.get(file_name.lower())
        if match:
            # Prefer the storage-verified path/container resolved for the citation.
            resolved = SourceRef(
                file_name=match.file_name,
                blob_path=match.blob_path or path,
                container_name=match.container_name or default_container,
                citation=match.citation,
                download_url=match.download_url,
            )
        else:
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

    # Cited documents that never appeared among the chunks (defensive).
    for key, ref in cited.items():
        if key not in seen_names and ref.blob_path:
            seen_names.add(key)
            sources.append(ref)

    # Cited sources first (in citation order); retrieval order is preserved
    # for the rest, since sort() is stable.
    def _citation_order(ref: SourceRef) -> tuple:
        digits = "".join(ch for ch in ref.citation if ch.isdigit())
        return (ref.citation == "", int(digits) if digits else 0)

    sources.sort(key=_citation_order)

    logger.debug(
        "Extracted sources from retrieved chunks",
        source_count=len(sources),
        cited_count=len(cited),
    )
    return sources
