"""
Graph Context Parser

Parses LightRAG's graph context format into structured JSON.
Handles entities, relationships, and document chunks.
"""
import json
import re
from typing import Any, Dict, List


def parse_graph_context(content: str) -> Dict[str, Any]:
    """
    Parse LightRAG graph context into structured JSON.

    Args:
        content: Raw context string from LightRAG containing graph data

    Returns:
        Structured dict with:
        - entities: List of entity objects
        - relationships: List of relationship objects
        - document_chunks: List of document chunk objects
        - raw_text: Original content for reference
    """
    result = {
        "entities": [],
        "relationships": [],
        "document_chunks": [],
        "metadata": {
            "has_entities": False,
            "has_relationships": False,
            "has_documents": False,
            "total_entities": 0,
            "total_relationships": 0,
            "total_chunks": 0
        },
        "raw_text": content
    }

    try:
        # Extract entities section
        entity_section = re.search(
            r'Knowledge Graph Data \(Entity\):\s*```json\s*(.*?)\s*```',
            content,
            re.DOTALL | re.IGNORECASE
        )
        if entity_section:
            entity_text = entity_section.group(1).strip()
            entities = _parse_json_lines(entity_text)
            result["entities"] = entities
            result["metadata"]["has_entities"] = len(entities) > 0
            result["metadata"]["total_entities"] = len(entities)

        # Extract relationships section
        relationship_section = re.search(
            r'Knowledge Graph Data \(Relationship\):\s*```json\s*(.*?)\s*```',
            content,
            re.DOTALL | re.IGNORECASE
        )
        if relationship_section:
            rel_text = relationship_section.group(1).strip()
            relationships = _parse_json_lines(rel_text)
            result["relationships"] = relationships
            result["metadata"]["has_relationships"] = len(relationships) > 0
            result["metadata"]["total_relationships"] = len(relationships)

        # Extract document chunks section
        doc_section = re.search(
            r'Document Chunks.*?```json\s*(.*?)\s*```',
            content,
            re.DOTALL | re.IGNORECASE
        )
        if doc_section:
            doc_text = doc_section.group(1).strip()
            chunks = _parse_json_lines(doc_text)
            result["document_chunks"] = chunks
            result["metadata"]["has_documents"] = len(chunks) > 0
            result["metadata"]["total_chunks"] = len(chunks)

    except Exception as e:
        # Log error but don't fail - return partial results
        result["metadata"]["parse_error"] = str(e)

    return result


def _parse_json_lines(text: str) -> List[Dict[str, Any]]:
    """
    Parse multiple JSON objects separated by newlines.

    Args:
        text: Multi-line text with one JSON object per line

    Returns:
        List of parsed JSON objects
    """
    objects = []

    if not text or not text.strip():
        return objects

    # Try parsing as a single JSON array first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Parse line by line
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                objects.append(obj)
            elif isinstance(obj, list):
                objects.extend(obj)
        except json.JSONDecodeError:
            # Skip invalid lines
            continue

    return objects


def format_chunk_with_graph_data(
    chunk_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format a chunk to include structured graph data if present.

    Args:
        chunk_dict: Raw chunk dictionary

    Returns:
        Enhanced chunk with:
        - content_type: "graph" or "document"
        - graph_data: Structured entities/relationships (if graph chunk)
        - summary: Human-readable summary of content
    """
    content = chunk_dict.get("content", "")

    # Check if this is a graph chunk
    if "Knowledge Graph Data" in content:
        parsed = parse_graph_context(content)

        # Build summary
        summary_parts = []
        if parsed["metadata"]["total_entities"] > 0:
            summary_parts.append(f"{parsed['metadata']['total_entities']} entities")
        if parsed["metadata"]["total_relationships"] > 0:
            summary_parts.append(f"{parsed['metadata']['total_relationships']} relationships")
        if parsed["metadata"]["total_chunks"] > 0:
            summary_parts.append(f"{parsed['metadata']['total_chunks']} document chunks")

        summary = f"Graph data: {', '.join(summary_parts)}" if summary_parts else "Empty graph data"

        # Enhance chunk with structured data
        enhanced = chunk_dict.copy()
        enhanced["content_type"] = "graph"
        enhanced["graph_data"] = {
            "entities": parsed["entities"],
            "relationships": parsed["relationships"],
            "document_chunks": parsed["document_chunks"],
            "metadata": parsed["metadata"]
        }
        enhanced["summary"] = summary
        enhanced["original_content"] = content

        # Replace content with structured summary
        enhanced["content"] = summary

        return enhanced

    # Regular document chunk
    enhanced = chunk_dict.copy()
    enhanced["content_type"] = "document"
    enhanced["summary"] = content[:200] + "..." if len(content) > 200 else content

    return enhanced


def extract_entities(chunk_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract entities from a chunk if it contains graph data.

    Args:
        chunk_dict: Chunk dictionary (potentially with graph_data)

    Returns:
        List of entity objects
    """
    if "graph_data" in chunk_dict:
        return chunk_dict["graph_data"].get("entities", [])

    # Try parsing from content
    content = chunk_dict.get("content", "")
    if "Knowledge Graph Data" in content:
        parsed = parse_graph_context(content)
        return parsed["entities"]

    return []


def extract_relationships(chunk_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract relationships from a chunk if it contains graph data.

    Args:
        chunk_dict: Chunk dictionary (potentially with graph_data)

    Returns:
        List of relationship objects
    """
    if "graph_data" in chunk_dict:
        return chunk_dict["graph_data"].get("relationships", [])

    # Try parsing from content
    content = chunk_dict.get("content", "")
    if "Knowledge Graph Data" in content:
        parsed = parse_graph_context(content)
        return parsed["relationships"]

    return []


def build_graph_summary(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a summary of graph data across multiple chunks.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Summary with aggregated entity/relationship counts
    """
    total_entities = 0
    total_relationships = 0
    total_doc_chunks = 0
    unique_entities = set()
    unique_relationship_types = set()

    for chunk in chunks:
        if chunk.get("content_type") == "graph":
            graph_data = chunk.get("graph_data", {})

            entities = graph_data.get("entities", [])
            total_entities += len(entities)
            for entity in entities:
                if isinstance(entity, dict) and "entity" in entity:
                    unique_entities.add(entity["entity"])

            relationships = graph_data.get("relationships", [])
            total_relationships += len(relationships)
            for rel in relationships:
                if isinstance(rel, dict):
                    rel_key = f"{rel.get('entity1')}-{rel.get('entity2')}"
                    unique_relationship_types.add(rel_key)

            doc_chunks = graph_data.get("document_chunks", [])
            total_doc_chunks += len(doc_chunks)

    return {
        "total_graph_chunks": sum(1 for c in chunks if c.get("content_type") == "graph"),
        "total_document_chunks": sum(1 for c in chunks if c.get("content_type") == "document"),
        "total_entities": total_entities,
        "unique_entities": len(unique_entities),
        "total_relationships": total_relationships,
        "unique_relationships": len(unique_relationship_types),
        "embedded_doc_chunks": total_doc_chunks
    }
