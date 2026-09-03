from unittest.mock import AsyncMock, patch

import pytest

from src.helpers.source_extractor import extract_sources
from src.models.rag_models import EnrichedSource, KnowledgeBase, QueryContext, RAGQueryResult, RetrievedChunk
from src.models.query_rag_models import SourceReferenceModel
from src.services.query_rag_executor import _resolve_source_containers
from src.services.query_strategies import MultiKBStrategy, SingleKBStrategy
from src.services.rag_query_service import SourceEnricher


def test_extract_sources_returns_only_sources_cited_by_answer():
    result = RAGQueryResult(
        answer="Claim [1]",
        sources=[
            EnrichedSource(
                file_name="[1] cited.docx",
                download_url="",
                container_name="workspace-files",
                blob_path="Other/Demo/1067/cited.docx",
                download_name="cited.docx",
                citation="[1]",
            )
        ],
        retrieved_chunks=[
            RetrievedChunk("1", "used", 1.0, "Other/Demo/1067/cited.docx"),
            RetrievedChunk("2", "retrieved only", 0.5, "Other/Demo/1067/unused.pdf"),
        ],
    )

    sources = extract_sources(result, default_container="workspace-files")

    assert [(source.file_name, source.citation) for source in sources] == [
        ("cited.docx", "[1]")
    ]


def test_extract_sources_orders_multiple_sources_by_citation():
    result = RAGQueryResult(
        answer="First [1], second [2]",
        sources=[
            EnrichedSource("[2] second.pdf", "", "workspace-files", "second.pdf", "second.pdf", "[2]"),
            EnrichedSource("[1] first.pdf", "", "workspace-files", "first.pdf", "first.pdf", "[1]"),
        ],
    )

    sources = extract_sources(result)

    assert [(source.file_name, source.citation) for source in sources] == [
        ("first.pdf", "[1]"),
        ("second.pdf", "[2]"),
    ]


def test_extract_sources_keeps_same_filename_from_different_containers():
    result = RAGQueryResult(
        answer="Workspace [1], KG [2]",
        sources=[
            EnrichedSource("[1] guide.pdf", "", "workspace-files", "workspace/guide.pdf", "guide.pdf", "[1]"),
            EnrichedSource("[2] guide.pdf", "", "aksknowledgecurator", "kg/guide.pdf", "guide.pdf", "[2]"),
        ],
    )

    sources = extract_sources(result)

    assert [(source.container_name, source.citation) for source in sources] == [
        ("workspace-files", "[1]"),
        ("aksknowledgecurator", "[2]"),
    ]


@pytest.mark.asyncio
async def test_single_kb_recovers_citation_from_retrieved_chunk_without_references():
    strategy = SingleKBStrategy()
    strategy._query_lightrag = AsyncMock(return_value={
        "answer": "Luxury guidance [1]",
        "retrieved_chunks": [
            {
                "chunk_id": "voyage-1",
                "content": "Luxury guidance",
                "score": 1.0,
                "source": "Other/Demo/Virgin_Voyages_Best_Practice.docx",
                "metadata": {
                    "file_path": "Other/Demo/Virgin_Voyages_Best_Practice.docx"
                },
            }
        ],
    })

    result = await strategy.execute(
        QueryContext(query="Summarize domain knowledge", workspace_id=1066, role_id=1),
        [KnowledgeBase(domain="Other", name="Demo")],
    )

    references = result.metadata["raw_references"]
    assert [(ref.citation_number, ref.file_path) for ref in references] == [
        ("[1]", "Other/Demo/Virgin_Voyages_Best_Practice.docx")
    ]


@pytest.mark.asyncio
async def test_source_enricher_uses_resolved_workspace_container():
    storage = AsyncMock()
    storage.container_name = "workspace-files"
    storage.provider_name = "azure"
    storage.blob_exists.return_value = True
    storage.generate_download_url.return_value = "https://example.test/file"
    reference = type(
        "Reference",
        (),
        {"file_path": "cited.docx", "file_name": "cited.docx", "citation_number": "[1]"},
    )()

    with patch("src.services.rag_query_service.get_storage_adapter", return_value=storage) as factory:
        source = await SourceEnricher().enrich_reference(
            reference,
            "Other",
            "Demo/onezerosixseven",
            workspace_id=1067,
            role_id=1,
            container_name="workspace-files",
        )

    factory.assert_called_once_with(container_override="workspace-files")
    assert source is not None
    assert source.container_name == "workspace-files"


@pytest.mark.asyncio
async def test_source_enricher_falls_back_to_kg_container_for_mixed_query():
    workspace_storage = AsyncMock()
    workspace_storage.container_name = "workspace-files"
    workspace_storage.provider_name = "azure"
    workspace_storage.blob_exists.return_value = False

    kg_storage = AsyncMock()
    kg_storage.container_name = "aksknowledgecurator"
    kg_storage.provider_name = "azure"
    kg_storage.blob_exists.return_value = True
    kg_storage.generate_download_url.return_value = "https://example.test/kg-file"

    def storage_for_container(*, container_override):
        return {
            "workspace-files": workspace_storage,
            "aksknowledgecurator": kg_storage,
        }[container_override]

    reference = type(
        "Reference",
        (),
        {
            "file_path": "Other/Demo Instances/Virgin/cited.docx",
            "file_name": "cited.docx",
            "citation_number": "[2]",
        },
    )()

    with patch(
        "src.services.rag_query_service.get_storage_adapter",
        side_effect=storage_for_container,
    ):
        source = await SourceEnricher().enrich_reference(
            reference,
            "Other",
            "Demo Instances/onezerosixseven",
            workspace_id=1067,
            role_id=1,
            container_name="workspace-files",
            fallback_container_names=["aksknowledgecurator"],
        )

    assert source is not None
    assert source.container_name == "aksknowledgecurator"
    workspace_storage.blob_exists.assert_awaited_once()
    kg_storage.blob_exists.assert_awaited_once()
    kg_storage.generate_download_url.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_rag_resolves_each_source_container_before_signing():
    workspace_storage = AsyncMock()
    workspace_storage.blob_exists.return_value = False
    kg_storage = AsyncMock()
    kg_storage.blob_exists.return_value = True

    def storage_for_container(*, container_override):
        return {
            "workspace": workspace_storage,
            "kg-files": kg_storage,
        }[container_override]

    source = SourceReferenceModel(
        file_id="",
        file_name="rules.pdf",
        container_name="workspace",
        blob_path="Other/Demo Instances/1067/rules.pdf",
        provider="azure",
    )

    with (
        patch(
            "src.services.query_rag_executor.settings.storage.WORKSPACE_CONTAINER_NAME",
            "workspace",
        ),
        patch(
            "src.services.query_rag_executor.settings.storage.STORAGE_CONTAINER_NAME",
            "kg-files",
        ),
        patch(
            "src.services.query_rag_executor.get_storage_adapter",
            side_effect=storage_for_container,
        ),
    ):
        resolved = await _resolve_source_containers([source])

    assert len(resolved) == 1
    assert resolved[0].container_name == "kg-files"
    workspace_storage.blob_exists.assert_awaited_once_with(source.blob_path)
    kg_storage.blob_exists.assert_awaited_once_with(source.blob_path)


def test_source_enricher_does_not_duplicate_complete_blob_path():
    enricher = SourceEnricher()

    path = enricher._build_blob_path(
        "Other",
        "Demo Instances/onezerosixseven",
        "Other/Demo Instances/1067/cited.docx",
        workspace_id=1067,
        role_id=1,
    )

    assert path == "Other/Demo Instances/1067/cited.docx"


def test_multi_kb_builds_global_citations_from_retrieved_chunks_without_references():
    strategy = MultiKBStrategy()
    kb_results = {
        "Other/Demo/onezerosixseven": {
            "answer": "Study guidance [1]",
            "_retrieved_chunks": [
                {
                    "chunk_id": "study-1",
                    "content": "Study guidance",
                    "score": 1.0,
                    "source": "Other/Demo/1067/study.pdf",
                    "metadata": {"file_path": "Other/Demo/1067/study.pdf"},
                }
            ],
        },
        "Other/Demo/Virgin": {
            "answer": "Voyage guidance [1]",
            "_retrieved_chunks": [
                {
                    "chunk_id": "voyage-1",
                    "content": "Voyage guidance",
                    "score": 1.0,
                    "source": "Other/Demo/Virgin/Virgin_Voyages_Best_Practices.docx",
                    "metadata": {
                        "file_path": "Other/Demo/Virgin/Virgin_Voyages_Best_Practices.docx"
                    },
                }
            ],
        },
    }

    references, citation_maps = strategy._build_global_references(kb_results)

    assert [(ref.citation_number, ref.file_name) for ref in references] == [
        ("[1]", "study.pdf"),
        ("[2]", "Virgin_Voyages_Best_Practices.docx"),
    ]
    assert citation_maps == {
        "Other/Demo/onezerosixseven": {"[1]": "[1]"},
        "Other/Demo/Virgin": {"[1]": "[2]"},
    }