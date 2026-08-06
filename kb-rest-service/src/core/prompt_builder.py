"""
Prompt Builder - Template Method Pattern

Builds prompts for RAG queries with consistent formatting and rules.
Follows Single Responsibility Principle - only builds prompts.
"""
from abc import ABC, abstractmethod
from typing import Dict, List


class PromptTemplate(ABC):
    """Abstract base class for prompt templates"""

    @abstractmethod
    def build(self, query: str, history: List[Dict[str, str]]) -> str:
        """Build prompt from query and history"""
        pass


class RAGPromptTemplate(PromptTemplate):
    """
    Standard RAG prompt template with citation rules.

    Design: Template Method Pattern
    - Fixed structure (build method)
    - Customizable parts (override _get_custom_rules)
    """

    def __init__(self):
        self._role_section = self._build_role_section()
        self._goal_section = self._build_goal_section()
        self._response_rules = self._build_response_rules()

    def build(self, query: str, history: List[Dict[str, str]]) -> str:
        """Build complete prompt"""
        # Enhance query with detail request
        enhanced_query = self._enhance_query(query)

        # Format history
        history_text = self._format_history(history)

        # Combine sections
        prompt = f"""{self._role_section}

{self._goal_section}

---Query to be answered---
{enhanced_query}

---Conversation History---
{history_text}

{self._response_rules}"""

        return prompt

    def _enhance_query(self, query: str) -> str:
        """Add detail request if not present"""
        detail_suffix = "Please provide detailed insights from official documents and reports."
        if not query.strip().endswith(detail_suffix):
            return f"{query.strip()} {detail_suffix}"
        return query

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history"""
        if not history:
            return "No previous conversation."

        formatted = []
        for msg in history[-5:]:  # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _build_role_section(self) -> str:
        """Build role definition section"""
        return """---Role---

You are a helpful assistant responding to user queries about Knowledge Graph and Document Chunks provided in JSON format below."""

    def _build_goal_section(self) -> str:
        """Build goal section with timestamp rules"""
        return """---Goal---

Generate a concise, accurate response based on the provided Knowledge Base. Follow all Response Rules strictly. Use both the conversation history and the current query to guide your response. Do not include any information not present in the Knowledge Base or conversation history.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge.
2. When encountering conflicting relationships, consider both semantic content and timestamp.
3. Do not automatically prefer the most recent relationship—use contextual judgment.
4. For time-specific queries, prioritize temporal information in the content before considering timestamps."""

    def _build_response_rules(self) -> str:
        """Build response formatting rules"""
        return """---Response Rules---

- **Format**: Use multiple paragraphs with markdown formatting and section headings.
- **Language**: Always respond in English, regardless of the language used in the question.
- **Emphasis**: Highlight all referenced information using **bold text**.
- **Inline Citations**:
    - Cite the source **immediately after** the referenced information using square brackets (e.g., [1], [2]).
    - Every time a source is used, it must be cited—even if it has been cited earlier.

- **Reference Mapping**:
    - Assign each source file a unique reference number starting from 1, in the order of **first appearance** in the main answer.
    - Maintain a mapping between source file names and their assigned reference numbers.
    - Use this mapping consistently throughout the answer.

- **References Section**:
    - **CRITICAL: Only cite information that comes from actual document files, NOT from entity or relationship names.**
    - **Only include a "References" section if actual document sources were cited in the main answer.**
    - If no information was found or no sources were cited, do not include a "References" section.
    - When including references, list the **complete file path** (e.g., Industry/SubIndustry/filename.ext) that was cited in the main answer.
    - The file path MUST be exactly as it appears in the knowledge base storage system.
    - List them in ascending order of their citation number.
    - Each file path should appear **only once**.
    - Do not include any file name that was not cited in the main answer.
    - Ensure that every reference number used in the main answer appears exactly once in the "References" section.
    - **DO NOT cite entity names, URLs, or server names - only cite actual document file paths with valid extensions.**
    - Ignore any file whose content is generic, such as "This text file belongs to Hagerty area in Demo Instances domain" or similar boilerplate.

**Reference Format Examples:**

✅ CORRECT (actual files with extensions):
- [1] Banking/Asset Management/Portfolio_Analysis.pdf
- [2] Finance/Reports/Client_Report.docx
- [3] Insurance/Policies/Coverage_Details.txt

❌ INCORRECT (entity names, not files):
- [1] Demo Industry
- [2] ForgeX-Dev-KB.AzureWebsites.Net
- [3] Company Name
- [4] Server Configuration

- **Integrity**:
    - Only cite references that are actual document files from which the information was extracted."""


class MultiKBSummaryTemplate(PromptTemplate):
    """
    Template for summarizing multiple KB results.

    Simpler than RAG template - focuses on aggregation.
    """

    def build(self, query: str, history: List[Dict[str, str]]) -> str:
        """Build multi-KB summary prompt"""
        return f"""You are summarizing results from multiple knowledge bases.

Original Query: {query}

Your task:
1. Synthesize information from all knowledge base responses below
2. Provide a coherent, unified answer
3. Maintain all citations using the format [KB:citation] where KB is the knowledge base name
4. Remove duplicate information
5. Prioritize the most relevant and recent information

Respond in English with clear sections and citations."""


def get_prompt_builder(template_type: str = "rag") -> PromptTemplate:
    """
    Factory function for prompt builders.

    Args:
        template_type: Type of template (rag, multi_kb)

    Returns:
        PromptTemplate instance

    Raises:
        ValueError: If template_type is unknown
    """
    templates = {
        "rag": RAGPromptTemplate,
        "multi_kb": MultiKBSummaryTemplate,
    }

    template_class = templates.get(template_type)
    if not template_class:
        raise ValueError(f"Unknown template type: {template_type}")

    return template_class()
