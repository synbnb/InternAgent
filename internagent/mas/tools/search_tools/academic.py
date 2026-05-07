"""
Academic Literature Search Tool

This module provides an aggregated academic search function for LLM agents.
It wraps the LiteratureSearch functionality from literature_search.py into
a simple tool interface suitable for tool-calling LLMs.
"""

import logging
from typing import Dict, Any, List, Optional

# Import core functionality from literature_search module
from ..literature_search import (
    LiteratureSearch,
    CitationManager,
    PaperMetadata
)

logger = logging.getLogger(__name__)


# Tool definition for LLM agents (OpenAI function calling format)
ACADEMIC_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "academic_search",
        "description": (
            "Search for academic papers across multiple sources including arXiv, "
            "Semantic Scholar, CrossRef, and CORE. Returns comprehensive paper metadata "
            "including title, authors, abstract, citations, and PDF links. "
            "Ideal for literature reviews, research surveys, and finding relevant papers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query for academic papers. Can include keywords, topics, "
                        "author names, or specific concepts. "
                        "Example: 'transformer neural networks', 'carbon capture catalysts', "
                        "'CRISPR gene editing'"
                    )
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to return (default: 10, max: 50)",
                    "default": 10
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["arxiv", "semantic_scholar", "crossref", "core"]
                    },
                    "description": (
                        "List of academic sources to search. Options: 'arxiv' (preprints), "
                        "'semantic_scholar' (comprehensive), 'crossref' (DOI database), "
                        "'core' (open access). Default: ['arxiv', 'semantic_scholar', 'crossref']"
                    ),
                    "default": ["arxiv", "semantic_scholar", "crossref"]
                },
                "include_citations": {
                    "type": "boolean",
                    "description": "Include formatted citations in the results (default: false)",
                    "default": False
                },
                "citation_format": {
                    "type": "string",
                    "enum": ["apa", "bibtex"],
                    "description": "Citation format to use if include_citations is true (default: 'apa')",
                    "default": "apa"
                }
            },
            "required": ["query"]
        }
    }
}


async def academic_search(
    query: str,
    max_results: int = 10,
    sources: Optional[List[str]] = None,
    include_citations: bool = False,
    citation_format: str = "apa",
    **kwargs
) -> Dict[str, Any]:
    """
    Search for academic papers across multiple sources.

    This function provides a unified interface for searching academic literature
    from multiple sources including arXiv, Semantic Scholar, CrossRef, and CORE.
    Results are deduplicated and formatted for easy consumption by LLMs.

    Args:
        query: Search query string
        max_results: Maximum number of papers to return (default: 10, max: 50)
        sources: List of sources to search (default: ['arxiv', 'semantic_scholar', 'crossref'])
        include_citations: Whether to include formatted citations (default: False)
        citation_format: Citation format ('apa' or 'bibtex') (default: 'apa')
        **kwargs: Additional parameters passed to the search backend

    Returns:
        Dictionary containing:
            - success: Boolean indicating if search was successful
            - query: The original search query
            - total_results: Number of papers found
            - papers: List of paper metadata dictionaries
            - sources_searched: List of sources that were searched
            - error: Error message if search failed

    Examples:
        >>> result = await academic_search("transformer neural networks", max_results=5)
        >>> result = await academic_search("CRISPR", sources=["arxiv", "semantic_scholar"])
    """
    logger.info(f"Academic search: {query}")

    # Validate inputs
    if not query or not isinstance(query, str):
        return {
            "success": False,
            "error": "Invalid query: must be a non-empty string"
        }

    # Limit max_results to prevent excessive API calls
    max_results = min(max_results, 50)

    # Set default sources if not provided
    if sources is None:
        sources = ["arxiv", "semantic_scholar", "crossref"]

    # Validate sources
    valid_sources = {"arxiv", "semantic_scholar", "crossref", "core"}
    sources = [s for s in sources if s in valid_sources]

    if not sources:
        return {
            "success": False,
            "error": f"No valid sources provided. Valid sources: {valid_sources}"
        }

    try:
        # Initialize literature search engine
        citation_manager = CitationManager()
        searcher = LiteratureSearch()

        # Perform search
        logger.info(f"Searching {len(sources)} sources: {sources}")
        papers = await searcher.search(
            query=query,
            max_results=max_results,
            sources=sources,
            **kwargs
        )

        # Format results
        formatted_papers = []
        for paper in papers:
            paper_dict = {
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "year": paper.year,
                "source": paper.source,
                "url": paper.url,
            }

            # Add optional fields if available
            if paper.doi:
                paper_dict["doi"] = paper.doi
            if paper.journal:
                paper_dict["journal"] = paper.journal
            if paper.citations is not None:
                paper_dict["citations"] = paper.citations
            if paper.pdf_url:
                paper_dict["pdf_url"] = paper.pdf_url

            # Add citation if requested
            if include_citations:
                paper_dict["citation"] = paper.to_citation(format_type=citation_format)

            formatted_papers.append(paper_dict)

        # Prepare response
        result = {
            "success": True,
            "query": query,
            "total_results": len(formatted_papers),
            "papers": formatted_papers,
            "sources_searched": sources,
            "message": f"Found {len(formatted_papers)} papers for query: '{query}'"
        }

        logger.info(f"Academic search completed: {len(formatted_papers)} papers found")
        return result

    except Exception as e:
        error_msg = f"Academic search failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "query": query,
            "error": error_msg,
            "sources_searched": sources
        }


# Additional utility function for detailed multi-source results
async def academic_search_by_source(
    query: str,
    max_results: int = 10,
    sources: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Search academic papers and return results organized by source.

    Unlike academic_search() which merges results, this function returns
    results grouped by each source, useful for comparing coverage across databases.

    Args:
        query: Search query string
        max_results: Maximum number of papers per source
        sources: List of sources to search
        **kwargs: Additional parameters

    Returns:
        Dictionary with results grouped by source name
    """
    logger.info(f"Multi-source academic search: {query}")

    if sources is None:
        sources = ["arxiv", "semantic_scholar", "crossref"]

    try:
        searcher = LiteratureSearch()
        results_dict = await searcher.multi_source_search(
            query=query,
            sources=sources,
            max_results=max_results,
            **kwargs
        )

        # Format results by source
        formatted_results = {}
        total_papers = 0

        for source, papers in results_dict.items():
            formatted_papers = [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "abstract": p.abstract,
                    "year": p.year,
                    "url": p.url,
                    "doi": p.doi,
                    "citations": p.citations
                }
                for p in papers
            ]
            formatted_results[source] = {
                "count": len(formatted_papers),
                "papers": formatted_papers
            }
            total_papers += len(formatted_papers)

        return {
            "success": True,
            "query": query,
            "total_results": total_papers,
            "results_by_source": formatted_results,
            "sources_searched": list(results_dict.keys())
        }

    except Exception as e:
        logger.error(f"Multi-source search failed: {e}", exc_info=True)
        return {
            "success": False,
            "query": query,
            "error": str(e)
        }
