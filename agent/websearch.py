import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

from tavily import TavilyClient

from .models import TutorAgentState

logger = logging.getLogger(__name__)


@dataclass
class WebSearchTool:
    """
    Wrapper around Tavily Search API.
    Provides a clean abstraction for web search operations.
    """

    api_key: str = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", "")
    )

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set.")

        self._client = TavilyClient(api_key=self.api_key)

    def search(self, query: str) -> Dict[str, Any]:
        """
        Execute a web search query.

        Args:
            query: User search query.

        Returns:
            Tavily API response.
        """
        try:
            return self._client.search(query=query)

        except Exception as exc:
            logger.exception("Tavily search failed for query: %s", query)
            raise RuntimeError("Web search operation failed.") from exc


def websearch_node(state: TutorAgentState) -> str:
    """
    Perform a web search and update the agent state.

    Args:
        state: Current agent state.

    Returns:
        Search results as a string representation.
    """
    search_tool = WebSearchTool()

    results = search_tool.search(state.user_query)

    state.web_search_results = results

    return str(results)