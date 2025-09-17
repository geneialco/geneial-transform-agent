import logging
import os
from typing import Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import TAVILY_MAX_RESULTS
from .decorators import create_logged_tool

logger = logging.getLogger(__name__)

# Wrap the Tavily tool in a lazy factory so we don't require the API key at import time
LoggedTavilySearch = create_logged_tool(TavilySearchResults)


def get_tavily_tool() -> Optional[object]:
    """Return a Tavily tool instance if TAVILY_API_KEY is available; otherwise None.

    This avoids import-time crashes in environments (e.g., MCP) where web search is
    disabled by default or the API key is not configured.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.info("Tavily search disabled: missing TAVILY_API_KEY in environment.")
        return None
    try:
        return LoggedTavilySearch(name="tavily_search", max_results=TAVILY_MAX_RESULTS)
    except Exception as exc:
        # Be defensive: if underlying library still raises due to config, degrade gracefully
        logger.warning(
            "Tavily search could not be initialized (%s). Disabling search.", exc
        )
        return None
