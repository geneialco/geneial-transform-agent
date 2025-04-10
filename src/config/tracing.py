import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def configure_langsmith_tracing() -> None:
    """Configure LangSmith tracing based on environment variables.

    This function will:
    1. Check if LANGCHAIN_API_KEY or LANGSMITH_API_KEY is set
    2. If key exists, enable tracing
    3. If no key exists, disable tracing
    4. Handle any configuration errors gracefully
    """
    # Check for API key
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")

    if not api_key:
        # No API key found, disable tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled (no API key found)")
        return

    try:
        # Enable tracing if we have an API key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Set project name if provided
        project = os.getenv("LANGSMITH_PROJECT", "default")
        os.environ["LANGSMITH_PROJECT"] = project

        logger.info(f"LangSmith tracing enabled (project: {project})")

    except Exception as e:
        # If anything goes wrong, disable tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.warning(f"Failed to configure LangSmith tracing: {str(e)}")
        logger.warning("Tracing has been disabled")
