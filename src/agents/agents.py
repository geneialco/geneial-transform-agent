from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.tools import (
    bash_tool,
    browser_tool,
    crawl_tool,
    python_repl_tool,
    get_tavily_tool,
    search_medical_terms_tool,
    validate_medical_terminology_tool,
    search_cuis_tool,
    get_cui_info_tool,
    calculate_cui_similarity_tool,
    enhance_phenotype_data_tool,
)

from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP


# Create agents using configured LLM types
def create_agent(agent_type: str, tools: list, prompt_template: str):
    """Factory function to create agents with consistent configuration."""
    return create_react_agent(
        get_llm_by_type(AGENT_LLM_MAP[agent_type]),
        tools=tools,
        prompt=lambda state: apply_prompt_template(prompt_template, state),
    )


def get_umls_tools():
    """Get the list of UMLS tools."""
    return [
        search_medical_terms_tool,
        validate_medical_terminology_tool,
        search_cuis_tool,
        get_cui_info_tool,
        calculate_cui_similarity_tool,
        enhance_phenotype_data_tool,
    ]


def create_agents_with_umls(use_umls: bool = False, use_search: bool = False):
    """Create agents with optional UMLS tools and optional web search.

    Args:
        use_umls: Include UMLS-related tools when True.
        use_search: Include Tavily search tool when True. Defaults to False (offline).
    """
    umls_tools = get_umls_tools() if use_umls else []

    research_tools = [crawl_tool] + umls_tools
    if use_search:
        tavily_tool_instance = get_tavily_tool()
        if tavily_tool_instance is not None:
            research_tools = [tavily_tool_instance] + research_tools
        else:
            # No API key or initialization failed; keep offline behavior
            pass

    research_agent = create_agent(
        "researcher",
        research_tools,
        "researcher",
    )
    coder_agent = create_agent(
        "coder",
        [python_repl_tool, bash_tool] + umls_tools,
        "coder",
    )
    browser_agent = create_agent("browser", [browser_tool], "browser")

    return research_agent, coder_agent, browser_agent


# Create default agents (without UMLS, offline by default)
research_agent, coder_agent, browser_agent = create_agents_with_umls(
    use_umls=False, use_search=False
)
