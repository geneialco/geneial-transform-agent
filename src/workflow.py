import logging

# Import message types if needed for type checking (replace with actual types used)
# from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from src.config import TEAM_MEMBER_CONFIGRATIONS, TEAM_MEMBERS
from src.config.tracing import configure_langsmith_tracing
from src.graph import build_graph
from src.agents.agents import create_agents_with_umls
from langgraph.graph import StateGraph, START
from src.graph.types import State
from src.graph.nodes import (
    supervisor_node,
    coordinator_node,
    browser_node,
    reporter_node,
    planner_node,
)
import logging
from copy import deepcopy
from typing import Literal
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.types import Command
from src.llms.llm import get_llm_by_type
from src.config import TEAM_MEMBERS
from src.config.agents import AGENT_LLM_MAP
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    # Ensure logger used here matches the one used in the module
    logging.getLogger("src").setLevel(logging.DEBUG)
    # Also set the root logger if messages aren't propagating
    # logging.getLogger().setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)  # Get logger for this module

# Configure LangSmith tracing
configure_langsmith_tracing()

# Create the graph
graph = build_graph()


# --- Helper Function ---
def find_ai_response(result_dict: dict) -> object | None:
    """
    Tries to find the AI response message object or content within the result dictionary.
    Returns the message object or a pseudo-message dictionary if found, else None.
    """
    logger.debug("Attempting to find AI response in workflow result...")

    # 1. Check if the last message in 'messages' list is already the AI response
    if (
        "messages" in result_dict
        and isinstance(result_dict["messages"], list)
        and len(result_dict["messages"]) > 0
    ):
        # Assume initial message is always at index 0 for comparison
        initial_message = (
            result_dict["messages"][0] if len(result_dict["messages"]) > 0 else None
        )
        last_msg = result_dict["messages"][-1]

        # Basic check: Does it have content and is it different from the first message?
        # A more robust check would use isinstance(last_msg, AIMessage) if type is known
        # Check if it's not the initial message object instance
        if hasattr(last_msg, "content") and last_msg is not initial_message:
            logger.info(
                "Found potential AI response as the last message in 'messages' list (different object than initial)."
            )
            return last_msg  # Return the message object

    # 2. Look for specific keys often used for LLM output
    #    (Add more keys based on potential graph node names if known)
    possible_keys = ["generation", "response", "answer", "output", "llm_output"]
    for key in possible_keys:
        if key in result_dict:
            value = result_dict[key]
            content_str = None
            message_obj = None

            if isinstance(value, str):
                content_str = value
                # Create a pseudo-message if it looks like the target output
                # Use heuristic check on content
                if "SCHEMA" in content_str or "OUTPUT" in content_str:
                    message_obj = {"role": "assistant", "content": content_str}
                    logger.debug(
                        f"Found string content in key '{key}' matching SCHEMA/OUTPUT pattern."
                    )
            elif hasattr(value, "content") and isinstance(
                getattr(value, "content"), str
            ):
                # If it's an object with content (like AIMessage)
                content_str = getattr(value, "content")
                message_obj = value  # Assume it's the message object we want
                logger.debug(f"Found object with .content attribute in key '{key}'.")
                # Check heuristic in content
                if not ("SCHEMA" in content_str or "OUTPUT" in content_str):
                    logger.debug(
                        f"Content in key '{key}' doesn't match SCHEMA/OUTPUT pattern, skipping."
                    )
                    message_obj = None  # Reset if content doesn't match heuristic

            # Return the first match found using common keys + heuristic
            if message_obj:
                logger.info(
                    f"Found potential AI response via common key heuristic in result['{key}']."
                )
                return message_obj

    # 3. Last resort: Iterate all values (less reliable, check content heuristic)
    logger.debug(
        "AI response not found via common keys or messages list. Iterating all values."
    )
    for key, value in result_dict.items():
        # Skip keys already checked or known not to be the response
        if (
            key == "messages"
            or key in possible_keys
            or key in TEAM_MEMBER_CONFIGRATIONS
            or key == "TEAM_MEMBERS"
        ):
            continue

        content_str = None
        message_obj = None

        if isinstance(value, str):
            content_str = value
            if "SCHEMA" in content_str or "OUTPUT" in content_str:
                message_obj = {"role": "assistant", "content": content_str}
        elif hasattr(value, "content") and isinstance(getattr(value, "content"), str):
            content_str = getattr(value, "content")
            if "SCHEMA" in content_str or "OUTPUT" in content_str:
                message_obj = value  # Assume it's the message object

        if message_obj:
            logger.info(
                f"Found potential AI response via iteration + content heuristic in result['{key}']."
            )
            return message_obj  # Return first match found via iteration

    logger.error(
        "Failed to find any likely AI response in the workflow result dictionary."
    )
    return None


# --- End Helper Function ---


def create_custom_nodes(use_umls: bool = False, use_search: bool = False):
    """Create nodes using agents configured with optional UMLS tools and optional Tavily search."""
    research_agent_custom, coder_agent_custom, browser_agent_custom = (
        create_agents_with_umls(use_umls=use_umls, use_search=use_search)
    )

    def research_node_custom(state: State) -> Command[Literal["supervisor"]]:
        """Research node using custom-configured agent (UMLS/search as requested)."""
        logger.info("Custom Research agent starting task")
        result = research_agent_custom.invoke(state)
        logger.info("Custom Research agent completed task")
        response_content = result["messages"][-1].content
        response_content = repair_json_output(response_content)
        logger.debug(f"UMLS Research agent response: {response_content}")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response_content,
                        name="researcher",
                    )
                ]
            },
            goto="supervisor",
        )

    def code_node_custom(state: State) -> Command[Literal["supervisor"]]:
        """Coder node using custom-configured agent (UMLS/search as requested)."""
        logger.info("Custom Coder agent starting task")
        result = coder_agent_custom.invoke(state)
        logger.info("Custom Coder agent completed task")
        response_content = result["messages"][-1].content
        response_content = repair_json_output(response_content)
        logger.debug(f"UMLS Coder agent response: {response_content}")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response_content,
                        name="coder",
                    )
                ]
            },
            goto="supervisor",
        )

    return research_node_custom, code_node_custom


def build_custom_graph(use_umls: bool = False, use_search: bool = False):
    """Build and return an agent workflow graph configured with optional UMLS tools and optional Tavily search."""
    research_node_custom, code_node_custom = create_custom_nodes(
        use_umls=use_umls, use_search=use_search
    )

    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node_custom)
    builder.add_node("coder", code_node_custom)
    builder.add_node("browser", browser_node)
    builder.add_node("reporter", reporter_node)
    return builder.compile()


def run_agent_workflow(
    user_input: str,
    debug: bool = False,
    use_umls: bool = False,
    use_search: bool = False,
):
    """Run the agent workflow with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging
        use_umls: If True, enables UMLS tools for medical terminology

    Returns:
        The final state after the workflow completes, potentially with the
        AI response appended to the 'messages' list.
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()  # Ensure this enables logging effectively

    # Use a custom-configured graph if UMLS or search is requested; otherwise use the default
    if use_umls or use_search:
        workflow_graph = build_custom_graph(use_umls=use_umls, use_search=use_search)
        logger.info(
            "Using custom workflow (UMLS: %s, search: %s)", use_umls, use_search
        )
    else:
        workflow_graph = graph
        logger.info("Using standard workflow (offline, no UMLS)")

    initial_state = {
        # Constants
        "TEAM_MEMBERS": TEAM_MEMBERS,
        "TEAM_MEMBER_CONFIGRATIONS": TEAM_MEMBER_CONFIGRATIONS,
        # Runtime Variables
        # Use actual message objects if graph expects them (e.g., HumanMessage)
        # from langchain_core.messages import HumanMessage
        # "messages": [HumanMessage(content=user_input)],
        # Using dicts for now, assuming graph handles conversion if needed
        "messages": [{"role": "user", "content": user_input}],
        "deep_thinking_mode": True,
        # Only search when explicitly enabled
        "search_before_planning": bool(use_search),
    }

    logger.info(
        f"Starting workflow with initial state keys: {list(initial_state.keys())}"
    )
    # Add verbose logging for the input message content itself if needed
    logger.debug(f"Initial messages state: {initial_state['messages']}")

    result = workflow_graph.invoke(initial_state)
    # Log the type and keys of the raw result to understand its structure
    logger.info(f"Raw workflow result type: {type(result)}")
    if isinstance(result, dict):
        logger.info(f"Raw workflow result keys: {list(result.keys())}")
    logger.debug(f"Raw workflow state returned by graph.invoke: {result}")

    # --- Post-process the result to ensure AI message is in 'messages' ---
    ai_response_message = find_ai_response(result)

    if ai_response_message:
        logger.info("AI response message identified for potential addition.")
        # Ensure 'messages' key exists and is a list in the result
        if "messages" not in result or not isinstance(result["messages"], list):
            logger.warning("Initializing 'messages' list in result dictionary.")
            result["messages"] = []

        # Check if the found AI response is already the last message
        already_present = False
        if len(result["messages"]) > 0:
            last_message_in_list = result["messages"][-1]
            # Check if they are the same object instance
            if ai_response_message is last_message_in_list:
                already_present = True
                logger.info(
                    "AI response object is identical to the last object in 'messages' list."
                )
            # Fallback: Check content equality if objects differ but might be equivalent
            else:
                ai_content = None
                last_content = None
                # Extract content carefully, handling both dict and object types
                if isinstance(ai_response_message, dict):
                    ai_content = ai_response_message.get("content")
                elif hasattr(ai_response_message, "content"):
                    ai_content = getattr(ai_response_message, "content", None)

                if isinstance(last_message_in_list, dict):
                    last_content = last_message_in_list.get("content")
                elif hasattr(last_message_in_list, "content"):
                    last_content = getattr(last_message_in_list, "content", None)

                if ai_content is not None and ai_content == last_content:
                    already_present = True
                    logger.info(
                        "AI response content matches the content of the last message in 'messages' list."
                    )

        if not already_present:
            logger.info(
                "Adding found AI response to the 'messages' list in the result dictionary."
            )
            # Make sure we append a dict or object that cli.py can handle
            result["messages"].append(ai_response_message)
        else:
            logger.info(
                "AI response already present in 'messages' list. No modification needed."
            )
    else:
        # If no AI response found, log the state for debugging why
        logger.error(
            "Could not find AI response to add to messages list. Final state before return: %s",
            result,
        )
    # --- End Post-processing ---

    # Log the final state being returned, especially the messages list
    logger.info("Workflow completed.")
    if isinstance(result, dict) and "messages" in result:
        logger.debug(f"Final messages list being returned: {result['messages']}")
    else:
        logger.debug(
            f"Final result being returned (no 'messages' list or not a dict): {result}"
        )

    return result


if __name__ == "__main__":
    # Example execution (optional, for testing workflow.py directly)
    try:
        # Add a simple example input for direct execution test
        test_input = "Generate a simple SCHEMA and OUTPUT section for testing."
        print(f"Running workflow directly with input: '{test_input}'")
        output = run_agent_workflow(test_input, debug=True)
        print("\n--- Workflow Output ---")
        # Pretty print the output dictionary for readability
        import json

        try:
            # Attempt to serialize, handling potential non-serializable objects
            print(json.dumps(output, indent=2, default=str))
        except TypeError as json_error:
            print(f"Could not JSON serialize output, printing raw: {json_error}")
            print(output)
        print("--- End Workflow Output ---")
    except Exception as workflow_error:
        print(f"Error running workflow directly: {workflow_error}")
        logger.exception("Error during direct execution test.")

    # Keep the mermaid print commented unless graphviz is installed and configured
    # print(graph.get_graph().draw_mermaid())
    pass
