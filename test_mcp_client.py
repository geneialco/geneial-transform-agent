#!/usr/bin/env python3
"""
Simple test client for the MCP server.
This demonstrates how to interact with the MCP server using JSON-RPC.
"""

import requests
import json
import sys

# Server URL
SERVER_URL = "http://localhost:8001"


def send_json_rpc_request(method, params=None, request_id=1):
    """Send a JSON-RPC request to the server."""
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method}

    if params is not None:
        payload["params"] = params

    response = requests.post(
        SERVER_URL, json=payload, headers={"Content-Type": "application/json"}
    )

    return response.json()


def test_initialize():
    """Test the initialize method."""
    print("Testing initialize...")
    result = send_json_rpc_request("initialize")
    print(json.dumps(result, indent=2))
    print()
    return result


def test_list_tools():
    """Test the tools/list method."""
    print("Testing tools/list...")
    result = send_json_rpc_request("tools/list")
    print(json.dumps(result, indent=2))
    print()
    return result


def test_call_tool(tool_name, arguments):
    """Test the tools/call method."""
    print(f"Testing tools/call with {tool_name}...")
    result = send_json_rpc_request(
        "tools/call", params={"name": tool_name, "arguments": arguments}
    )
    print(json.dumps(result, indent=2))
    print()
    return result


def main():
    """Run all tests."""
    print("MCP Server Test Client")
    print("=" * 50)

    # Test initialize
    init_result = test_initialize()
    if "error" in init_result and init_result["error"] is not None:
        print("Error initializing:", init_result["error"])
        sys.exit(1)

    # Test list tools
    tools_result = test_list_tools()
    if "error" in tools_result and tools_result["error"] is not None:
        print("Error listing tools:", tools_result["error"])
        sys.exit(1)

    # Test calling a tool
    test_input = """
    Patient Information:
    - Name: John Doe
    - Age: 45
    - Diagnosis: Type 2 Diabetes
    - Symptoms: Increased thirst, frequent urination
    """

    # Test transform to LinkML
    linkml_result = test_call_tool(
        "transform_to_linkml",
        {
            "input_content": test_input,
            "user_prompt": "Convert this patient information to LinkML format",
            "debug": False,
        },
    )

    # Test transform to Phenopackets JSON
    phenopackets_result = test_call_tool(
        "transform_to_phenopackets_json",
        {
            "input_content": test_input,
            "user_prompt": "Convert this patient information to Phenopackets JSON format",
            "debug": False,
        },
    )

    print("All tests completed!")


if __name__ == "__main__":
    main()
