#!/usr/bin/env python3
"""
Test script for the stdio-based MCP server.
This simulates how Claude Desktop would communicate with the server.
"""

import subprocess
import json
import sys
import time
import threading


def read_stderr(process):
    """Read stderr in a separate thread."""
    for line in process.stderr:
        print(f"[STDERR] {line.decode().strip()}", file=sys.stderr)


def send_request(process, request):
    """Send a JSON-RPC request to the server and get the response."""
    # Send the request
    request_str = json.dumps(request) + "\n"
    process.stdin.write(request_str.encode())
    process.stdin.flush()

    # Give the server a moment to process
    time.sleep(0.5)

    # Read the response
    response_line = process.stdout.readline().decode().strip()
    if response_line:
        try:
            return json.loads(response_line)
        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {response_line}")
            print(f"Error: {e}")
            return None
    else:
        print("No response received")
        return None


def main():
    """Test the stdio MCP server."""
    print("Starting stdio MCP server test...")

    # Start the server process
    process = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server_stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,  # Use binary mode
    )

    # Start stderr reader thread
    stderr_thread = threading.Thread(target=read_stderr, args=(process,))
    stderr_thread.daemon = True
    stderr_thread.start()

    # Give the server time to start
    time.sleep(1)

    try:
        # Test 1: Initialize
        print("\n1. Testing initialize...")
        response = send_request(
            process, {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        )
        if response:
            print(json.dumps(response, indent=2))
        else:
            print("Failed to get initialize response")
            return

        # Test 2: List tools
        print("\n2. Testing tools/list...")
        response = send_request(
            process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        )
        if response and "result" in response:
            print(f"Found {len(response['result']['tools'])} tools")
        else:
            print("Failed to get tools list")
            return

        # Test 3: Call a tool
        print("\n3. Testing tools/call...")
        test_data = """ID,Age,Sex,Ethnicity,Race,BMI
NWD000001,65,M,Non-Hispanic,White,28.4
NWD000002,45,F,Hispanic,White,22.1"""

        response = send_request(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "transform_to_linkml",
                    "arguments": {
                        "input_content": test_data,
                        "user_prompt": "Convert this TOPMed data to LinkML",
                        "debug": False,
                    },
                },
            },
        )

        if response and "result" in response:
            print("Tool call successful!")
            for content in response["result"]["content"]:
                print(f"\nContent type: {content['type']}")
                print(f"First 200 chars: {content['text'][:200]}...")
        else:
            print("Tool call failed:", response)

        print("\nAll tests completed!")

    finally:
        # Clean up
        process.terminate()
        process.wait()


if __name__ == "__main__":
    main()
