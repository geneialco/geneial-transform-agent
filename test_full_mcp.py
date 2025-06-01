#!/usr/bin/env python3
"""
Test all MCP methods with the stdio server.
"""

import subprocess
import json
import sys
import time


def test_mcp_server():
    """Test the MCP server with all methods."""

    # Start the server process
    process = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server_stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
    )

    try:
        # Test requests
        requests = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                },
            },
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "resources/list"},
            {"jsonrpc": "2.0", "id": 4, "method": "prompts/list"},
        ]

        # Send each request and get response
        for request in requests:
            request_json = json.dumps(request)
            print(f"Sending: {request_json}")

            # Send request
            process.stdin.write(request_json + "\n")
            process.stdin.flush()

            # Read response
            response_line = process.stdout.readline()
            if response_line:
                print(f"Received: {response_line.strip()}")
                try:
                    response = json.loads(response_line.strip())
                    print(f"Parsed response: {json.dumps(response, indent=2)}")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse response: {e}")
            else:
                print("No response received")

            print("-" * 50)
            time.sleep(0.5)  # Small delay between requests

    finally:
        # Terminate the process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        # Print any stderr output
        stderr_output = process.stderr.read()
        if stderr_output:
            print("Server stderr:")
            print(stderr_output)


if __name__ == "__main__":
    test_mcp_server()
