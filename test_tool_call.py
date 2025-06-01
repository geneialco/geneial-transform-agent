#!/usr/bin/env python3
"""
Test tool call functionality with the fixed server.
"""

import subprocess
import json
import sys
import time


def test_tool_call():
    """Test a tool call with sample data."""

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
        # Test data
        test_data = """ID,Age,Sex,Ethnicity,Race,BMI,SmokingStatus,SystolicBP,DiastolicBP,TotalCholesterol,MedicalHistory,SNP_rs1234,SNP_rs2345,SNP_rs3456,AncestryPC1,AncestryPC2
NWD000001,65,M,Non-Hispanic,White,28.4,Former,142,90,230,Coronary Artery Disease; Hypertension; Hyperlipidemia,AG,CT,AT,0.12,-0.03
NWD000002,45,F,Hispanic,White,22.1,Never,118,75,180,None,AA,TT,AA,0.04,0.01"""

        # Send initialize first
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }

        print("Step 1: Initialize server")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()

        response_line = process.stdout.readline()
        print(f"Initialize response: {response_line.strip()}")

        # Send tool call
        tool_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "transform_to_linkml",
                "arguments": {
                    "input_content": test_data,
                    "user_prompt": "Transform this TOPMed data to LinkML format",
                },
            },
        }

        print("\nStep 2: Send tool call")
        request_json = json.dumps(tool_request)
        print(f"Sending tool call: {request_json[:200]}...")

        process.stdin.write(request_json + "\n")
        process.stdin.flush()

        # Read response
        print("Step 3: Waiting for tool response...")
        response_line = process.stdout.readline()
        if response_line:
            print(f"Tool response received: {len(response_line)} characters")
            try:
                response = json.loads(response_line.strip())
                print(f"Response structure: {response.keys()}")

                if "result" in response:
                    result = response["result"]
                    if "content" in result:
                        print(f"Content items: {len(result['content'])}")
                        for i, content in enumerate(result["content"]):
                            print(f"Content {i}: {content.get('text', '')[:200]}...")
                    print(f"Is error: {result.get('isError', False)}")
                else:
                    print(
                        f"Error in response: {response.get('error', 'Unknown error')}"
                    )

            except json.JSONDecodeError as e:
                print(f"Failed to parse response: {e}")
                print(f"Raw response: {response_line.strip()[:500]}...")
        else:
            print("No response received!")

    finally:
        # Terminate the process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    test_tool_call()
