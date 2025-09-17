#!/usr/bin/env python3
"""
Stdio-based MCP server for Claude Desktop compatibility.
This wraps the existing MCP server functionality to work over stdio instead of HTTP.
"""

import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional
import os
import io
import signal
import time

# Redirect stdout during imports to prevent any output
original_stdout = sys.stdout
sys.stdout = io.StringIO()

# Disable all logging from imported modules before importing them
logging.getLogger("browser_use").setLevel(logging.CRITICAL)
logging.getLogger("src.config.tracing").setLevel(logging.CRITICAL)
logging.getLogger("langsmith").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.WARNING)

# Set environment variables to disable logging
os.environ["BROWSER_USE_LOGGING_LEVEL"] = "CRITICAL"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

try:
    # Import the handler functions from the HTTP server
    from .mcp_server import (
        handle_initialize,
        handle_list_tools,
        handle_call_tool,
        handle_list_resources,
        handle_list_prompts,
        METHOD_HANDLERS,
        JSONRPCRequest,
        JSONRPCResponse,
    )
finally:
    # Restore stdout after imports
    sys.stdout = original_stdout

# Configure logging to stderr so it doesn't interfere with stdio communication
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True,  # Force reconfiguration
)
logger = logging.getLogger(__name__)

# Ensure no output goes to stdout from logging
for handler in logging.root.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        logging.root.removeHandler(handler)

# Global flag for graceful shutdown
shutdown_requested = False

# Whether to respond using Content-Length framing (set dynamically when a framed
# request is received). Defaults to False to preserve newline-delimited behavior
# for simple CLI tests and backward compatibility.
use_content_length_response = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info(f"Received signal {signum}, requesting shutdown")


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def process_json_rpc_request(
    request_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Process a single JSON-RPC request and return a response."""
    try:
        # Validate JSON-RPC request
        rpc_request = JSONRPCRequest(**request_data)

        logger.info(
            f"Processing method: {rpc_request.method} with id: {rpc_request.id}"
        )

        # Check if method is supported
        if rpc_request.method not in METHOD_HANDLERS:
            if rpc_request.id is not None:
                return JSONRPCResponse(
                    id=rpc_request.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {rpc_request.method}",
                    },
                ).model_dump()
            # For notifications with unsupported methods, return None
            return None

        # Handle the method
        handler = METHOD_HANDLERS[rpc_request.method]
        result = await handler(rpc_request.params)

        # If it's a notification (no id), don't return a response
        if rpc_request.id is None:
            return None

        # Return successful response
        response_data = result.model_dump() if hasattr(result, "model_dump") else result
        response = JSONRPCResponse(id=rpc_request.id, result=response_data).model_dump(
            exclude_none=True
        )

        logger.info(f"Sending response for method: {rpc_request.method}")
        return response

    except Exception as e:
        logger.error(f"Error processing JSON-RPC request: {str(e)}")
        # Only return error response if it's not a notification
        if "id" in request_data and request_data["id"] is not None:
            return JSONRPCResponse(
                id=request_data.get("id"),
                error={"code": -32603, "message": f"Internal error: {str(e)}"},
            ).model_dump(exclude_none=True)
        return None


async def read_json_rpc_message() -> Optional[Dict[str, Any]]:
    """Read a JSON-RPC message from stdin."""
    try:
        global use_content_length_response
        loop = asyncio.get_event_loop()

        # Read the first line (bytes) which may be a Content-Length header or raw JSON
        first_line_bytes = await loop.run_in_executor(None, sys.stdin.buffer.readline)

        # Check if we got EOF
        if not first_line_bytes:
            logger.info("Got EOF from stdin")
            return None

        first_line_stripped = first_line_bytes.strip().decode("utf-8", errors="ignore")

        # Skip empty lines
        if not first_line_stripped:
            logger.info("Skipping empty line")
            return {"skip": True}

        logger.info(
            f"Received line: {first_line_stripped[:100]}..."
            if len(first_line_stripped) > 100
            else f"Received line: {first_line_stripped}"
        )

        # LSP-style framing: Content-Length header
        if first_line_stripped.lower().startswith("content-length:"):
            try:
                _, value = first_line_stripped.split(":", 1)
                content_length = int(value.strip())
            except Exception as e:
                logger.error(
                    f"Invalid Content-Length header: {first_line_stripped} ({e})"
                )
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }

            # Consume remaining headers until a blank line
            while True:
                header_line_bytes = await loop.run_in_executor(
                    None, sys.stdin.buffer.readline
                )
                if header_line_bytes is None or header_line_bytes == b"":
                    logger.info("Got EOF while reading headers")
                    return None
                if (
                    header_line_bytes in (b"\r\n", b"\n")
                    or header_line_bytes.strip() == b""
                ):
                    break

            # Read the body of exactly content_length bytes
            try:
                body_bytes = await loop.run_in_executor(
                    None, sys.stdin.buffer.read, content_length
                )
            except Exception as e:
                logger.error(f"Failed to read body bytes: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }

            if not body_bytes or len(body_bytes) < content_length:
                logger.error(
                    f"Incomplete body read: expected {content_length}, got {0 if not body_bytes else len(body_bytes)}"
                )
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }

            try:
                body_text = body_bytes.decode("utf-8")
            except Exception as e:
                logger.error(f"Failed to decode body as UTF-8: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }

            try:
                obj = json.loads(body_text)
                # Mark that we should respond using Content-Length framing
                use_content_length_response = True
                return obj
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse JSON body: {e}, body: {body_text[:100]}..."
                )
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }

        # Fallback: newline-delimited raw JSON
        try:
            obj = json.loads(first_line_stripped)
            # For raw input, ensure we respond with raw newline JSON for compatibility
            use_content_length_response = False
            return obj
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}, line: {first_line_stripped}")
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
    except Exception as e:
        logger.error(f"Error reading message: {e}")
        # Don't exit on read errors, just skip
        return {"skip": True}


def write_json_rpc_message(message: Dict[str, Any]):
    """Write a JSON-RPC message to stdout."""
    try:
        json_str = json.dumps(message, separators=(",", ":"))

        # If the client used Content-Length framing, respond in kind
        if use_content_length_response:
            body_bytes = json_str.encode("utf-8")
            header = f"Content-Length: {len(body_bytes)}\r\n\r\n".encode("ascii")
            logger.info(
                f"Writing framed response (Content-Length {len(body_bytes)}): {json_str[:200]}..."
                if len(json_str) > 200
                else f"Writing framed response (Content-Length {len(body_bytes)}): {json_str}"
            )
            sys.stdout.buffer.write(header)
            sys.stdout.buffer.write(body_bytes)
            sys.stdout.flush()
            logger.info("Framed response written and flushed")
        else:
            # Fallback: write newline-delimited JSON
            logger.info(
                f"Writing response: {json_str[:200]}..."
                if len(json_str) > 200
                else f"Writing response: {json_str}"
            )
            sys.stdout.write(json_str + "\n")
            sys.stdout.flush()
            logger.info("Response written and flushed")
    except Exception as e:
        logger.error(f"Error writing message: {e}")


async def main():
    """Main loop for the stdio-based MCP server."""
    global shutdown_requested
    logger.info("Starting stdio-based MCP server")

    # Keep track of consecutive errors
    consecutive_errors = 0
    max_consecutive_errors = 5

    while not shutdown_requested:
        try:
            # Read a message from stdin
            logger.info("Waiting for next message from stdin...")
            message = await read_json_rpc_message()

            # Check for special skip marker
            if message and message.get("skip"):
                logger.info("Skipping message processing")
                consecutive_errors = 0  # Reset error count on any activity
                await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                continue

            # Check for EOF
            if message is None:
                logger.info("Received EOF from stdin, shutting down")
                break

            # Reset error counter on successful read
            consecutive_errors = 0

            # If it's a parse error, send it immediately
            if "error" in message and message.get("error", {}).get("code") == -32700:
                write_json_rpc_message(message)
                continue

            # Process the request
            logger.info("Processing JSON-RPC request")
            response = await process_json_rpc_request(message)

            # Write the response if there is one (not a notification)
            if response is not None:
                write_json_rpc_message(response)
            else:
                logger.info("No response to send (notification)")

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down")
            break
        except Exception as e:
            consecutive_errors += 1
            logger.error(
                f"Unexpected error in main loop: {e} (error {consecutive_errors}/{max_consecutive_errors})"
            )

            # If too many consecutive errors, exit
            if consecutive_errors >= max_consecutive_errors:
                logger.error("Too many consecutive errors, shutting down")
                break

            # Send an internal error response
            try:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                }
                write_json_rpc_message(error_response)
            except:
                pass  # If we can't even send error response, just continue

            # Small delay before retrying
            await asyncio.sleep(1)

    logger.info("MCP server shutting down")


if __name__ == "__main__":
    # Run the main loop
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
