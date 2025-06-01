from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal
import os
import json
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import uuid

# Import the workflow function from the existing module
from .workflow import run_agent_workflow
from .cli import get_system_prompt, split_schema_and_output, validate_output

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Geneial Langmanus Transform MCP Server",
    description="MCP Server for transforming data from TOPMed format to LinkML, Phenopackets JSON or Phenopackets CSV)",
)

# Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "10"))
PHENOPACKET_TOOLS_JAR_PATH = os.getenv(
    "PHENOPACKET_TOOLS_JAR_PATH", "/Applications/phenopacket-tools-cli-1.0.0-RC3.jar"
)

# MCP Protocol Version
MCP_PROTOCOL_VERSION = "2024-11-05"


# JSON-RPC Models
class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int, None] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int, None]
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class JSONRPCNotification(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None


# MCP Models
class ServerInfo(BaseModel):
    name: str = "Geneial Langmanus Transform MCP Server"
    version: str = "0.2.0"


class InitializeResult(BaseModel):
    protocolVersion: str = MCP_PROTOCOL_VERSION
    capabilities: Dict[str, Any] = Field(default_factory=lambda: {"tools": {}})
    serverInfo: ServerInfo = Field(default_factory=ServerInfo)


class Tool(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]


class ListToolsResult(BaseModel):
    tools: List[Tool]


class CallToolResult(BaseModel):
    content: List[Dict[str, Any]]
    isError: bool = False


# Tool definitions
TOOLS = [
    Tool(
        name="transform_to_linkml",
        description="Transform input data from TOPMed format to LinkML format",
        inputSchema={
            "type": "object",
            "properties": {
                "input_content": {
                    "type": "string",
                    "description": "The input data to transform from TOPMed format",
                },
                "user_prompt": {
                    "type": "string",
                    "description": "Additional instructions for the transformation",
                    "default": "Please process the following content and transform it according to the specified rules:",
                },
                "validate": {
                    "type": "boolean",
                    "description": "Whether to validate the output",
                    "default": True,
                },
                "debug": {
                    "type": "boolean",
                    "description": "Enable debug mode",
                    "default": False,
                },
            },
            "required": ["input_content"],
        },
    ),
    Tool(
        name="transform_to_phenopackets_json",
        description="Transform input data from TOPMed format to Phenopackets JSON format",
        inputSchema={
            "type": "object",
            "properties": {
                "input_content": {
                    "type": "string",
                    "description": "The input data to transform from TOPMed format",
                },
                "user_prompt": {
                    "type": "string",
                    "description": "Additional instructions for the transformation",
                    "default": "Please process the following content and transform it according to the specified rules:",
                },
                "validate": {
                    "type": "boolean",
                    "description": "Whether to validate the output",
                    "default": True,
                },
                "debug": {
                    "type": "boolean",
                    "description": "Enable debug mode",
                    "default": False,
                },
            },
            "required": ["input_content"],
        },
    ),
    Tool(
        name="transform_to_phenopackets_csv",
        description="Transform input data from TOPMed format to Phenopackets CSV format",
        inputSchema={
            "type": "object",
            "properties": {
                "input_content": {
                    "type": "string",
                    "description": "The input data to transform from TOPMed format",
                },
                "user_prompt": {
                    "type": "string",
                    "description": "Additional instructions for the transformation",
                    "default": "Please process the following content and transform it according to the specified rules:",
                },
                "validate": {
                    "type": "boolean",
                    "description": "Whether to validate the output",
                    "default": True,
                },
                "debug": {
                    "type": "boolean",
                    "description": "Enable debug mode",
                    "default": False,
                },
            },
            "required": ["input_content"],
        },
    ),
]


# Helper functions
async def handle_initialize(params: Optional[Dict[str, Any]]) -> InitializeResult:
    """Handle the initialize method."""
    logger.info("Initializing MCP server")
    # In the future, params might contain client capabilities
    return InitializeResult()


async def handle_list_tools(params: Optional[Dict[str, Any]]) -> ListToolsResult:
    """Handle the tools/list method."""
    logger.info("Listing available tools")
    return ListToolsResult(tools=TOOLS)


async def handle_list_resources(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Handle the resources/list method."""
    logger.info("Listing available resources")
    # Currently no resources are implemented
    return {"resources": []}


async def handle_list_prompts(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Handle the prompts/list method."""
    logger.info("Listing available prompts")
    # Currently no prompts are implemented
    return {"prompts": []}


async def handle_call_tool(params: Optional[Dict[str, Any]]) -> CallToolResult:
    """Handle the tools/call method."""
    if not params or "name" not in params:
        raise ValueError("Tool name is required")

    tool_name = params["name"]
    arguments = params.get("arguments", {})

    logger.info(f"Calling tool: {tool_name} with arguments: {arguments}")

    try:
        # Map tool names to output formats
        format_mapping = {
            "transform_to_linkml": "linkml",
            "transform_to_phenopackets_json": "phenopackets-json",
            "transform_to_phenopackets_csv": "phenopackets-csv",
        }

        if tool_name not in format_mapping:
            raise ValueError(f"Unknown tool: {tool_name}")

        output_format = format_mapping[tool_name]

        # Extract arguments
        input_content = arguments.get("input_content", "")
        user_prompt = arguments.get(
            "user_prompt",
            "Please process the following content and transform it according to the specified rules:",
        )
        validate = arguments.get("validate", True)
        debug = arguments.get("debug", False)

        # Get the appropriate system prompt
        system_prompt = get_system_prompt(output_format)

        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}\n\n{input_content}"

        if debug:
            logger.debug(f"Full prompt:\n{full_prompt}")

        # Run the workflow
        result = run_agent_workflow(full_prompt, debug=debug)

        # Split result into schema and output
        schema_content, output_content = split_schema_and_output(result)

        if not output_content:
            raise ValueError("No output was generated in the response")

        # Prepare response content
        content = []

        # Run validation if requested
        if validate:
            logger.info("Running validation...")
            try:
                (
                    corrected_schema_content,
                    corrected_output_content,
                    validation_successful,
                    validation_message,
                ) = validate_output(
                    schema_content,
                    output_content,
                    output_format,
                    full_prompt,
                    debug=debug,
                    max_retries=MAX_RETRIES,
                )

                if validation_successful:
                    # Use the potentially corrected content
                    schema_content = corrected_schema_content
                    output_content = corrected_output_content

                    # Add validation success message
                    content.append(
                        {
                            "type": "text",
                            "text": f"✅ Validation successful: {validation_message}",
                        }
                    )
                else:
                    # Add validation failure message but still return the content
                    content.append(
                        {
                            "type": "text",
                            "text": f"⚠️ Validation failed: {validation_message}",
                        }
                    )
                    logger.warning(
                        f"Validation failed but returning content anyway: {validation_message}"
                    )

            except Exception as validation_error:
                # If validation itself fails, log the error but continue
                validation_error_msg = (
                    f"Validation process failed: {str(validation_error)}"
                )
                logger.error(validation_error_msg)
                content.append(
                    {
                        "type": "text",
                        "text": f"❌ Validation error: {validation_error_msg}",
                    }
                )
        else:
            content.append({"type": "text", "text": "ℹ️ Validation was not requested"})

        # Add output content
        content.append({"type": "text", "text": output_content})

        # Add schema content if available
        if schema_content:
            content.append({"type": "text", "text": f"Schema:\n{schema_content}"})

        return CallToolResult(content=content)

    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {str(e)}")
        return CallToolResult(
            content=[{"type": "text", "text": f"Error: {str(e)}"}], isError=True
        )


# JSON-RPC method handlers
METHOD_HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_list_tools,
    "tools/call": handle_call_tool,
    "resources/list": handle_list_resources,
    "prompts/list": handle_list_prompts,
}


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


@app.post("/")
async def handle_json_rpc(request: Request):
    """Main JSON-RPC endpoint."""
    try:
        # Get request body
        body = await request.body()

        # Parse JSON
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return Response(
                content=json.dumps(
                    JSONRPCResponse(
                        id=None, error={"code": -32700, "message": "Parse error"}
                    ).model_dump()
                ),
                media_type="application/json",
            )

        # Handle batch requests
        if isinstance(data, list):
            responses = []
            for request_data in data:
                response = await process_json_rpc_request(request_data)
                if response is not None:
                    responses.append(response)

            # If no responses (all were notifications), return nothing
            if not responses:
                return Response(status_code=204)

            return Response(
                content=json.dumps(responses), media_type="application/json"
            )
        else:
            # Single request
            response = await process_json_rpc_request(data)

            # If it's a notification, return 204 No Content
            if response is None:
                return Response(status_code=204)

            return Response(content=json.dumps(response), media_type="application/json")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return Response(
            content=json.dumps(
                JSONRPCResponse(
                    id=None,
                    error={"code": -32603, "message": f"Internal error: {str(e)}"},
                ).model_dump()
            ),
            media_type="application/json",
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Geneial Langmanus Transform MCP Server",
        "protocol": "MCP",
        "version": MCP_PROTOCOL_VERSION,
    }


# SSE endpoint for server-initiated events (optional, for future use)
@app.get("/events")
async def server_sent_events():
    """Server-Sent Events endpoint for server-initiated notifications."""

    async def event_generator():
        while True:
            # Send a heartbeat every 30 seconds
            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
            await asyncio.sleep(30)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
