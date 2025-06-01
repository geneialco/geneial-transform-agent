# Langmanus Transform MCP Server

This is an implementation of a Model Context Protocol (MCP) server that provides data transformation capabilities from TOPMed to different formats (LinkML, Phenopackets JSON, and Phenopackets CSV).


## Features

- Transform data between formats:
  - LinkML (Linked Data Modeling Language)
  - Phenopackets JSON
  - Phenopackets CSV
- JSON-RPC 2.0 protocol support
- MCP protocol compliance
- Batch request support
- Server-Sent Events (SSE) for future real-time updates
- Health check endpoint
- Error handling with proper JSON-RPC error codes

## Installation

1. Install the required dependencies:
```bash
pip install fastapi==0.104.1 uvicorn==0.23.2 pydantic==2.4.2 httpx==0.25.0 python-dotenv==1.0.0
```

2. Set up environment variables:
```bash
# Create a .env file with:
PORT=8001
MAX_RETRIES=10
PHENOPACKET_TOOLS_JAR_PATH=/path/to/phenopacket-tools-cli.jar
```

## Running the Server

### HTTP Server (for testing and development)

```bash
# From the project root
python -m src.mcp_server

# Or with uvicorn directly
uvicorn src.mcp_server:app --host 0.0.0.0 --port 8001
```

### Stdio Server (for Claude Desktop)

```bash
# From the project root
python -m src.mcp_server_stdio
```

## Claude Desktop Integration

To use this MCP server with Claude Desktop:

1. Locate your Claude Desktop configuration file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. Add the following configuration to the `mcpServers` section:

```json
{
  "mcpServers": {
    "langmanus-transform": {
      "command": "python",
      "args": ["-m", "src.mcp_server_stdio"],
      "cwd": "/path/to/langmanus-transform",
      "env": {
        "PYTHONPATH": "/path/to/langmanus-transform",
        "MAX_RETRIES": "10",
        "PHENOPACKET_TOOLS_JAR_PATH": "/path/to/phenopacket-tools-cli.jar"
      }
    }
  }
}
```

3. Replace `/path/to/langmanus-transform` with the actual path to your project directory.

4. Restart Claude Desktop to load the new MCP server.

5. You can now use the transformation tools in Claude by asking it to transform TOPMed data to LinkML, Phenopackets JSON, or CSV formats.

## MCP Methods Implemented

### 1. `initialize`
Initializes the connection and returns server capabilities.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {}
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {},
      "resources": null,
      "prompts": null
    },
    "serverInfo": {
      "name": "Geneial Langmanus Transform MCP Server",
      "version": "0.2.0"
    }
  }
}
```

### 2. `tools/list`
Lists all available tools.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "transform_to_linkml",
        "description": "Transform input data from TOPMed format to LinkML format",
        "inputSchema": {
          "type": "object",
          "properties": {
            "input_content": {
              "type": "string",
              "description": "The input data to transform"
            },
            "user_prompt": {
              "type": "string",
              "description": "Additional instructions for the transformation",
              "default": "Please process the following content..."
            },
            "validate": {
              "type": "boolean",
              "description": "Whether to validate the output",
              "default": true
            },
            "debug": {
              "type": "boolean",
              "description": "Enable debug mode",
              "default": false
            }
          },
          "required": ["input_content"]
        }
      }
      // ... other tools
    ]
  }
}
```

### 3. `tools/call`
Executes a specific tool.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "transform_to_linkml",
    "arguments": {
      "input_content": "Patient data...",
      "user_prompt": "Convert to LinkML",
      "validate": false,
      "debug": false
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Transformed output..."
      }
    ],
    "isError": false
  }
}
```

## Available Tools

1. **transform_to_linkml**: Transform input data from TOPMed format to LinkML format
2. **transform_to_phenopackets_json**: Transform input data from TOPMed format to Phenopackets JSON format
3. **transform_to_phenopackets_csv**: Transform input data from TOPMed format to Phenopackets CSV format

## Testing the Server

Use the provided test client:

```bash
python test_mcp_client.py
```

Or test manually with curl:

```bash
# Initialize
curl -X POST http://localhost:8001 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize"}'

# List tools
curl -X POST http://localhost:8001 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'

# Call a tool
curl -X POST http://localhost:8001 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "transform_to_linkml",
      "arguments": {
        "input_content": "ID,Age,Sex,Ethnicity,Race,BMI,SmokingStatus,SystolicBP,DiastolicBP,TotalCholesterol,MedicalHistory,SNP_rs1234,SNP_rs2345,SNP_rs3456,AncestryPC1,AncestryPC2
NWD000001,65,M,Non-Hispanic,White,28.4,Former,142,90,230,Coronary Artery Disease; Hypertension; Hyperlipidemia,AG,CT,AT,0.12,-0.03
NWD000002,45,F,Hispanic,White,22.1,Never,118,75,180,None,AA,TT,AA,0.04,0.01
",
        "user_prompt": "Convert to LinkML"
      }
    }
  }'
```

## Example Usage with Python

```python
import requests
import json

# Server URL
SERVER_URL = "http://localhost:8001"

# Initialize connection
response = requests.post(
    SERVER_URL,
    json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize"
    }
)
print(response.json())

# List available tools
response = requests.post(
    SERVER_URL,
    json={
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
)
print(response.json())

# Call a transformation tool
response = requests.post(
    SERVER_URL,
    json={
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "transform_to_linkml",
            "arguments": {
                "input_content": "Your patient data here",
                "user_prompt": "Convert to LinkML format"
            }
        }
    }
)
result = response.json()
if "result" in result:
    for content in result["result"]["content"]:
        print(content["text"])
```

## Integration with MCP Clients

This server can be used with any MCP-compatible client, such as:
- Claude Desktop (with MCP support)
- Custom MCP clients
- AI assistants that support MCP

## Additional Features

- **Health Check**: GET `/health` endpoint for monitoring
- **Server-Sent Events**: GET `/events` endpoint for future real-time updates (currently sends heartbeats)
- **Error Handling**: Proper JSON-RPC error responses with standard error codes
- **Batch Requests**: Support for JSON-RPC batch requests

## Error Handling

The server returns standard JSON-RPC error codes:
- `-32700`: Parse error (invalid JSON)
- `-32601`: Method not found
- `-32603`: Internal error

Error responses follow the JSON-RPC format:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found: unknown_method"
  }
}
```
