import argparse
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Union
from .workflow import run_agent_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_system_prompt(output_format: str) -> str:
    """Get the appropriate system prompt based on the output format."""
    base_prompt = """You are a helpful assistant that can process and transform content according to specified rules.
Your task is to transform the input content in TOPMED format into the specified output format."""

    format_specific_prompts = {
        "linkml": """
Output Format: LINKML

Instructions for LinkML format:
- Ensure the output follows LinkML schema conventions
- Maintain proper YAML structure
- Include all required fields and types
- Use correct LinkML syntax for class definitions
- Preserve relationships between entities
- Create a linkml schema for the output and output it first after a line that says SCHEMA
- After the schema output the output text after a line that says OUTPUT
- Do not include any other text before or after the SCHEMA or OUTPUT lines""",
        "phenopackets-json": """
Output Format: PHENOPACKETS (JSON)

Instructions for Phenopackets JSON format:
- Follow the Phenopackets schema specification v2.0
- Include all necessary phenotypic information
- Maintain proper JSON structure
- Include required fields: id, subject, phenotypic_features
- Use proper ontology terms and references
- Create a phenopackets schema for the output and output it first after a line that says SCHEMA
- After the schema output the output text after a line that says OUTPUT
- Do not include any other text before or after the SCHEMA or OUTPUT lines""",
        "phenopackets-csv": """
Output Format: PHENOPACKETS (CSV)

Instructions for Phenopackets CSV format:
- Create a tabular representation of phenotypic data
- Include headers for all required phenopacket fields
- Use semicolon as delimiter for multiple values
- Include columns: id, subject_id, phenotype_id, phenotype_label, onset
- Ensure proper escaping of special characters
- Create a phenopackets schema for the output and output it first after a line that says SCHEMA
- After the schema output the output text after a line that says OUTPUT
- Do not include any other text before or after the SCHEMA or OUTPUT lines""",
    }

    return base_prompt + format_specific_prompts.get(output_format, "")


def load_input_file(file_path: str) -> str:
    """Load content from input file based on file extension."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in [".json"]:
            return json.dumps(json.load(f), indent=2)
        elif path.suffix in [".yaml", ".yml"]:
            return yaml.safe_dump(yaml.safe_load(f))
        elif path.suffix in [".csv"]:
            return f.read()  # Keep CSV as is
        else:
            return f.read()


def save_output_file(content: Any, file_path: str, output_format: str):
    """Save content to output file based on format and extension."""
    path = Path(file_path)
    with path.open("w", encoding="utf-8") as f:
        if output_format == "phenopackets-json" or path.suffix == ".json":
            json.dump(content, f, indent=2)
        elif output_format == "linkml" or path.suffix in [".yaml", ".yml"]:
            yaml.safe_dump(content, f)
        elif output_format == "phenopackets-csv" or path.suffix == ".csv":
            if isinstance(content, str):
                f.write(content)
            else:
                # Convert dict/list to CSV if needed
                import csv

                if isinstance(content, dict):
                    writer = csv.DictWriter(f, fieldnames=content.keys())
                    writer.writeheader()
                    writer.writerow(content)
                elif isinstance(content, list):
                    if content and isinstance(content[0], dict):
                        writer = csv.DictWriter(f, fieldnames=content[0].keys())
                        writer.writeheader()
                        writer.writerows(content)
                    else:
                        writer = csv.writer(f)
                        writer.writerows(content)
        else:
            if isinstance(content, (dict, list)):
                json.dump(content, f, indent=2)
            else:
                f.write(str(content))


def split_schema_and_output(result: Union[str, Dict, Any]) -> Tuple[str, str]:
    """Split the result into schema and output parts."""
    logger.info("Processing result of type: %s", type(result))

    # Handle dictionary with messages key
    if isinstance(result, dict) and "messages" in result:
        logger.info("Found messages in result dictionary")
        # Look through all messages for schema and output
        for i, msg in enumerate(result["messages"]):
            logger.debug("Processing message %d: %s", i, type(msg))
            if hasattr(msg, "content") and isinstance(msg.content, str):
                content = msg.content
                logger.debug(
                    "Message %d content (first 200 chars): %s", i, content[:200] + "..."
                )

                # If this message contains both SCHEMA and OUTPUT, use it
                if "SCHEMA" in content and "OUTPUT" in content:
                    logger.info(
                        "Found message %d with both SCHEMA and OUTPUT markers", i
                    )
                    result_str = content
                    break
        else:
            # If we didn't find both markers in any single message, combine all messages
            logger.info("No single message had both markers - combining all messages")
            result_str = "\n".join(
                str(msg.content)
                for msg in result["messages"]
                if hasattr(msg, "content")
            )
    else:
        # Handle other types as before
        if isinstance(result, (list, tuple)):
            messages_content = []
            logger.info("Processing list of %d messages", len(result))
            for i, msg in enumerate(result):
                logger.debug("Processing list item %d: %s", i, type(msg))
                if hasattr(msg, "content"):
                    messages_content.append(str(msg.content))
                elif isinstance(msg, dict) and "content" in msg:
                    messages_content.append(str(msg["content"]))
                elif isinstance(msg, str):
                    messages_content.append(msg)
                else:
                    messages_content.append(str(msg))
            result_str = "\n".join(messages_content)
        elif hasattr(result, "content"):
            logger.info("Processing single message with content attribute")
            result_str = str(result.content)
        elif isinstance(result, str):
            logger.info("Processing raw string result")
            result_str = result
        else:
            logger.info("Processing unknown type, converting to string")
            result_str = str(result)

    logger.debug(
        "Final combined content (first 200 chars):\n%s", result_str[:200] + "..."
    )

    # Remove any wrapping backticks and language identifiers
    lines = result_str.split("\n")
    cleaned_lines = []
    in_code_block = False
    for line in lines:
        # Skip lines that are just backticks or have language identifiers
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            logger.debug("Found code block marker: %s", line.strip())
            continue
        cleaned_lines.append(line)

    result_str = "\n".join(cleaned_lines)
    logger.debug("Cleaned content (first 200 chars):\n%s", result_str[:200] + "...")

    # Split the result into parts
    parts = result_str.split("\n")
    schema = []
    output = []
    current_section = None

    logger.info("Splitting content into sections")
    logger.debug("Total lines to process: %d", len(parts))

    # Look for SCHEMA and OUTPUT markers
    for i, line in enumerate(parts):
        stripped_line = line.strip()
        if stripped_line == "SCHEMA":
            current_section = "schema"
            schema = []  # Reset schema when we find a new SCHEMA marker
            logger.info("Found SCHEMA marker at line %d", i + 1)
        elif stripped_line == "OUTPUT":
            current_section = "output"
            output = []  # Reset output when we find a new OUTPUT marker
            logger.info("Found OUTPUT marker at line %d", i + 1)
        elif current_section == "schema" and stripped_line != "SCHEMA":
            schema.append(line)
        elif current_section == "output" and stripped_line != "OUTPUT":
            output.append(line)

    schema_content = "\n".join(schema).strip()
    output_content = "\n".join(output).strip()

    logger.info("Final schema section length: %d characters", len(schema_content))
    logger.info("Final output section length: %d characters", len(output_content))

    if not schema_content and not output_content:
        logger.error(
            "Neither schema nor output content found. Full received content:\n%s",
            result_str,
        )
    elif not schema_content:
        logger.warning("No schema content found")
    elif not output_content:
        logger.error("No output content found")

    return schema_content, output_content


def process_file(
    input_path: str,
    output_path: str,
    schema_path: str,
    user_prompt: str,
    output_format: str,
    debug: bool = False,
) -> None:
    """Process input file using the workflow and save results."""
    if debug:
        logger.setLevel(logging.DEBUG)

    max_retries = 3
    retry_delay = 2  # seconds between retries

    # Load input content
    input_content = load_input_file(input_path)
    logger.info("Loaded input file: %s", input_path)

    # Get the appropriate system prompt
    system_prompt = get_system_prompt(output_format)

    # Combine system prompt with user prompt and input content
    full_prompt = f"{system_prompt}\n\n{user_prompt}\n\n{input_content}"
    logger.debug("Full prompt:\n%s", full_prompt)

    last_error = None
    for attempt in range(max_retries):
        try:
            logger.info("Starting attempt %d of %d", attempt + 1, max_retries)

            # Run the workflow
            result = run_agent_workflow(full_prompt, debug=debug)
            logger.info("Workflow completed, processing result")

            # Log the raw result for debugging
            logger.info("Raw result type: %s", type(result))
            logger.debug(
                "Raw result content:\n%s",
                str(result)[:2000] + "..." if len(str(result)) > 2000 else str(result),
            )

            # Split result into schema and output
            schema_content, output_content = split_schema_and_output(result)

            # Log the split contents for debugging
            logger.info(
                "Schema content length: %d",
                len(schema_content) if schema_content else 0,
            )
            logger.info(
                "Output content length: %d",
                len(output_content) if output_content else 0,
            )
            logger.debug(
                "Schema content:\n%s",
                (
                    schema_content[:500] + "..."
                    if schema_content and len(schema_content) > 500
                    else schema_content
                ),
            )
            logger.debug(
                "Output content:\n%s",
                (
                    output_content[:500] + "..."
                    if output_content and len(output_content) > 500
                    else output_content
                ),
            )

            # Check if output content exists (required)
            if not output_content:
                raise ValueError("No output was generated in the response")

            # Try parsing the schema content if it exists
            if schema_content:
                if output_format == "phenopackets-json":
                    json.loads(schema_content)  # Validate JSON
                    logger.info("Successfully validated JSON schema")
                elif output_format == "linkml":
                    yaml.safe_load(schema_content)  # Validate YAML
                    logger.info("Successfully validated YAML schema")
                # Save the schema if it exists and is valid
                save_output_file(schema_content, schema_path, output_format)
                logger.info("Saved schema to: %s", schema_path)
            else:
                logger.warning(
                    "No schema was generated, but continuing since output exists"
                )
                # Write an empty file for schema to maintain consistent behavior
                Path(schema_path).touch()

            # Save the output content
            save_output_file(output_content, output_path, output_format)
            logger.info("Saved output to: %s", output_path)

            # If successful, print attempt number if not first try
            if attempt > 0:
                logger.info("Succeeded on attempt %d", attempt + 1)
            return

        except (ValueError, json.JSONDecodeError, yaml.YAMLError) as e:
            last_error = str(e)
            logger.error("Attempt %d failed: %s", attempt + 1, str(e))
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                logger.info("Retrying in %d seconds...", retry_delay)
                time.sleep(retry_delay)
                # Add a note to the prompt about the error for next attempt
                full_prompt = f"{full_prompt}\n\nNote: Previous attempt failed because: {str(e)}. Please ensure to include an OUTPUT section with valid formatting."

    # If we get here, all retries failed
    logger.error("All attempts failed. Last error: %s", last_error)
    raise ValueError(f"Failed after {max_retries} attempts. Last error: {last_error}")


def main():
    parser = argparse.ArgumentParser(description="Process files using embedded prompts")
    parser.add_argument("input", help="Path to input file")
    parser.add_argument("output", help="Path to output file")
    parser.add_argument("--schema", help="Path to schema output file", required=True)
    parser.add_argument(
        "--format",
        choices=["linkml", "phenopackets-json", "phenopackets-csv"],
        required=True,
        help="Specify output format (linkml, phenopackets-json, or phenopackets-csv)",
    )
    parser.add_argument(
        "--prompt", help="Additional prompt to append after system prompt", default=None
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts (default: 3)",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=2,
        help="Delay between retries in seconds (default: 2)",
    )

    args = parser.parse_args()

    # Default user prompt template - can be overridden via CLI
    default_user_prompt = """Please process the following content and transform it according to the specified rules:"""

    # Use provided prompt or default
    user_prompt = args.prompt if args.prompt else default_user_prompt

    try:
        process_file(
            args.input, args.output, args.schema, user_prompt, args.format, args.debug
        )
        print(f"Successfully processed {args.input}")
        print(f"Schema saved to: {args.schema}")
        print(f"Output saved to: {args.output}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise


if __name__ == "__main__":
    main()
