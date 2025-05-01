import argparse
import json
import yaml
import time
import logging
import re
import subprocess
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
- Create a linkml schema for the output and output it first after a line that says SCHEMA
- After the schema output the output text after a line that says OUTPUT
- Ensure the output follows LinkML schema conventions
- Maintain proper YAML structure
- The `imports:` section MUST be a YAML list of *scalar* CURIE or URI strings
  (e.g.  `imports:\n  - linkml:types`). Never include a mapping such as
  `{id: …, name: …}` under `imports`.
- Declare namespace bindings in a top-level `prefixes:` section instead:
    prefixes:
      linkml: https://w3id.org/linkml/
      xsd:    http://www.w3.org/2001/XMLSchema#
    default_prefix: <base-IRI>
- Includes a root class (tree_root: true) with a list of records
- In the schema, any attribute that should hold a list **MUST** include both `inlined_as_list: true` **and** `multivalued: true` so the instance YAML can supply an array without validation errors.
- When adding attributes to a class:
  - List only the **slot names** under `slots`.  
  - Define the slot's properties in a top-level `slots:` section, or (if class-specific) under `slot_usage:`.  
  - **Do NOT** embed a mapping inside `classes.<Class>.slots`.
- Place slot_usage: only inside the class where the overrides apply. Never put a slot_usage: mapping at the top level or keyed by the class name.
- In the schema defines all classes, enums, attributes, types, and ranges explicitly
- In the schema you must generate a complete LinkML schema with the following required top-level metadata fields:
  - id: A unique identifier URI or string for the schema
  - name: A required short name for the schema that follows NCName rules (from XML standards used by LinkML):
    - No spaces
    - Cannot start with a number
    - Cannot include colons
    - Should be a simple identifier like TopmedData or topmed_data
  - description: A brief description of what the schema represents
- Includes permissible enum values for any categorical fields
- Uses correct range references for enums (not enum:)
- Always define enums using permissible_values (not values) and make sure the key and text are identical
- Use a dictionary format, not a list
- Quote enum values if they contain spaces or special characters
- In the schema, use range: (not type:) for all attribute definitions in classes
- Do not use permissible_values inside class attributes — only define them in the enums: section
- All enums must be declared in the enums: section and referenced using range: in the class attributes
- Enum definitions must use permissible_values: with dictionary format (not list format):
- Ensure the instance YAML matches enum values exactly
- Include all required fields and types
- Use correct LinkML syntax for class definitions
- Preserve relationships between entities
- Under every classes.<Class>.slots key, express the slot names as a proper YAML list using leading hyphens (- age, - gender, etc.); never combine them into one line or mapping.
- Create a separate root collection class (e.g., RecordCollection) marked tree_root: true that contains one multivalued, inlined_as_list: true slot (e.g., records) whose range is the record class, so the instance YAML can legitimately start with that slot.
- In enums:, use permissible_values with text: and description:, not value:
- In the schema defines all classes, enums, attributes, types, and ranges explicitly.  
   - **Enum rules** – for every `permissible_values` entry:  
   - The *key* and its `text` **MUST be identical** (case-sensitive).  
   - If you need a display label that differs from the key, omit the `text` field entirely or use annotations instead.  
- With permissable_values be sure to match the possible values in the enum and the output exactly including any spacing and quotes
- Reference each slot's range to the corresponding EnumName
- Continue importing/linking xsd types so integer fields validate
- Create a linkml schema for the output and output it first after a line that says SCHEMA
- After the schema output the output text after a line that says OUTPUT
- Starts with a top-level key that matches the root class (e.g., records:)
- Matches the data structure defined by the schema
- Do not include any other text before or after the SCHEMA or OUTPUT lines
- Ensure that the output can be copied as embedded YAML text""",
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
        # Handle JSON output - expects Python object, try parsing if string
        if output_format == "phenopackets-json" or path.suffix == ".json":
            try:
                parsed_content = (
                    json.loads(content) if isinstance(content, str) else content
                )
                json.dump(parsed_content, f, indent=2)
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse content as JSON for {file_path}, writing as string."
                )
                f.write(str(content))  # Fallback
        # Handle LinkML/YAML - content from split_schema_and_output is already a formatted string
        elif output_format == "linkml" or path.suffix in [".yaml", ".yml"]:
            if isinstance(content, str):
                f.write(content)  # Write the string directly
                # Add a newline if the content doesn't end with one, for cleaner files
                if not content.endswith("\n"):
                    f.write("\n")
            else:
                # Fallback if content is not a string (unexpected)
                logger.warning(
                    f"Received non-string content for {output_format} format. Attempting yaml.safe_dump."
                )
                yaml.safe_dump(content, f)
        # Handle CSV - already handles strings correctly
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
    logger.info("=" * 80)
    logger.info("RAW RESULT:")
    logger.info(str(result))
    logger.info("=" * 80)

    result_str = ""  # Initialize result_str

    # --- REVISED HANDLING LOGIC ---

    potential_content = []

    # 1. If it's a dictionary, check all its values
    if isinstance(result, dict):
        logger.info("Result is a dictionary. Searching through its values for content.")
        for key, value in result.items():
            logger.debug(f"Checking dictionary key: '{key}', type: {type(value)}")
            # --- REVISED: Handle lists by iterating items ---
            if isinstance(value, list) or isinstance(value, tuple):
                # If value is a list/tuple, process each item individually
                logger.debug(
                    f"Value for key '{key}' is a list/tuple. Processing items individually..."
                )
                for i, item in enumerate(value):
                    item_part = None
                    if isinstance(item, str):
                        item_part = item
                    elif hasattr(item, "content") and isinstance(
                        getattr(item, "content"), str
                    ):
                        item_part = getattr(item, "content")

                    if item_part:
                        logger.info(
                            f"Found potential content in list item {i} from key '{key}'"
                        )
                        potential_content.append(
                            item_part
                        )  # Add item content separately
            # --- Handle strings ---
            elif isinstance(value, str):
                content_part = value
                logger.info(
                    f"Found potential string content in dictionary value associated with key '{key}'"
                )
                potential_content.append(content_part)
            # --- Handle objects with .content ---
            elif hasattr(value, "content") and isinstance(
                getattr(value, "content"), str
            ):
                content_part = getattr(value, "content")
                logger.info(
                    f"Found potential content in object with .content attribute associated with key '{key}'"
                )
                potential_content.append(content_part)
            # --- Other types are ignored in this loop (handled by fallback later if needed) ---

    # 2. If it's a list or tuple directly (outside a dict)
    elif isinstance(result, (list, tuple)):
        logger.info("Result is a list/tuple directly. Processing items.")
        for i, item in enumerate(result):
            content_part = None
            if isinstance(item, str):
                content_part = item
            elif hasattr(item, "content") and isinstance(getattr(item, "content"), str):
                content_part = getattr(item, "content")
            if content_part:
                # logger.debug(f"Content part from list item {i}:\n{content_part}")
                potential_content.append(content_part)

    # 3. If it has a .content attribute directly
    elif hasattr(result, "content") and isinstance(getattr(result, "content"), str):
        logger.info("Result has a direct .content attribute.")
        content_part = getattr(result, "content")
        potential_content.append(content_part)
        # logger.debug(f"Content part:\n{content_part}")

    # 4. If it's just a string
    elif isinstance(result, str):
        logger.info("Result is a raw string.")
        potential_content.append(result)
        # logger.debug(f"Content part:\n{result}")

    # 5. Fallback: Convert to string
    else:
        logger.warning(
            "Result type %s not directly handled for content extraction. Converting to string.",
            type(result),
        )
        potential_content.append(str(result))

    # Combine all found potential content
    # --- REVISED: Prioritize content with markers ---
    prioritized_content = None
    if not potential_content:
        logger.error(
            "Could not extract any potential string content from the workflow result."
        )
        result_str = ""  # Ensure result_str is empty
    else:
        logger.info(
            f"Found {len(potential_content)} potential content part(s). Searching for the one containing SCHEMA and OUTPUT."
        )
        for i, part in enumerate(potential_content):
            # Check if this part contains both markers (case-sensitive)
            # --- REVISED CHECK: Look for a more specific pattern ---
            schema_marker_pattern = (
                "SCHEMA"  # Expect this potentially at start or after newline
            )
            output_marker_pattern = (
                "\nOUTPUT"  # Expect OUTPUT likely preceded by newline
            )

            schema_pos = part.find(schema_marker_pattern)
            # Search for output marker *after* the schema marker position
            output_pos = -1
            if schema_pos != -1:
                output_pos = part.find(output_marker_pattern, schema_pos)

            # if "SCHEMA" in part and "OUTPUT" in part: # Old, too simple check
            if (
                schema_pos != -1 and output_pos != -1
            ):  # Check if both found, and output is after schema
                logger.info(
                    f"Prioritizing content part {i} as it contains distinct SCHEMA and OUTPUT markers in sequence."
                )
                prioritized_content = part
                break  # Use the first part found that contains both in sequence

        if prioritized_content:
            result_str = prioritized_content
        else:
            # Fallback: If no single part has both markers, combine everything as before (might indicate an issue)
            logger.warning(
                "No single content part contained both SCHEMA and OUTPUT. Combining all parts as fallback."
            )
            result_str = "\n\n".join(
                potential_content
            )  # Join with double newline for separation

    # --- END REVISED ---

    logger.info("=" * 80)
    logger.info("FINAL CONTENT TO PARSE (BEFORE CLEANING/SPLITTING):")
    logger.info(result_str)
    logger.info("=" * 80)

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
    logger.info("AFTER CLEANING CODE BLOCKS:")
    logger.info(result_str)
    logger.info("=" * 80)

    # --- MODIFIED SPLITTING LOGIC ---
    schema_content = ""
    output_content = ""

    schema_marker = "SCHEMA"
    output_marker = "OUTPUT"

    logger.info("Searching for SCHEMA and OUTPUT markers using string find...")

    schema_start_index = result_str.find(schema_marker)
    output_start_index = result_str.find(output_marker)

    logger.info(f"SCHEMA marker found at index: {schema_start_index}")
    logger.info(f"OUTPUT marker found at index: {output_start_index}")

    if schema_start_index != -1 and output_start_index != -1:
        # Found both markers
        logger.info("Both markers found.")
        # Schema content starts after the SCHEMA marker + space/newline
        schema_content_start = schema_start_index + len(schema_marker)
        # Schema content ends just before the OUTPUT marker
        schema_content_end = output_start_index
        schema_content = result_str[schema_content_start:schema_content_end].strip()

        # Output content starts after the OUTPUT marker + space/newline
        output_content_start = output_start_index + len(output_marker)
        output_content = result_str[output_content_start:].strip()

        logger.info("Successfully extracted schema and output based on markers.")

    elif schema_start_index != -1:
        # Found only SCHEMA marker
        logger.warning(
            "Only SCHEMA marker found. Extracting schema content, output will be empty."
        )
        schema_content_start = schema_start_index + len(schema_marker)
        schema_content = result_str[schema_content_start:].strip()
        output_content = ""  # Explicitly set output to empty

    elif output_start_index != -1:
        # Found only OUTPUT marker (unlikely given prompt, but handle anyway)
        logger.warning("Only OUTPUT marker found. Schema will be empty.")
        output_content_start = output_start_index + len(output_marker)
        output_content = result_str[output_content_start:].strip()
        schema_content = ""  # Explicitly set schema to empty

    else:
        # Found neither marker
        logger.error("Neither SCHEMA nor OUTPUT marker found in the cleaned string.")
        # Keep both empty, error will be raised later
        schema_content = ""
        output_content = ""

    # --- END MODIFIED SPLITTING LOGIC ---

    logger.info("=" * 80)
    logger.info("FINAL SCHEMA CONTENT:")
    logger.info(schema_content if schema_content else "[EMPTY]")
    logger.info("=" * 80)
    logger.info("FINAL OUTPUT CONTENT:")
    logger.info(output_content if output_content else "[EMPTY]")
    logger.info("=" * 80)

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
    validate: bool = False,
    max_retries: int = 10,
) -> None:
    """Process input file using the workflow and save results."""
    if debug:
        logger.setLevel(logging.DEBUG)

    # --- Initial Generation Attempt ---
    # Use max_retries for initial loop as well
    initial_retry_delay = 2  # seconds between retries

    # Load input content
    input_content = load_input_file(input_path)
    logger.info("Loaded input file: %s", input_path)

    # Get the appropriate system prompt
    system_prompt = get_system_prompt(output_format)

    # Combine system prompt with user prompt and input content
    initial_full_prompt = f"{system_prompt}\n\n{user_prompt}\n\n{input_content}"
    current_prompt = (
        initial_full_prompt  # Use a separate var for potential modifications
    )
    logger.debug("Initial Full prompt:\n%s", current_prompt)

    last_initial_error = None
    schema_content = ""
    output_content = ""

    for attempt in range(max_retries):
        try:
            logger.info(
                "Starting initial generation attempt %d of %d",
                attempt + 1,
                max_retries,
            )

            # Run the workflow
            result = run_agent_workflow(current_prompt, debug=debug)
            logger.info("Workflow completed, processing result")

            # Log the raw result for debugging
            logger.info("=" * 80)
            logger.info(
                "RAW OUTPUT FROM run_agent_workflow (BEFORE split_schema_and_output):"
            )
            logger.info(f"Type: {type(result)}")
            logger.info(f"Content:\n{str(result)}")
            logger.info("=" * 80)

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

            # Try parsing the schema content if it exists and is linkml/yaml
            if schema_content and output_format == "linkml":
                try:
                    yaml.safe_load(schema_content)  # Validate YAML syntax
                    logger.info("Successfully syntax-checked YAML schema")
                except yaml.YAMLError as e:
                    raise ValueError(f"Generated schema is not valid YAML: {e}")
            elif schema_content and output_format == "phenopackets-json":
                try:
                    json.loads(schema_content)  # Validate JSON syntax
                    logger.info("Successfully syntax-checked JSON schema")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Generated schema is not valid JSON: {e}")

            # Save the potentially valid schema (if exists)
            if schema_content:
                save_output_file(schema_content, schema_path, output_format)
                logger.info("Saved schema to: %s", schema_path)
            else:
                logger.warning(
                    "No schema was generated, but continuing since output exists"
                )
                Path(schema_path).touch()  # Create empty file if no schema

            # Save the output content (parsing/validation depends on format/validation step)
            save_output_file(output_content, output_path, output_format)
            logger.info("Saved output to: %s", output_path)

            # If we got here, initial generation was successful enough to proceed
            logger.info("Initial generation successful (attempt %d)", attempt + 1)
            break  # Exit the initial generation retry loop

        except (ValueError, json.JSONDecodeError, yaml.YAMLError) as e:
            last_initial_error = str(e)
            logger.error(
                "Initial generation attempt %d failed: %s", attempt + 1, str(e)
            )
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                logger.info(
                    "Retrying initial generation in %d seconds...", initial_retry_delay
                )
                time.sleep(initial_retry_delay)
                # Add a note to the prompt about the error for next attempt
                current_prompt = f"{initial_full_prompt}\n\nNote: Previous attempt failed because: {str(e)}. Please ensure to include both SCHEMA and OUTPUT sections with valid formatting as requested."
            else:
                # If all initial retries failed
                logger.error(
                    "All initial generation attempts failed. Last error: %s",
                    last_initial_error,
                )
                raise ValueError(
                    f"Failed initial generation after {max_retries} attempts. Last error: {last_initial_error}"
                )

    # --- Validation and Correction Loop (if requested) ---
    if validate and output_format == "linkml":
        # Remove hardcoded value, use max_retries from args
        validation_retry_delay = 2  # seconds
        validation_successful = False
        successful_val_attempt_count = 0  # To store the attempt number on success

        for val_attempt in range(max_retries):  # Use max_retries here
            logger.info(
                "Starting validation attempt %d of %d for LinkML.",
                val_attempt + 1,
                max_retries,  # Use max_retries here
            )

            if not Path(schema_path).exists() or Path(schema_path).stat().st_size == 0:
                logger.error(
                    "Validation failed: Schema file '%s' is missing or empty.",
                    schema_path,
                )
                # Treat this as a validation failure, maybe prompt LLM to generate it?
                # For now, we break the validation loop if schema is missing after generation
                # Could potentially add logic here to prompt for schema generation specifically.
                validation_successful = False
                break  # Exit validation loop

            # 1. Extract Target Class from Schema
            target_class = None
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema_file_content = f.read()
                # Regex to find the first class name after 'classes:'
                # Looks for 'classes:', optional whitespace/newline, then captures the word before ':'
                match = re.search(
                    r"^classes:\s*\n(?:\s*#.*\n)*\s*(\w+):",
                    schema_file_content,
                    re.MULTILINE,
                )
                if match:
                    target_class = match.group(1)
                    logger.info(f"Extracted target class: {target_class}")
                else:
                    logger.error(
                        "Could not find 'classes:' section or target class in schema file: %s",
                        schema_path,
                    )
                    # This is a validation failure, maybe prompt LLM to fix schema structure?
                    # Break validation loop for now.
                    validation_successful = False
                    break
            except Exception as e:
                logger.error(
                    f"Error reading schema or extracting target class from {schema_path}: {e}"
                )
                validation_successful = False
                break  # Exit validation loop

            # 2. Run linkml-validate command
            validation_cmd = [
                "linkml-validate",
                "--schema",
                str(schema_path),
                "--target-class",
                target_class,
                str(output_path),
            ]
            logger.info(f"Running validation command: {' '.join(validation_cmd)}")

            try:
                process = subprocess.run(
                    validation_cmd, capture_output=True, text=True, check=False
                )  # check=False to handle errors manually
                stdout = process.stdout.strip()
                stderr = process.stderr.strip()
                logger.debug(f"linkml-validate stdout:\n{stdout}")
                logger.debug(f"linkml-validate stderr:\n{stderr}")

                # 3. Check validation result
                # linkml-validate exits 0 on success and prints "No issues found" to stdout
                # It exits non-zero on failure and prints errors to stderr
                if process.returncode == 0 and "No issues found" in stdout:
                    logger.info("Validation successful! No issues found.")
                    validation_successful = True
                    successful_val_attempt_count = (
                        val_attempt + 1
                    )  # Record successful attempt number
                    break  # Exit validation loop
                else:
                    # Validation failed
                    error_message = (
                        stderr if stderr else stdout
                    )  # Prefer stderr for errors
                    logger.error(
                        f"Validation attempt {val_attempt + 1} failed. Error:\n{error_message}"
                    )

                    if val_attempt < max_retries - 1:
                        logger.info("Attempting to fix using LLM...")
                        # Construct prompt for LLM to fix the issue
                        fix_prompt = (
                            f"{initial_full_prompt}\n\n"
                            f"--- PREVIOUSLY GENERATED SCHEMA ---\n{schema_content}\n"
                            f"--- PREVIOUSLY GENERATED OUTPUT ---\n{output_content}\n\n"
                            f"--- VALIDATION ERROR ---\n{error_message}\n\n"
                            f"--- INSTRUCTION ---\n"
                            f"The previous generation resulted in the validation error above. "
                            f"Please analyze the schema, output, and the error message. "
                            f"Fix the issues in the schema and/or output according to the LinkML rules and the error. "
                            f"Provide the corrected full schema content after a 'SCHEMA' line and the corrected full output content after an 'OUTPUT' line."
                        )
                        logger.debug("LLM Fix Prompt:\n%s", fix_prompt)

                        # Call LLM again
                        try:
                            result = run_agent_workflow(fix_prompt, debug=debug)
                            logger.info("Correction attempt workflow completed.")

                            # Re-split and save
                            new_schema_content, new_output_content = (
                                split_schema_and_output(result)
                            )

                            if not new_output_content:
                                logger.warning(
                                    f"Correction attempt {val_attempt + 1} did not produce output content. Retrying validation with previous files."
                                )
                                # Keep old content, validation will likely fail again, but prevents erroring out here
                            else:
                                # Basic check for schema validity before saving
                                if new_schema_content:
                                    try:
                                        yaml.safe_load(new_schema_content)
                                        schema_content = (
                                            new_schema_content  # Update schema content
                                        )
                                        save_output_file(
                                            schema_content, schema_path, output_format
                                        )
                                        logger.info(
                                            f"Saved corrected schema to: {schema_path}"
                                        )
                                    except yaml.YAMLError as e:
                                        logger.warning(
                                            f"LLM correction produced invalid YAML schema: {e}. Keeping previous schema."
                                        )
                                        # Keep old schema content
                                else:
                                    logger.warning(
                                        "LLM correction did not produce schema content. Keeping previous schema."
                                    )
                                    # Keep old schema content if LLM didn't provide one

                                output_content = (
                                    new_output_content  # Update output content
                                )
                                save_output_file(
                                    output_content, output_path, output_format
                                )
                                logger.info(f"Saved corrected output to: {output_path}")

                        except Exception as llm_e:
                            logger.error(
                                f"LLM Correction attempt {val_attempt + 1} failed during workflow: {llm_e}"
                            )
                            # Continue to next validation attempt with old files

                        logger.info(
                            f"Retrying validation in {validation_retry_delay} seconds..."
                        )
                        time.sleep(validation_retry_delay)
                    else:
                        # Max validation retries reached
                        logger.error(
                            "Validation failed after %d attempts.",
                            max_retries,  # Use max_retries here
                        )
                        validation_successful = False
                        # Keep loop going so it exits naturally and raises error below

            except FileNotFoundError:
                logger.error(
                    f"Validation failed: 'linkml-validate' command not found. Make sure LinkML is installed and in the system PATH."
                )
                validation_successful = False
                break  # Cannot proceed without the command
            except Exception as val_e:
                logger.error(f"An unexpected error occurred during validation: {val_e}")
                validation_successful = False
                break  # Exit validation loop on unexpected error

        # After validation loop: check final status
        if not validation_successful:
            raise ValueError(
                f"LinkML validation failed after {max_retries} attempts. Check logs for details."  # Use max_retries here
            )
        else:
            logger.info(
                f"LinkML validation completed successfully after {successful_val_attempt_count} attempt(s)."
            )

    # --- End of Validation --- (or if validation wasn't requested) ---
    # If we reach here, either validation was not requested, not applicable, or succeeded.
    logger.info("Processing finished successfully.")
    # Return is implicit (None)


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
        default=10,
        help="Maximum number of retry attempts for initial generation and validation (default: 10)",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=2,
        help="Delay between initial generation retries in seconds (default: 2)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation on the generated output (currently only LinkML supported)",
    )

    args = parser.parse_args()

    # Default user prompt template - can be overridden via CLI
    default_user_prompt = """Please process the following content and transform it according to the specified rules:"""

    # Use provided prompt or default
    user_prompt = args.prompt if args.prompt else default_user_prompt

    try:
        process_file(
            args.input,
            args.output,
            args.schema,
            user_prompt,
            args.format,
            args.debug,
            args.validate,
            args.max_retries,
        )
        print(f"Successfully processed {args.input}")
        print(f"Schema saved to: {args.schema}")
        print(f"Output saved to: {args.output}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise


if __name__ == "__main__":
    main()
