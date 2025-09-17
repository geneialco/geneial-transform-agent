import argparse
import json
import yaml
import time
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Union
from .workflow import run_agent_workflow
from .utils.umls_client import get_umls_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Path to the Phenopacket tools JAR file.
# This is based on the user's provided example and might need to be configured
# or ensured it's available in the execution environment.
PHENOPACKET_TOOLS_JAR_PATH = "/Applications/phenopacket-tools-cli-1.0.0-RC3.jar"


def get_system_prompt(output_format: str, use_umls: bool = False) -> str:
    """Get the appropriate system prompt based on the output format."""
    base_prompt = """You are a helpful assistant that can process and transform content according to specified rules.
Your task is to transform the input content in TOPMED format into the specified output format."""

    format_specific_prompts = {
        "linkml": """
Output Format: LINKML

####################################################
##  SECTION A – HOW TO EMIT THE RESULT
####################################################
1. Emit **exactly two** top-level blocks, in this order­­:
   • A line `SCHEMA` followed by the complete LinkML schema  
   • A line `OUTPUT` followed by the instance YAML  
   No other prose may appear before, between, or after those blocks.

####################################################
##  SECTION B – SCHEMA RULES
####################################################
# ──  Metadata ────────────────────────────────────
2. Provide required keys **id**, **name**, **description**.  
   • `name` must be a valid NCName (no spaces, no leading digit, no colon).

3. `prefixes:` **must** include  
   linkml: https://w3id.org/linkml/  
   xsd:    http://www.w3.org/2001/XMLSchema#  
   default_prefix: <base-IRI>

4. `imports:` is a YAML list; include only  
   - linkml:types  
   Never use mappings inside `imports:`.

# ──  Root design ─────────────────────────────────
5. Create a collection class (e.g., **RecordCollection**) with `tree_root: true`.  
   • It owns one slot (e.g., **records**) that is `multivalued: true` and `inlined_as_list: true`, with `range: Record`.  
   • In the instance YAML, start with that slot key (`records:`).

# ──  Classes & slots ─────────────────────────────
6. Under `classes.<Class>.slots`, list **only slot names** (YAML list, each prefixed with "  ").  
7. Define every slot in either  
   • the top-level **slots:** block, **or**  
   • the owning class's **slot_usage:** block.  
8. If a slot name appears in classes.<Class>.slots, it must either be:
   • defined under top-level slots: or
   • defined inline under slot_usage in the same class (with full attributes like range:, multivalued:, etc.) so that the engine recognizes it as a genuine slot.
9. For list-valued attributes, always include both `multivalued: true` *and* `inlined_as_list: true`.  
10. Use `range:` (never `type:`) for attribute typing. Primitives reference `xsd:` literals.
12. For any slot mentioned under classes.<MyClass>.slots, also ensure it is defined either in the top-level slots: 
block or as a fully-defined slot in slot_usage: (for newer LinkML versions). Never just list a slot name without a matching definition.
13. "Do not list any slot in classes.<Class>.slots unless there is a matching slot definition in the top-level slots: or inline in slot_usage:. 
In particular, the 'records' slot on RecordCollection must be explicitly declared with range: Record, multivalued: true, and inlined_as_list: true."

# ──  Enumerations ────────────────────────────────
14. Declare all enums under `enums:` using dictionary form with `permissible_values:`.  
    • The **key** and its `text` must be identical (case-sensitive).  
    • Quote keys containing spaces or special characters.  
    • Reference enums in slots via `range: <EnumName>`.  
    • Never embed `permissible_values` inside a class attribute.

# ──  Validation guard-rails ─────────────────────
15. **Missing-slot safeguard** — *Every slot listed under any `classes.<Class>.slots` MUST have a matching definition.*  
    Correct pattern (copyable example):  
    ```yaml
    classes:
      Person:
        slots:
          - given_name        # ← slot reference

    slots:
      given_name:             # ← matching definition
        range: xsd:string
    ```
16. **Self-check directive** — After constructing `classes:` and their slot lists, iterate over **all** referenced slots (including primitives like `age`, `height`, etc.).  
    • **If even one slot lacks a definition, do NOT output the schema; regenerate instead until the check passes.**

# ──  YAML style ─────────────────────────────────
17. Use 2-space indentation throughout; never use TABs.

####################################################
##  SECTION C – INSTANCE YAML RULES
####################################################
18. Instance YAML must conform exactly to the schema (correct keys, ranges, enum spellings).  
19. Begin with the collection slot key (`records:`).  
20. Ensure the YAML can be pasted directly—no Markdown fencing or extra prose.

The following is an example schema snippet that shows how to strcuture the schema especially around how the slot section is defined:

```yaml
id: http://example.org/my-schema
name: ExampleSchema
description: An example schema for representing data records.
prefixes:
  linkml: https://w3id.org/linkml/
  xsd: http://www.w3.org/2001/XMLSchema#
  default_prefix: ex
imports:
- linkml:types
classes:
  DataCollection:
    tree_root: true
    slots:
    - data_entries
  DataEntry:
    slots:
    - timestamp
    - category
    - value
    - source
    - description
slots:
  timestamp:
    range: xsd:dateTime
    multivalued: false
  category:
    range: xsd:string
    multivalued: false
  value:
    range: xsd:float
    multivalued: false
  source:
    range: xsd:string
    multivalued: false
  description:
    range: xsd:string
    multivalued: false
  data_entries:
    range: DataEntry
    multivalued: true
    inlined_as_list: true
```

""",
        "phenopackets-json": """
Output Format: PHENOPACKETS (JSON)

Instructions for Phenopackets JSON format:
- Follow the Phenopackets schema specification v2.0 for cohort structure
- Generate a cohort containing multiple phenopackets as members
- Use proper cohort structure: {"id": "cohort-id", "description": "cohort description", "members": [array of phenopackets], "metaData": {...}}
- The cohort-level metaData must include: created, createdBy, resources, phenopacketSchemaVersion
- Each member in the "members" array should be a complete phenopacket with:
  - Required fields: id, subject, phenotypicFeatures, metaData
  - Subject structure must use:
    * "id": "subject-id" (REQUIRED field - typically matches the phenopacket id)
    * "timeAtLastEncounter": {"age": {"iso8601duration": "P35Y"}} (not direct "age" field)
    * "sex": "FEMALE" or "MALE" (enum values, not objects)
    * "karyotypicSex": "UNKNOWN_KARYOTYPE"
  - PhenotypicFeatures structure: [{"type": {"id": "HP:0000118", "label": "Height"}, "excluded": false}]
  - Required metaData fields: created, createdBy, resources, phenopacketSchemaVersion
  - Resources array must have proper structure: [{"id": "hp", "name": "human phenotype ontology", "url": "http://purl.obolibrary.org/obo/hp.owl", "version": "2022-06-11", "namespacePrefix": "HP", "iriPrefix": "http://purl.obolibrary.org/obo/HP_"}]
- Use proper ontology terms (HP: for phenotypes, etc.)
- Create a phenopackets schema for the output and output it first after a line that says SCHEMA
- After the schema output the cohort JSON directly (NOT wrapped in a "cohort" field) after a line that says OUTPUT
- The OUTPUT should start directly with {"id": "cohort-id", "description": ...}, not {"cohort": {...}}
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

    prompt = base_prompt + format_specific_prompts.get(output_format, "")

    # Add UMLS integration instructions if requested
    if use_umls:
        umls_instructions = """

####################################################
##  UMLS INTEGRATION INSTRUCTIONS
####################################################

When processing the data, you should identify and map clinical concepts using UMLS terminology:

1. **Identify Clinical Concepts**: Look for medical terms, conditions, measurements, and phenotypes in the input data.

2. **Clinical Concept Categories to Map**:
   - Physical measurements (BMI, blood pressure, cholesterol levels)
   - Medical conditions (diabetes, hypertension, asthma, COPD, etc.)
   - Demographic information (age, sex, ethnicity)
   - Behavioral factors (smoking status)
   - Laboratory values and vital signs

3. **UMLS Mapping Process**:
   - For each clinical concept identified, find the appropriate UMLS CUI (Concept Unique Identifier)
   - Use HPO (Human Phenotype Ontology) as the primary ontology for phenotypic features
   - Use SNOMED CT for general medical concepts when HPO is not applicable
   - Include both the UMLS code and the standardized term name

4. **Integration in Output**:
   - Add UMLS mappings as comments in the schema and output files
   - For LinkML: Include UMLS codes as comments above relevant class and slot definitions
   - For Phenopackets: Include UMLS annotations in the phenotypic features
   - Create a mapping section that shows: Original Term → UMLS Code (Standardized Term)

5. **Example UMLS Mapping Format**:
   ```
   # UMLS Mapping: BMI → C1305855 (Quetelet Index)
   # UMLS Mapping: Hypertension → C0020538 (Hypertension)
   # UMLS Mapping: Type 2 Diabetes → C0011860 (Type 2 diabetes mellitus)
   ```

6. **Documentation**: Include a comment section listing all UMLS mappings found during the transformation process.

Please ensure that the UMLS mappings are comprehensive and accurate, covering all clinical concepts present in the input data.
"""
        prompt += umls_instructions

    return prompt


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


def validate_output(
    schema_content: str,
    output_content: str,
    output_format: str,
    initial_full_prompt: str,
    debug: bool = False,
    max_retries: int = 10,
    use_umls: bool = False,
    use_search: bool = False,
) -> Tuple[str, str, bool, str, Dict]:
    """
    Validate output content and potentially correct it using LLM.

    Args:
        schema_content: The schema content to validate
        output_content: The output content to validate
        output_format: The output format ('linkml', 'phenopackets-json', 'phenopackets-csv')
        initial_full_prompt: The original prompt for potential LLM correction
        debug: Enable debug logging
        max_retries: Maximum number of validation retry attempts

    Returns:
        Tuple of (corrected_schema_content, corrected_output_content, validation_successful, validation_message, validation_stats)
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    # Initialize return values
    corrected_schema_content = schema_content
    corrected_output_content = output_content
    validation_successful = False
    validation_message = ""

    # Initialize validation statistics
    validation_stats = {
        "initial_validation_success": False,
        "total_retries": 0,
        "final_validation_success": False,
        "validation_attempts": 0,
        "llm_correction_attempts": 0,
    }

    if output_format == "linkml":
        validation_retry_delay = 2  # seconds

        for val_attempt in range(max_retries):
            validation_stats["validation_attempts"] += 1
            logger.info(
                "Starting LinkML validation attempt %d of %d.",
                val_attempt + 1,
                max_retries,
            )

            # Create temporary files for validation
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as schema_file:
                schema_file.write(corrected_schema_content)
                schema_path = schema_file.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as output_file:
                output_file.write(corrected_output_content)
                output_path = output_file.name

            try:
                if not corrected_schema_content.strip():
                    validation_message = (
                        "LinkML validation failed: Schema content is empty."
                    )
                    logger.error(validation_message)
                    validation_successful = False
                    break

                target_class = None
                try:
                    match = re.search(
                        r"^classes:\s*\n(?:\s*#.*\n)*\s*(\w+):",
                        corrected_schema_content,
                        re.MULTILINE,
                    )
                    if match:
                        target_class = match.group(1)
                        logger.info(
                            f"Extracted target class for LinkML: {target_class}"
                        )
                    else:
                        validation_message = "Could not find 'classes:' section or target class in LinkML schema."
                        logger.error(validation_message)
                        validation_successful = False
                        break
                except Exception as e:
                    validation_message = (
                        f"Error extracting target class from LinkML schema: {e}"
                    )
                    logger.error(validation_message)
                    validation_successful = False
                    break

                validation_cmd = [
                    "linkml-validate",
                    "--schema",
                    schema_path,
                    "--target-class",
                    target_class,
                    output_path,
                ]
                logger.info(
                    f"Running LinkML validation command: {' '.join(validation_cmd)}"
                )

                try:
                    process = subprocess.run(
                        validation_cmd, capture_output=True, text=True, check=False
                    )
                    stdout = process.stdout.strip()
                    stderr = process.stderr.strip()
                    logger.debug(f"linkml-validate stdout:\n{stdout}")
                    logger.debug(f"linkml-validate stderr:\n{stderr}")

                    if process.returncode == 0 and "No issues found" in stdout:
                        validation_message = (
                            "LinkML validation successful! No issues found."
                        )
                        logger.info(validation_message)
                        validation_successful = True
                        validation_stats["final_validation_success"] = True
                        if val_attempt == 0:
                            validation_stats["initial_validation_success"] = True
                        break
                    else:
                        error_message = stderr if stderr else stdout
                        validation_message = f"LinkML validation attempt {val_attempt + 1} failed. Error:\n{error_message}"
                        logger.error(validation_message)

                        validation_stats["total_retries"] = val_attempt + 1

                        if val_attempt < max_retries - 1:
                            validation_stats["llm_correction_attempts"] += 1
                            logger.info("Attempting to fix LinkML using LLM...")
                            fix_prompt = (
                                f"{initial_full_prompt}\n\n"
                                f"--- PREVIOUSLY GENERATED SCHEMA ---\n{corrected_schema_content}\n"
                                f"--- PREVIOUSLY GENERATED OUTPUT ---\n{corrected_output_content}\n\n"
                                f"--- VALIDATION ERROR ---\n{error_message}\n\n"
                                f"--- INSTRUCTION ---\n"
                                f"The previous LinkML generation resulted in the validation error above. "
                                f"Please analyze the schema, output, and the error message. "
                                f"Fix the issues in the schema and/or output according to the LinkML rules and the error. "
                                f"Provide the corrected full schema content after a 'SCHEMA' line and the corrected full output content after an 'OUTPUT' line."
                            )
                            logger.debug("LLM Fix Prompt (LinkML):\n%s", fix_prompt)

                            try:
                                result = run_agent_workflow(
                                    fix_prompt,
                                    debug=debug,
                                    use_umls=use_umls,
                                    use_search=use_search,
                                )
                                logger.info(
                                    "LinkML correction attempt workflow completed."
                                )
                                new_schema_content, new_output_content = (
                                    split_schema_and_output(result)
                                )

                                if not new_output_content:
                                    logger.warning(
                                        f"LinkML correction attempt {val_attempt + 1} did not produce output content. Retrying validation with previous content."
                                    )
                                else:
                                    if new_schema_content:
                                        try:
                                            yaml.safe_load(new_schema_content)
                                            corrected_schema_content = (
                                                new_schema_content
                                            )
                                            logger.info(
                                                "Updated corrected LinkML schema content"
                                            )
                                        except yaml.YAMLError as e:
                                            logger.warning(
                                                f"LLM correction produced invalid YAML schema for LinkML: {e}. Keeping previous schema."
                                            )
                                    else:
                                        logger.warning(
                                            "LLM correction did not produce LinkML schema content. Keeping previous schema."
                                        )
                                    corrected_output_content = new_output_content
                                    logger.info(
                                        "Updated corrected LinkML output content"
                                    )
                            except Exception as llm_e:
                                logger.error(
                                    f"LLM Correction attempt {val_attempt + 1} for LinkML failed during workflow: {llm_e}"
                                )
                            logger.info(
                                f"Retrying LinkML validation in {validation_retry_delay} seconds..."
                            )
                            time.sleep(validation_retry_delay)
                        else:
                            validation_message = f"LinkML validation failed after {max_retries} attempts."
                            logger.error(validation_message)
                            validation_successful = False
                except FileNotFoundError:
                    validation_message = "LinkML validation failed: 'linkml-validate' command not found. Make sure LinkML is installed and in the system PATH."
                    logger.error(validation_message)
                    validation_successful = False
                    break
                except Exception as val_e:
                    validation_message = f"An unexpected error occurred during LinkML validation: {val_e}"
                    logger.error(validation_message)
                    validation_successful = False
                    break
            finally:
                # Clean up temporary files
                try:
                    Path(schema_path).unlink()
                    Path(output_path).unlink()
                except Exception as e:
                    logger.debug(f"Error cleaning up temporary files: {e}")

    elif output_format == "phenopackets-json":
        validation_retry_delay = 2  # seconds

        for val_attempt in range(max_retries):
            validation_stats["validation_attempts"] += 1
            logger.info(
                "Starting Phenopacket validation attempt %d of %d.",
                val_attempt + 1,
                max_retries,
            )

            # Create temporary file for validation
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as output_file:
                output_file.write(corrected_output_content)
                output_path = output_file.name

            try:
                if not corrected_output_content.strip():
                    validation_message = (
                        "Phenopacket validation failed: Output content is empty."
                    )
                    logger.error(validation_message)
                    validation_successful = False
                    break

                if not Path(PHENOPACKET_TOOLS_JAR_PATH).exists():
                    validation_message = f"Phenopacket validator JAR not found at {PHENOPACKET_TOOLS_JAR_PATH}. Cannot perform validation."
                    logger.error(validation_message)
                    validation_successful = False
                    break

                validation_cmd = [
                    "java",
                    "-jar",
                    PHENOPACKET_TOOLS_JAR_PATH,
                    "validate",
                    "-f",  # Format specification
                    "json",
                    "-e",  # Element type to validate
                    "cohort",
                    output_path,
                ]
                logger.info(
                    f"Running Phenopacket validation command: {' '.join(validation_cmd)}"
                )

                process = subprocess.run(
                    validation_cmd, capture_output=True, text=True, check=False
                )

                raw_stdout = process.stdout
                raw_stderr = process.stderr
                cli_message_output = (raw_stdout if raw_stdout else raw_stderr).strip()

                logger.debug(f"Phenopacket validator raw stdout:\n{raw_stdout}")
                logger.debug(f"Phenopacket validator raw stderr:\n{raw_stderr}")
                logger.debug(f"Phenopacket validator return code: {process.returncode}")
                logger.debug(
                    f"Phenopacket validator interpreted CLI message: '{cli_message_output}'"
                )

                if process.returncode == 0 and not cli_message_output:
                    validation_message = (
                        "Phenopacket validation successful! No issues found."
                    )
                    logger.info(validation_message)
                    validation_successful = True
                    validation_stats["final_validation_success"] = True
                    if val_attempt == 0:
                        validation_stats["initial_validation_success"] = True
                    break
                else:
                    error_detail = cli_message_output
                    if not error_detail and process.returncode != 0:
                        error_detail = "Validator exited with a non-zero status but provided no specific error message via stdout/stderr."
                    elif not error_detail and process.returncode == 0:
                        error_detail = "Validator exited with status 0 but produced unexpected output, indicating a potential issue."

                    error_message = f"Phenopacket validation failed with return code {process.returncode}.\nValidator output:\n{error_detail}"
                    validation_message = f"Phenopacket validation attempt {val_attempt + 1} failed. Error:\n{error_message}"
                    logger.error(validation_message)

                    validation_stats["total_retries"] = val_attempt + 1

                    if val_attempt < max_retries - 1:
                        validation_stats["llm_correction_attempts"] += 1
                        logger.info("Attempting to fix Phenopacket using LLM...")
                        fix_prompt = (
                            f"{initial_full_prompt}\n\n"
                            f"--- PREVIOUSLY GENERATED SCHEMA ---\n{corrected_schema_content}\n"
                            f"--- PREVIOUSLY GENERATED PHENOPACKET OUTPUT ---\n{corrected_output_content}\n\n"
                            f"--- VALIDATION ERROR ---\n{error_message}\n\n"
                            f"--- INSTRUCTION ---\n"
                            f"The previous Phenopacket generation resulted in the validation error above. "
                            f"Please analyze the schema (if any), the Phenopacket output, and the error message. "
                            f"Fix the issues in the Phenopacket output and, if necessary, the schema. "
                            f"Provide the corrected full schema content after a 'SCHEMA' line and the corrected full Phenopacket JSON output content after an 'OUTPUT' line."
                        )
                        logger.debug("LLM Fix Prompt (Phenopacket):\n%s", fix_prompt)

                        try:
                            result = run_agent_workflow(
                                fix_prompt,
                                debug=debug,
                                use_umls=use_umls,
                                use_search=use_search,
                            )
                            logger.info(
                                "Phenopacket correction attempt workflow completed."
                            )
                            new_schema_content, new_output_content = (
                                split_schema_and_output(result)
                            )

                            if not new_output_content:
                                logger.warning(
                                    f"Phenopacket correction attempt {val_attempt + 1} did not produce output content. Retrying validation with previous content."
                                )
                            else:
                                if new_schema_content:
                                    try:
                                        json.loads(
                                            new_schema_content
                                        )  # Check if valid JSON
                                        corrected_schema_content = new_schema_content
                                        logger.info(
                                            "Updated corrected Phenopacket schema content"
                                        )
                                    except json.JSONDecodeError as e:
                                        logger.warning(
                                            f"LLM correction produced invalid JSON schema for Phenopacket: {e}. Keeping previous schema."
                                        )
                                else:
                                    logger.warning(
                                        "LLM correction did not produce Phenopacket schema content. Keeping previous schema."
                                    )

                                corrected_output_content = new_output_content
                                logger.info(
                                    "Updated corrected Phenopacket output content"
                                )

                        except Exception as llm_e:
                            logger.error(
                                f"LLM Correction attempt {val_attempt + 1} for Phenopacket failed during workflow: {llm_e}"
                            )

                        logger.info(
                            f"Retrying Phenopacket validation in {validation_retry_delay} seconds..."
                        )
                        time.sleep(validation_retry_delay)
                    else:
                        validation_message = f"Phenopacket validation failed after {max_retries} attempts."
                        logger.error(validation_message)
                        validation_successful = False

            except FileNotFoundError as fnf_e:
                validation_message = f"Phenopacket validation failed due to FileNotFoundError: {fnf_e}. Ensure Java is installed and PHENOPACKET_TOOLS_JAR_PATH is correct."
                logger.error(validation_message)
                validation_successful = False
                break
            except Exception as val_e:
                validation_message = f"An unexpected error occurred during Phenopacket validation: {val_e}"
                logger.error(validation_message)
                validation_successful = False
                break
            finally:
                # Clean up temporary file
                try:
                    Path(output_path).unlink()
                except Exception as e:
                    logger.debug(f"Error cleaning up temporary file: {e}")

    else:
        # For formats that don't have validation implemented
        validation_successful = True
        validation_message = f"Validation not implemented for format: {output_format}"
        validation_stats["final_validation_success"] = True
        validation_stats["initial_validation_success"] = True
        logger.info(validation_message)

    return (
        corrected_schema_content,
        corrected_output_content,
        validation_successful,
        validation_message,
        validation_stats,
    )


def enhance_content_with_umls(
    schema_content: str,
    output_content: str,
    output_format: str,
    original_input: str,
    debug: bool = False,
) -> Tuple[str, str]:
    """Enhance generated content with UMLS mappings."""
    logger.info("Enhancing content with UMLS mappings...")

    try:
        # Check if UMLS server is accessible
        umls_client = get_umls_client()
        if not umls_client.health_check():
            logger.warning("UMLS server is not accessible. Skipping UMLS enhancement.")
            return schema_content, output_content

        # Extract clinical concepts from the original input
        clinical_concepts = extract_clinical_concepts(original_input)

        # Get UMLS mappings
        umls_mappings = {}
        for concept in clinical_concepts:
            logger.debug(f"Mapping clinical concept: {concept}")

            # Search in HPO first
            hpo_results = umls_client.search_terms(concept, "HPO", limit=1)
            if hpo_results:
                umls_mappings[concept] = {
                    "code": hpo_results[0].code,
                    "term": hpo_results[0].term,
                    "ontology": "HPO",
                }
            else:
                # Try SNOMED CT as fallback
                snomed_results = umls_client.search_terms(
                    concept, "SNOMEDCT_US", limit=1
                )
                if snomed_results:
                    umls_mappings[concept] = {
                        "code": snomed_results[0].code,
                        "term": snomed_results[0].term,
                        "ontology": "SNOMEDCT_US",
                    }

        # Add UMLS mappings to content
        enhanced_schema = add_umls_mappings_to_content(
            schema_content, umls_mappings, output_format, "schema"
        )
        enhanced_output = add_umls_mappings_to_content(
            output_content, umls_mappings, output_format, "output"
        )

        logger.info(
            f"Successfully mapped {len(umls_mappings)} clinical concepts using UMLS"
        )

        return enhanced_schema, enhanced_output

    except Exception as e:
        logger.error(f"Error during UMLS enhancement: {e}")
        logger.warning("Continuing without UMLS enhancement")
        return schema_content, output_content


def extract_clinical_concepts(input_content: str) -> list:
    """Extract clinical concepts from input content."""
    # Common clinical terms that are likely to appear in medical data
    clinical_terms = [
        "BMI",
        "blood pressure",
        "systolic",
        "diastolic",
        "cholesterol",
        "age",
        "sex",
        "gender",
        "smoking",
        "diabetes",
        "hypertension",
        "asthma",
        "COPD",
        "coronary artery disease",
        "hyperlipidemia",
        "obesity",
        "heart disease",
        "cancer",
        "stroke",
        "myocardial infarction",
        "chronic kidney disease",
        "liver disease",
        "depression",
        "anxiety",
        "sleep apnea",
    ]

    found_concepts = []
    input_lower = input_content.lower()

    # Look for exact matches
    for term in clinical_terms:
        if term.lower() in input_lower:
            found_concepts.append(term)

    # Look for common medical history patterns
    if "medical history" in input_lower or "medicalhistory" in input_lower:
        # Extract conditions from medical history fields
        import re

        history_patterns = [
            r"coronary artery disease",
            r"type 2 diabetes",
            r"diabetes mellitus",
            r"hypertension",
            r"hyperlipidemia",
            r"asthma",
            r"copd",
            r"chronic obstructive pulmonary disease",
        ]

        for pattern in history_patterns:
            if re.search(pattern, input_lower):
                found_concepts.append(pattern.replace(r"\s+", " "))

    # Remove duplicates and return
    return list(set(found_concepts))


def add_umls_mappings_to_content(
    content: str, mappings: dict, output_format: str, content_type: str
) -> str:
    """Add UMLS mappings to content as comments."""
    if not mappings:
        return content

    # Create UMLS mapping comment section
    mapping_comments = []
    mapping_comments.append("# UMLS Concept Mappings")
    mapping_comments.append(
        "# Generated automatically from clinical concepts in the input data"
    )
    mapping_comments.append("#")

    for original_term, mapping in mappings.items():
        comment = f"# {original_term} → {mapping['code']} ({mapping['term']}) [{mapping['ontology']}]"
        mapping_comments.append(comment)

    mapping_comments.append("#")

    # Add mappings at the top of the content
    mapping_section = "\n".join(mapping_comments) + "\n\n"

    # For LinkML, add to both schema and output
    if output_format == "linkml":
        return mapping_section + content

    # For Phenopackets, add to schema only to avoid breaking JSON structure
    elif output_format == "phenopackets-json":
        if content_type == "schema":
            return mapping_section + content
        else:
            # For JSON output, we can't add comments, so just return original
            return content

    # For CSV, add as header comments
    elif output_format == "phenopackets-csv":
        if content_type == "schema":
            return mapping_section + content
        else:
            return mapping_section + content

    return content


def process_file(
    input_path: str,
    output_path: str,
    schema_path: str,
    user_prompt: str,
    output_format: str,
    debug: bool = False,
    validate: bool = False,
    max_retries: int = 10,
    output_stats: bool = False,
    use_umls: bool = False,
    use_search: bool = False,
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
    system_prompt = get_system_prompt(output_format, use_umls)

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
            result = run_agent_workflow(
                current_prompt, debug=debug, use_umls=use_umls, use_search=use_search
            )
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

    # --- UMLS Enhancement (if requested) ---
    if use_umls:
        logger.info("Performing UMLS enhancement...")
        schema_content, output_content = enhance_content_with_umls(
            schema_content, output_content, output_format, input_content, debug
        )
        logger.info("UMLS enhancement completed")

    # --- Validation and Correction (if requested) ---
    if validate:
        logger.info("Running validation...")
        (
            corrected_schema_content,
            corrected_output_content,
            validation_successful,
            validation_message,
            validation_stats,
        ) = validate_output(
            schema_content,
            output_content,
            output_format,
            initial_full_prompt,
            debug,
            max_retries,
            use_umls,
            use_search,
        )

        if not validation_successful:
            raise ValueError(f"Validation failed: {validation_message}")
        else:
            logger.info(f"Validation completed successfully: {validation_message}")
            # Use the potentially corrected content
            schema_content = corrected_schema_content
            output_content = corrected_output_content

        # Output benchmark statistics if requested
        if output_stats:
            stats_json = json.dumps(validation_stats)
            print(f"BENCHMARK_STATS:{stats_json}")
    else:
        logger.info("Validation not requested, skipping validation step.")

        # Output benchmark statistics if requested (no validation case)
        if output_stats:
            no_validation_stats = {
                "initial_validation_success": True,  # No validation means "success"
                "total_retries": 0,
                "final_validation_success": True,
                "validation_attempts": 0,
                "llm_correction_attempts": 0,
            }
            stats_json = json.dumps(no_validation_stats)
            print(f"BENCHMARK_STATS:{stats_json}")

    # Save the final content to files
    if schema_content:
        save_output_file(schema_content, schema_path, output_format)
        logger.info("Saved schema to: %s", schema_path)
    else:
        logger.warning("No schema was generated, but continuing since output exists")
        Path(schema_path).touch()  # Create empty file if no schema

    save_output_file(output_content, output_path, output_format)
    logger.info("Saved output to: %s", output_path)

    logger.info("Processing finished successfully.")


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
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Output benchmark statistics for retry tracking",
    )
    parser.add_argument(
        "--use-umls",
        action="store_true",
        dest="use_umls",
        help="Enable UMLS integration for clinical concept mapping",
    )
    parser.add_argument(
        "--use-search",
        "-use-search",
        action="store_true",
        dest="use_search",
        help="Enable Tavily search. By default, search is disabled and the workflow stays offline.",
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
            args.stats,
            args.use_umls,
            args.use_search,
        )
        print(f"Successfully processed {args.input}")
        print(f"Schema saved to: {args.schema}")
        print(f"Output saved to: {args.output}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise


if __name__ == "__main__":
    main()
