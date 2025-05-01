# Data Transformation CLI

This tool uses an AI agent workflow to transform input data (e.g., TOPMED format) into specified output formats like LinkML or Phenopackets. It includes an optional validation step for LinkML output.

## Prerequisites

- Python 3.8+
- `pip` for installing packages

## Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install required Python packages:**
    ```bash
    pip install pyyaml # Required for YAML handling
    ```

4.  **Install LinkML (if using validation):**
    The `--validate` feature for LinkML requires the `linkml` package, which provides the `linkml-validate` command-line tool.
    ```bash
    pip install linkml
    ```
    Ensure that the `linkml-validate` command is available in your system's PATH after installation.

## Usage

The main script is `src/cli.py`. You run it as a module from the root directory of the project.

### Command-Line Arguments

-   `input`: Path to the input data file.
-   `output`: Path where the transformed output data will be saved.
-   `--schema`: (Required) Path where the generated schema (if applicable for the format) will be saved.
-   `--format`: (Required) The target output format. Choices: `linkml`, `phenopackets-json`, `phenopackets-csv`.
-   `--validate`: (Optional) If included, runs a validation step *after* generating the output. Currently only supported for `--format linkml`. Requires `linkml-validate` to be installed and in the PATH.
-   `--prompt`: (Optional) Additional text to append to the system prompt given to the AI agent.
-   `--debug`: (Optional) Enables detailed debug logging.
-   `--max-retries`: (Optional, Default: 10) Maximum retries for the initial AI generation attempt.
-   `--retry-delay`: (Optional, Default: 2) Delay in seconds between initial generation retries.

### Example: Transforming to LinkML with Validation

This example takes `example-topmed.yaml` as input, transforms it to LinkML, saves the output to `example-linkml-output.yaml`, saves the generated schema to `example-linkml-schema.yaml`, and runs validation.

```bash
python -m src.cli example-topmed.yaml example-linkml-output.yaml --schema example-linkml-schema.yaml --format linkml --validate
```

## Validation (LinkML)

When using `--format linkml` and including the `--validate` flag, the script will perform an extra step after generating the `output` and `schema` files:

1.  It calls the external `linkml-validate` command.
2.  It automatically extracts the required `--target-class` from the generated schema file.
3.  If validation passes (`linkml-validate` reports "No issues found"), the process finishes successfully.
4.  If validation fails, the script will:
    *   Send the error message, the generated schema, and the generated output back to the AI agent.
    *   Instruct the agent to fix the issues based on the validation error and the LinkML rules.
    *   Retry validation with the corrected files.
    *   Repeat this correction loop up to 10 times. If it still fails, an error is raised.

### Running `linkml-validate` Manually

You can also run the validation tool directly if you have the `linkml` package installed. This is useful for testing or debugging schemas and instance data independently.

**Example:**

Assuming `example-linkml-schema.yaml` contains a root class named `Person` (or whatever the actual root class is defined as in your schema), you would run:

```bash
linkml-validate --schema example-linkml-schema.yaml --target-class Person example-linkml-output.yaml
```

Replace `Person` with the actual name of the root class defined in your schema file (usually found under the top-level `classes:` key).
