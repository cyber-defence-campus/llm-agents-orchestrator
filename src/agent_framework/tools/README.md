# Tool Management in Agent Core

This document explains how tools are managed within the `src/agent_framework/tools/` directory, including creation, deletion, and potential methods for restriction.

## Tool Availability

A tool is considered available to an agent if it meets two primary conditions:
1.  It is defined within a subdirectory of `src/agent_framework/tools/`.
2.  It is explicitly imported into the main `src/agent_framework/tools/__init__.py` file.

The overall set of tools loaded is dynamically controlled by the `AGENT_SANDBOX_MODE`, which dictate conditional imports in `src/agent_framework/tools/__init__.py`. Agents receive the full list of tools registered for their specific execution environment (default or sandbox).

## How to Create a New Tool

To create a new tool (e.g., `file_reader`), follow these steps:

1.  **Create the Tool Directory and Files**:
    Create a new folder: `src/agent_framework/tools/file_reader/`.
    Inside this folder, create three essential files:
    *   `actions.py`: Contains the Python implementation of your tool functions.
    *   `tool_def.xml`: Provides the tool's description, parameters, and examples for the agent (LLM).
    *   `__init__.py`: Marks the directory as a Python package and exposes the tool functions.

2.  **Write the Tool Logic (`actions.py`)**:
    Define your tool function(s) and decorate them with `@register_tool`. The `sandbox_execution` flag (defaulting to `True`) indicates if the tool requires a restricted environment.

    ```python
    # src/agent_framework/tools/file_reader/actions.py
    from agent_framework.tools import register_tool
    from typing import Any

    @register_tool(sandbox_execution=True)
    def read_file(file_path: str, max_lines: int = 100) -> dict[str, Any]:
        """Reads a file from the filesystem."""
        # ... (your implementation) ...
        return {"content": "Example file content...", "lines_read": 10}
    ```

3.  **Define the Agent Schema (`tool_def.xml`)**:
    This XML file is crucial for the agent to understand how to use your tool. Provide a clear description, define all parameters with their types and requirements, and include concrete examples.

    ```xml
    <!-- src/agent_framework/tools/file_reader/tool_def.xml -->
    <tools>
      <tool name="read_file">
        <description>Reads content from a specified file.</description>
        <parameters>
          <parameter name="file_path" type="string" required="true">
            <description>The path to the file to read.</description>
          </parameter>
          <parameter name="max_lines" type="integer" required="false">
            <description>Maximum number of lines to read. Defaults to 100.</description>
          </parameter>
        </parameters>
        <returns type="Dict[str, Any]">
          <description>A dictionary containing the file content.</description>
        </returns>
        <examples>
          <function call="read_file">
            <parameter name="file_path">/tmp/example.txt</parameter>
            <parameter name="max_lines">50</parameter>
          </function>
        </examples>
      </tool>
    </tools>
    ```

4.  **Expose the Tool (`__init__.py`)**:
    In the tool's `__init__.py`, import the function(s) from `actions.py` and add them to `__all__`.

    ```python
    # src/agent_framework/tools/file_reader/__init__.py
    from .actions import read_file

    __all__ = ["read_file"]
    ```

5.  **Register the Tool Module**:
    Finally, import your new tool module into the top-level `src/agent_framework/tools/__init__.py`. Place it within the appropriate conditional block (e.g., `if not SANDBOX_MODE:`).

    ```python
    # src/agent_framework/tools/__init__.py (excerpt)
    if not SANDBOX_MODE:
        # ... existing imports ...
        from .file_reader import * # Add this line
    ```

## How to Delete a Tool

To delete a tool:

1.  **Unregister the Tool Module**: Remove the corresponding import line from `src/agent_framework/tools/__init__.py`.
2.  **Delete the Tool Directory**: Delete the entire tool's directory (e.g., `src/agent_framework/tools/file_reader/`).

## How to Restrict Tools When Creating a Job (Future Enhancement)

Currently, there is no built-in mechanism to restrict tools on a per-job basis; tools are loaded globally based on environment variables. Implementing this would require modifications to the core system:

1.  **Modify Tool Prompt Generation**: The `get_tools_prompt` function in `src/agent_framework/tools/registry.py` would need to be updated to accept an optional list of allowed or disallowed tool modules. This would filter the tools presented to the agent.
2.  **Update Agent Logic**: The `BaseAgent` class would need to be modified to read a new configuration field (e.g., `allowed_tool_modules`) from the `job_config` and pass it to the `get_tools_prompt` function.
3.  **Pass Configuration at Job Creation**: The orchestrating service or user interface would need to provide the desired `allowed_tool_modules` as part of the job configuration when initiating a new job.
