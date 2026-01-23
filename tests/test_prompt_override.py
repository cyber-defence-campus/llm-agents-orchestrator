"""Tests for external prompt template override functionality.

These tests verify that the AGENT_PROMPT_PATHS environment variable correctly
enables external templates to override the default agent-orchestrator templates.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from jinja2 import Environment, FileSystemLoader


class TestPromptTemplateOverride:
    """Tests for the prompt template override mechanism."""

    def test_external_template_takes_priority(self):
        """Verify external templates override default templates with the same name."""
        with tempfile.TemporaryDirectory() as external_dir:
            # Create directory structure mimicking tactics prompts
            coordination_dir = Path(external_dir) / "coordination"
            coordination_dir.mkdir()

            # Create an external template that should override the default
            external_template = coordination_dir / "root_agent.jinja"
            external_template.write_text("EXTERNAL OVERRIDE CONTENT")

            # Simulate the search path construction from llm.py
            prompts_dir = (
                Path(__file__).parent.parent / "src" / "agent_framework" / "prompts"
            )

            search_paths = []

            # 1. External paths (should have priority)
            p = Path(external_dir)
            search_paths.append(str(p))
            # Add subdirectories from external paths
            for subdir in p.rglob(""):
                if subdir.is_dir() and not subdir.name.startswith("__"):
                    search_paths.append(str(subdir))

            # 2. Local paths
            search_paths.append(str(prompts_dir))
            for subdir in prompts_dir.rglob(""):
                if subdir.is_dir():
                    search_paths.append(str(subdir))

            loader = FileSystemLoader(search_paths)
            env = Environment(loader=loader)

            # Test: short name should resolve to external template
            template = env.get_template("root_agent.jinja")
            content = template.render()

            assert content == "EXTERNAL OVERRIDE CONTENT"
            assert str(external_template) == template.filename

    def test_external_template_with_full_path(self):
        """Verify external templates work with category/module path format."""
        with tempfile.TemporaryDirectory() as external_dir:
            coordination_dir = Path(external_dir) / "coordination"
            coordination_dir.mkdir()

            external_template = coordination_dir / "test_module.jinja"
            external_template.write_text("TEST MODULE CONTENT {{ var }}")

            # Simulate search path construction
            prompts_dir = (
                Path(__file__).parent.parent / "src" / "agent_framework" / "prompts"
            )

            search_paths = [str(external_dir)]
            for subdir in Path(external_dir).rglob(""):
                if subdir.is_dir() and not subdir.name.startswith("__"):
                    search_paths.append(str(subdir))
            search_paths.append(str(prompts_dir))

            loader = FileSystemLoader(search_paths)
            env = Environment(loader=loader)

            # Both formats should work
            template1 = env.get_template("coordination/test_module.jinja")
            template2 = env.get_template("test_module.jinja")

            assert template1.render(var="hello") == "TEST MODULE CONTENT hello"
            assert template2.render(var="world") == "TEST MODULE CONTENT world"

    def test_context_variables_available_at_top_level(self):
        """Verify context variables are accessible as top-level template variables."""
        with tempfile.TemporaryDirectory() as external_dir:
            # Create template that uses top-level variables
            template_file = Path(external_dir) / "test.jinja"
            template_file.write_text(
                """
{% if automatic %}AUTOMATIC MODE{% else %}MANUAL MODE{% endif %}
{% if aggressive %}AGGRESSIVE{% endif %}
"""
            )

            loader = FileSystemLoader([str(external_dir)])
            env = Environment(loader=loader)
            template = env.get_template("test.jinja")

            # Simulate render_params.update(agent_state.context)
            context = {"automatic": True, "aggressive": True}
            render_params = {"context": context}
            render_params.update(context)  # Unpack as top-level vars

            content = template.render(**render_params)

            assert "AUTOMATIC MODE" in content
            assert "AGGRESSIVE" in content

    def test_context_variables_false_by_default(self):
        """Verify missing context variables evaluate as false in templates."""
        with tempfile.TemporaryDirectory() as external_dir:
            template_file = Path(external_dir) / "test.jinja"
            template_file.write_text(
                """
{% if automatic %}AUTOMATIC{% else %}MANUAL{% endif %}
{% if aggressive %}AGGRESSIVE{% else %}CONSERVATIVE{% endif %}
"""
            )

            loader = FileSystemLoader([str(external_dir)])
            env = Environment(loader=loader)
            template = env.get_template("test.jinja")

            # No context provided - variables should be undefined (falsy)
            content = template.render()

            assert "MANUAL" in content
            assert "CONSERVATIVE" in content

    def test_default_template_used_when_no_external(self):
        """Verify default templates are used when no external override exists."""
        prompts_dir = (
            Path(__file__).parent.parent / "src" / "agent_framework" / "prompts"
        )

        # Only add local paths, no external paths
        search_paths = [str(prompts_dir)]
        for subdir in prompts_dir.rglob(""):
            if subdir.is_dir():
                search_paths.append(str(subdir))

        loader = FileSystemLoader(search_paths)
        env = Environment(loader=loader)

        # Should find the default root_agent.jinja
        template = env.get_template("root_agent.jinja")

        assert "coordination" in template.filename
        # Default template should contain generic content
        content = template.render(task="test", agent_hierarchy=[])
        assert "Root Agent" in content or "objective" in content.lower()

    def test_subdirectory_override_priority(self):
        """Verify subdirectory modules from external paths have priority over local."""
        with tempfile.TemporaryDirectory() as external_dir:
            # Create nested structure
            vuln_dir = Path(external_dir) / "vulnerabilities"
            vuln_dir.mkdir()

            (vuln_dir / "sql_injection.jinja").write_text("EXTERNAL SQL INJECTION")

            prompts_dir = (
                Path(__file__).parent.parent / "src" / "agent_framework" / "prompts"
            )

            # Build search paths with external subdirs first
            search_paths = []

            # External paths first
            search_paths.append(str(external_dir))
            for subdir in Path(external_dir).rglob(""):
                if subdir.is_dir() and not subdir.name.startswith("__"):
                    search_paths.append(str(subdir))

            # Then local paths
            search_paths.append(str(prompts_dir))
            for subdir in prompts_dir.rglob(""):
                if subdir.is_dir():
                    search_paths.append(str(subdir))

            loader = FileSystemLoader(search_paths)
            env = Environment(loader=loader)

            # Should find external version
            template = env.get_template("sql_injection.jinja")
            content = template.render()

            assert content == "EXTERNAL SQL INJECTION"
