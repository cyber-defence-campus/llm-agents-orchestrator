import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class TestPromptTemplateOverride:
    def test_external_template_takes_priority(self):
        with tempfile.TemporaryDirectory() as external_dir:
            coordination_dir = Path(external_dir) / "coordination"
            coordination_dir.mkdir()

            external_template = coordination_dir / "root_agent.jinja"
            external_template.write_text("EXTERNAL OVERRIDE CONTENT")

            # Simulate the search path construction from llm.py
            prompts_dir = (
                Path(__file__).parent.parent / "src" / "agent_framework" / "prompts"
            )

            search_paths = []

            p = Path(external_dir)
            search_paths.append(str(p))
            for subdir in p.rglob(""):
                if subdir.is_dir() and not subdir.name.startswith("__"):
                    search_paths.append(str(subdir))

            search_paths.append(str(prompts_dir))
            for subdir in prompts_dir.rglob(""):
                if subdir.is_dir():
                    search_paths.append(str(subdir))

            loader = FileSystemLoader(search_paths)
            env = Environment(loader=loader)

            template = env.get_template("root_agent.jinja")
            content = template.render()

            assert content == "EXTERNAL OVERRIDE CONTENT"
            assert str(external_template) == template.filename

    def test_external_template_with_full_path(self):
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

            template1 = env.get_template("coordination/test_module.jinja")
            template2 = env.get_template("test_module.jinja")

            assert template1.render(var="hello") == "TEST MODULE CONTENT hello"
            assert template2.render(var="world") == "TEST MODULE CONTENT world"

    def test_context_variables_available_at_top_level(self):
        with tempfile.TemporaryDirectory() as external_dir:
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
            render_params.update(context)

            content = template.render(**render_params)

            assert "AUTOMATIC MODE" in content
            assert "AGGRESSIVE" in content

    def test_context_variables_false_by_default(self):
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

            content = template.render()

            assert "MANUAL" in content
            assert "CONSERVATIVE" in content

    def test_default_template_used_when_no_external(self):
        prompts_dir = (
            Path(__file__).parent.parent / "src" / "agent_framework" / "prompts"
        )

        search_paths = [str(prompts_dir)]
        for subdir in prompts_dir.rglob(""):
            if subdir.is_dir():
                search_paths.append(str(subdir))

        loader = FileSystemLoader(search_paths)
        env = Environment(loader=loader)

        template = env.get_template("root_agent.jinja")

        assert "coordination" in template.filename
        content = template.render(task="test", agent_hierarchy=[])
        assert "Root Agent" in content or "objective" in content.lower()

    def test_subdirectory_override_priority(self):
        with tempfile.TemporaryDirectory() as external_dir:
            vuln_dir = Path(external_dir) / "vulnerabilities"
            vuln_dir.mkdir()

            (vuln_dir / "sql_injection.jinja").write_text("EXTERNAL SQL INJECTION")

            prompts_dir = (
                Path(__file__).parent.parent / "src" / "agent_framework" / "prompts"
            )

            search_paths = []

            search_paths.append(str(external_dir))
            for subdir in Path(external_dir).rglob(""):
                if subdir.is_dir() and not subdir.name.startswith("__"):
                    search_paths.append(str(subdir))

            search_paths.append(str(prompts_dir))
            for subdir in prompts_dir.rglob(""):
                if subdir.is_dir():
                    search_paths.append(str(subdir))

            loader = FileSystemLoader(search_paths)
            env = Environment(loader=loader)

            template = env.get_template("sql_injection.jinja")
            content = template.render()

            assert content == "EXTERNAL SQL INJECTION"
