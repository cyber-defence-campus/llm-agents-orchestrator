from pathlib import Path

from jinja2 import Environment


import os


def get_available_prompt_modules() -> dict[str, list[str]]:
    prompts_dirs = [Path(__file__).parent]

    extra_paths_env = os.getenv("AGENT_PROMPT_PATHS", "")
    if extra_paths_env:
        for path in extra_paths_env.split(os.pathsep):
            if path:
                p = Path(path)
                if p.is_dir():
                    prompts_dirs.append(p)

    available_modules = {}

    for prompts_dir in prompts_dirs:
        for category_dir in prompts_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith("__"):
                category_name = category_dir.name
                if category_name not in available_modules:
                    available_modules[category_name] = []

                for file_path in category_dir.glob("*.jinja"):
                    module_name = file_path.stem
                    if module_name == "system_prompt":
                        continue
                    if module_name not in available_modules[category_name]:
                        available_modules[category_name].append(module_name)

    # Sort modules in each category
    for category in available_modules:
        available_modules[category].sort()

    return available_modules


def get_all_module_names() -> set[str]:
    all_modules = set()
    for category_modules in get_available_prompt_modules().values():
        all_modules.update(category_modules)
    return all_modules


def validate_module_names(module_names: list[str]) -> dict[str, list[str]]:
    available_modules = get_all_module_names()
    valid_modules = []
    invalid_modules = []

    for module_name in module_names:
        if module_name in available_modules:
            valid_modules.append(module_name)
        else:
            invalid_modules.append(module_name)

    return {"valid": valid_modules, "invalid": invalid_modules}


def generate_modules_description() -> str:
    available_modules = get_available_prompt_modules()

    if not available_modules:
        return "No prompt modules available"

    all_module_names = get_all_module_names()

    if not all_module_names:
        return "No prompt modules available"

    sorted_modules = sorted(all_module_names)
    modules_str = ", ".join(sorted_modules)

    description = f"List of prompt modules to load for this agent (max 3). Available modules: {modules_str}. "

    example_modules = sorted_modules[:2]
    if example_modules:
        example = f"Example: {', '.join(example_modules)} for specialized agent"
        description += example

    return description


def load_prompt_modules(
    module_names: list[str], jinja_env: Environment
) -> dict[str, str]:
    import logging

    logger = logging.getLogger(__name__)
    module_content = {}

    # All possible prompt roots
    prompts_dirs = [Path(__file__).parent]
    extra_paths_env = os.getenv("AGENT_PROMPT_PATHS", "")
    if extra_paths_env:
        for path in extra_paths_env.split(os.pathsep):
            if path:
                p = Path(path)
                if p.is_dir():
                    prompts_dirs.append(p)

    available_modules = get_available_prompt_modules()

    for module_name in module_names:
        try:
            module_path = None

            # Try to find which directory contains this module
            if "/" in module_name:
                module_rel_path = f"{module_name}.jinja"
                for p_dir in prompts_dirs:
                    if (p_dir / module_rel_path).exists():
                        module_path = module_rel_path
                        break
            else:
                for category, modules in available_modules.items():
                    if module_name in modules:
                        module_rel_path = f"{category}/{module_name}.jinja"
                        for p_dir in prompts_dirs:
                            if (p_dir / module_rel_path).exists():
                                module_path = module_rel_path
                                break
                        if module_path:
                            break

                if not module_path:
                    root_candidate = f"{module_name}.jinja"
                    for p_dir in prompts_dirs:
                        if (p_dir / root_candidate).exists():
                            module_path = root_candidate
                            break

            if module_path:
                template = jinja_env.get_template(module_path)
                var_name = module_name.split("/")[-1]
                try:
                    module_content[var_name] = template.render()
                    logger.info(f"Loaded prompt module: {module_name} -> {var_name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to pre-render prompt module {module_name} (likely due to missing context): {e}"
                    )
                    module_content[var_name] = ""
            else:
                logger.warning(f"Prompt module not found: {module_name}")

        except (FileNotFoundError, OSError, ValueError) as e:
            logger.warning(f"Failed to load prompt module {module_name}: {e}")

    return module_content
