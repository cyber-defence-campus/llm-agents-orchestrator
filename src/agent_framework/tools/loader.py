import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)


def load_external_tools():
    """
    Loads tool packages from paths specified in AGENT_TOOL_PATHS environment variable.
    """
    paths_env = os.getenv("AGENT_TOOL_PATHS")
    if not paths_env:
        return

    logger.info(f"Loading external tools from: {paths_env}")

    paths = [p.strip() for p in paths_env.split(os.pathsep) if p.strip()]

    for base_path in paths:
        if not os.path.exists(base_path):
            logger.warning(f"Tool path does not exist: {base_path}")
            continue

        # Add parent directory to sys.path so subpackages can be imported
        parent_dir = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)

        if parent_dir and parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            logger.debug(f"Added to sys.path: {parent_dir}")

        # Find all subdirectories with __init__.py (packages)
        try:
            for entry in os.listdir(base_path):
                pkg_dir = os.path.join(base_path, entry)
                init_file = os.path.join(pkg_dir, "__init__.py")

                if os.path.isdir(pkg_dir) and os.path.exists(init_file):
                    pkg_name = f"{base_name}.{entry}"

                    try:
                        importlib.import_module(pkg_name)
                        logger.info(f"Loaded external tool package: {pkg_name}")
                    except Exception as e:
                        logger.error(
                            f"Failed to load external tool package {pkg_name}: {e}"
                        )
        except OSError as e:
            logger.error(f"Failed to list directory {base_path}: {e}")
