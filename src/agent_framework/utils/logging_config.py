import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import ClassVar

from pythonjsonlogger import jsonlogger


class ColorFormatter(logging.Formatter):
    """A logging formatter that adds color to the output."""

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    format_str = "%(asctime)s - [%(levelname)s] - %(name)s:%(lineno)d - %(message)s"

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format_str)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["@timestamp"] = log_record.get("asctime")
        log_record["level"] = log_record.get("levelname")
        log_record["logger"] = log_record.get("name")
        if "asctime" in log_record:
            del log_record["asctime"]
        if "levelname" in log_record:
            del log_record["levelname"]
        if "name" in log_record:
            del log_record["name"]


def setup_logging() -> None:
    """
    Sets up logging for the application.
    - Creates the log directory if it doesn't exist.
    - Platform application logs are set to DEBUG level.
    - Logs are sent to a file (`logs/agent_framework_dev.log`).
    - Console logging (stdout) is enabled *only if AGENT_INTERACTIVE_MODE is not 'true'*.
    - Third-party library logs are silenced to reduce noise.
    """
    is_tty = sys.stdout.isatty()
    use_json_logs = os.getenv("USE_JSON_LOGS", "true").lower() == "true" and not is_tty

    noisy_loggers = [
        "httpcore",
        "httpx",
        "litellm",
        "docker",
        "urllib3",
        "gql",
        "websockets",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    agent_framework_logger = logging.getLogger("agent_framework")
    if agent_framework_logger.hasHandlers():
        agent_framework_logger.handlers.clear()

    agent_framework_logger.setLevel(logging.DEBUG)
    agent_framework_logger.propagate = False

    file_formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file_path = Path("logs/agent_framework_dev.log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    if use_json_logs:
        file_handler.setFormatter(
            CustomJsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
    else:
        file_handler.setFormatter(file_formatter)
    agent_framework_logger.addHandler(file_handler)

    is_tui_mode = os.getenv("AGENT_INTERACTIVE_MODE", "false").lower() == "true"
    console_handler = None
    if not is_tui_mode:
        console_handler = logging.StreamHandler(sys.stdout)
        if use_json_logs:
            console_handler.setFormatter(
                CustomJsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
            )
        elif is_tty:
            console_handler.setFormatter(ColorFormatter())
        else:
            console_handler.setFormatter(file_formatter)
        agent_framework_logger.addHandler(console_handler)

    loggers_to_configure = [
        "agent_worker",
        "sandbox_service",
        "agent_framework.services",
        "agent_framework.cli",
        "agent_framework.state_manager",
        "agent_framework.tools",
        "agent_framework.llm",
        "agent_framework.agents",
        "agent_framework.cli.tracer",
    ]
    log_levels = {
        "agent_framework.cli.app": logging.INFO,
        "agent_framework.llm": logging.INFO,
        "agent_framework.tools": logging.INFO,
        "agent_framework.agents": logging.INFO,
        "agent_framework.cli.tracer": logging.INFO,
    }

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(file_handler)
        if console_handler:
            logger.addHandler(console_handler)

        logger.setLevel(log_levels.get(logger_name, logging.DEBUG))
        logger.propagate = False

    if not is_tui_mode:
        agent_framework_logger.info(
            f"Logging initialized. Platform logs will be written to {log_file_path} and console."
        )
    else:
        agent_framework_logger.info(
            f"Logging initialized for TUI. Platform logs will be written to {log_file_path}."
        )
