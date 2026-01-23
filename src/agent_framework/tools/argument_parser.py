import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Type, Union, get_args, get_origin

logger = logging.getLogger("agent.arg_parser")


class ArgumentConversionError(ValueError):
    """Raised when argument conversion fails."""

    pass


class ArgumentParser:
    """Helper to convert string-based arguments from LLM to proper Python types."""

    @staticmethod
    def parse(func: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts the values in `args` to match the type signatures of `func`.
        """
        sig = inspect.signature(func)
        parsed = {}

        for name, val in args.items():
            if name not in sig.parameters:
                parsed[name] = val
                continue

            param = sig.parameters[name]
            target_type = param.annotation

            # Skip if no type hint or None
            if target_type == inspect.Parameter.empty or val is None:
                parsed[name] = val
                continue

            # If expected type is str but we got something else (e.g. dict parsed by parser previously), keep it?
            # Actually we assume inputs are strings from XML/Regex parser usually
            if not isinstance(val, str):
                # Double check if we need to convert complex objects that were passed as non-strings?
                # Usually safely pass through
                parsed[name] = val
                continue

            try:
                parsed[name] = ArgumentParser._convert_value(val, target_type)
            except Exception as e:
                raise ArgumentConversionError(f"Failed to convert '{name}': {e}")

        return parsed

    @staticmethod
    def _convert_value(value: str, target: Type) -> Any:
        origin = get_origin(target)
        t_args = get_args(target)

        # Optional/Union handling
        if origin is Union:
            for t in t_args:
                if t is type(None):
                    continue
                try:
                    return ArgumentParser._convert_value(value, t)
                except:
                    continue
            return value  # Fallback

        # Primitives
        if target is bool:
            return value.lower() in ("true", "1", "yes", "enabled")
        if target is int:
            return int(value)
        if target is float:
            return float(value)
        if target is str:
            return value

        # Collections
        if target in (dict, list, Dict, List) or origin in (dict, list, Dict, List):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Comma-separated list fallback
                if target in (list, List) or origin in (list, List):
                    return [x.strip() for x in value.split(",")]
                return {}

        return value


def convert_arguments(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return ArgumentParser.parse(func, kwargs)
