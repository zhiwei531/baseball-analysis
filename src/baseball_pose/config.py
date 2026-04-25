"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only before dependencies are installed.
    yaml = None


DEFAULT_PIPELINE_STAGES = (
    "load_metadata",
    "sample_frames",
    "apply_roi",
    "apply_preprocessing",
    "estimate_pose",
    "postprocess_pose",
    "extract_features",
    "evaluate_metrics",
    "write_visualizations",
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated runtime configuration."""

    path: Path
    raw: dict[str, Any]

    @property
    def clip_ids(self) -> list[str]:
        return list(self.raw.get("dataset", {}).get("clip_ids", []))

    @property
    def condition_ids(self) -> list[str]:
        return list(self.raw.get("experiments", {}).get("default_conditions", []))

    @property
    def pipeline_stages(self) -> tuple[str, ...]:
        return DEFAULT_PIPELINE_STAGES


def load_config(path: str | Path) -> RuntimeConfig:
    """Load a YAML config and merge one optional parent config."""

    config_path = Path(path)
    data = _read_yaml(config_path)
    parent_ref = data.pop("extends", None)
    if parent_ref:
        parent_path = (config_path.parent / parent_ref).resolve()
        parent_data = _read_yaml(parent_path)
        data = _deep_merge(parent_data, data)

    config = RuntimeConfig(path=config_path, raw=data)
    validate_config(config)
    return config


def validate_config(config: RuntimeConfig) -> None:
    """Validate required top-level fields and referenced conditions."""

    required_sections = ("project", "dataset", "video", "pose", "postprocess", "conditions")
    missing = [section for section in required_sections if section not in config.raw]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    if not config.clip_ids:
        raise ValueError("Config must define at least one dataset.clip_ids entry.")

    if not config.condition_ids:
        raise ValueError("Config must define experiments.default_conditions.")

    conditions = config.raw.get("conditions", {})
    missing_conditions = [name for name in config.condition_ids if name not in conditions]
    if missing_conditions:
        raise ValueError(f"Experiment references undefined conditions: {missing_conditions}")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        if yaml is None:
            data = _read_simple_yaml(handle.read())
        else:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by this repository's configs.

    This fallback keeps the CLI usable before `pyyaml` is installed. It supports
    nested mappings, scalar lists, list items that are mappings, booleans,
    numbers, and simple inline lists such as `[8, 8]`.
    """

    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    parsed, index = _parse_yaml_block(lines, 0, 0)
    if index != len(lines):
        raise ValueError("Could not parse complete YAML file.")
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML value must be a mapping.")
    return parsed


def _parse_yaml_block(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index

    is_list = lines[index][1].startswith("- ")
    if is_list:
        values: list[Any] = []
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent < indent or not content.startswith("- "):
                break
            if line_indent > indent:
                raise ValueError(f"Unexpected indentation near: {content}")

            item_text = content[2:].strip()
            if not item_text:
                item, index = _parse_yaml_block(lines, index + 1, indent + 2)
                values.append(item)
                continue

            if ":" in item_text:
                key, raw_value = _split_key_value(item_text)
                item: dict[str, Any] = {key: _parse_scalar(raw_value)}
                index += 1
                if index < len(lines) and lines[index][0] > indent:
                    nested, index = _parse_yaml_block(lines, index, lines[index][0])
                    if isinstance(nested, dict):
                        item.update(nested)
                    else:
                        raise ValueError(f"List item mapping cannot merge non-mapping: {item_text}")
                values.append(item)
            else:
                values.append(_parse_scalar(item_text))
                index += 1
        return values, index

    values: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent or content.startswith("- "):
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation near: {content}")

        key, raw_value = _split_key_value(content)
        index += 1
        if raw_value == "":
            if index < len(lines) and lines[index][0] > indent:
                value, index = _parse_yaml_block(lines, index, lines[index][0])
            else:
                value = {}
        else:
            value = _parse_scalar(raw_value)
        values[key] = value
    return values, index


def _split_key_value(content: str) -> tuple[str, str]:
    if ":" not in content:
        raise ValueError(f"Expected key/value pair: {content}")
    key, value = content.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
