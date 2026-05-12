import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a small YAML or JSON config file."""
    if not path:
        return {}

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("YAML configs require PyYAML. Install it or use a JSON config.") from exc
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping at top level: {config_path}")
    return payload


def apply_overrides(config: Dict[str, Any], overrides: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    """Apply argparse-style overrides where None means keep the config value."""
    merged = dict(config)
    for key in keys:
        value = overrides.get(key)
        if value is not None:
            merged[key] = value
    return merged


def expand_path(value: Any) -> Optional[Path]:
    """Convert a path-like config value to an expanded Path, preserving None and empty strings."""
    if value is None or value == "":
        return None
    return Path(str(value)).expanduser()


def to_jsonable(value: Any) -> Any:
    """Convert common Python objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(dict(payload)), handle, ensure_ascii=False, indent=2)

