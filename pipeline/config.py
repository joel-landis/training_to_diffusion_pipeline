import tomllib
from pathlib import Path

SITE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "site_config.toml"


def deep_merge(base, override):
    """Copy of base with override merged in; nested dicts are merged recursively, other values replaced. Neither input is modified."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_stage_settings(pipeline_config, stage_config):
    """Settings of one [[stage]]: its presets deep-merged in order, then the stage's own keys on top, without "presets"."""
    merged_settings = {}
    for preset_name in stage_config.get("presets", []):
        merged_settings = deep_merge(
            merged_settings, pipeline_config["preset"][preset_name])
    return deep_merge(merged_settings, {key: value for key, value in stage_config.items() if key != "presets"})


def load_site_config():
    """Machine-specific settings from site_config.toml in the repository root; raises if the file does not exist."""
    if not SITE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{SITE_CONFIG_PATH} not found; copy site_config.template.toml to site_config.toml and adapt it")
    return tomllib.loads(SITE_CONFIG_PATH.read_text())
