import os
import tomllib
from pathlib import Path

from pipeline.io_utils import write_json

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


def standalone_settings(config_path, module_name, command_line_inputs):
    """Settings and inputs of a standalone run: the one [[stage]] of module_name in a pipeline config, merged with its presets.

    The inputs are command_line_inputs, or the stage's inputs if none are given on the command line.
    """
    pipeline_config = tomllib.loads(Path(config_path).read_text())
    stage_configs = [stage_config for stage_config in pipeline_config["stage"]
                     if stage_config["module"] == module_name]
    if len(stage_configs) != 1:
        raise ValueError(
            f"{config_path}: a standalone {module_name} run needs exactly one [[stage]] of it, found {len(stage_configs)}")
    settings = merge_stage_settings(pipeline_config, stage_configs[0])
    settings.pop("module")
    stage_inputs = settings.pop("inputs", [])
    return settings, command_line_inputs or stage_inputs


def load_site_config():
    """Machine-specific settings from site_config.toml in the repository root; raises if the file does not exist."""
    if not SITE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{SITE_CONFIG_PATH} not found; copy site_config.template.toml to site_config.toml and adapt it")
    return tomllib.loads(SITE_CONFIG_PATH.read_text())


def write_merged_settings(stage_spec, settings, output_dir, inputs):
    """Merge a stage's settings over the defaults of its stage_specs entry and write <output_dir>/<name>_settings.json.

    Returns (merged settings, output prefix <output_dir>/<name>).
    """
    merged_settings = {**stage_spec["defaults"], **settings}
    output_prefix = os.path.join(output_dir, merged_settings["name"])
    write_json(f"{output_prefix}_settings.json", {
               **merged_settings, "inputs": inputs})
    return merged_settings, output_prefix
