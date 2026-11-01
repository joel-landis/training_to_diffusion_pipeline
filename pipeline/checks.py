"""Settings checks of every stage, run before anything executes; no stage module is imported.

Version-dependent name lists (MACE foundation models, VASP INCAR tags) come from lookups.json, written by
python -m pipeline.make_lookups for the installed mace-torch and ase versions.
"""
import json
import os
from importlib.metadata import version
from pathlib import Path

from pipeline.config import load_site_config
from pipeline.stage_specs import STAGES

LOOKUPS_PATH = Path(__file__).resolve().parent / "lookups.json"
# keys every stage accepts in addition to its own; "name" is also required
GENERAL_SETTING_KEYS = {"name", "slurm"}
SLURM_KEYS = {"cpus_per_task", "mem_per_cpu", "time",
              "gpus", "sbatch_args", "poll_interval_s"}
DIFFUSION_SPECIES_KEYS = {"vacf_cutoff_ps", "vacf_epsilon",
                          "n_samples_per_block", "fit_start_ps", "fit_width_ps"}
HIGHLIGHT_KEYS = {"name", "group", "systems", "elements"}


def load_lookup(distribution):
    """Entry of lookups.json for a distribution; raises if the file is missing or was written for another installed version."""
    if not LOOKUPS_PATH.exists():
        raise FileNotFoundError(
            f"{LOOKUPS_PATH} not found; run python -m pipeline.make_lookups")
    lookup = json.loads(LOOKUPS_PATH.read_text())[distribution]
    installed_version = version(distribution)
    if lookup["version"] != installed_version:
        raise RuntimeError(f"{LOOKUPS_PATH} was written for {distribution} {lookup['version']}, installed is "
                           f"{installed_version}; run python -m pipeline.make_lookups")
    return lookup


def missing_files(paths, run_dir):
    """The paths that do not exist, except paths inside run_dir, which earlier stages may still write; run_dir None exempts nothing."""
    run_dir_absolute = None if run_dir is None else os.path.abspath(run_dir)

    def inside_run_dir(path):
        return run_dir_absolute is not None and os.path.commonpath([os.path.abspath(path), run_dir_absolute]) == run_dir_absolute
    return [path for path in paths if not os.path.exists(path) and not inside_run_dir(path)]


def key_problems(stage_spec, settings):
    """Unknown keys, missing required keys and unknown slurm keys."""
    allowed_keys = set(stage_spec["required"]) | set(
        stage_spec["defaults"]) | set(stage_spec["optional"]) | GENERAL_SETTING_KEYS
    problems = []
    unknown_keys = set(settings) - allowed_keys
    if unknown_keys:
        problems.append(f"unknown settings {sorted(unknown_keys)}")
    missing_keys = (set(stage_spec["required"]) | {"name"}) - set(settings)
    if missing_keys:
        problems.append(f"missing required settings {sorted(missing_keys)}")
    unknown_slurm_keys = set(settings.get("slurm", {})) - SLURM_KEYS
    if unknown_slurm_keys:
        problems.append(
            f"unknown slurm keys {sorted(unknown_slurm_keys)}; allowed {sorted(SLURM_KEYS)}")
    return problems


def mace_model_problems(settings, run_dir):
    """A model that is neither a MACE foundation model name nor a URL must be an existing file."""
    model = settings.get("model")
    if model is None or str(model).startswith("https:") or model in load_lookup("mace-torch")["mace_mp_names"]:
        return []
    return [f"file not found: {path}" for path in missing_files([model], run_dir)]


def run_md_problems(settings, run_dir):
    plumed_templates = settings.get("plumed")
    if isinstance(plumed_templates, dict):
        plumed_templates = list(plumed_templates.values())
    elif plumed_templates is not None:
        plumed_templates = [plumed_templates]
    return mace_model_problems(settings, run_dir) + [f"file not found: {path}" for path in missing_files(plumed_templates or [], run_dir)]


def select_frames_problems(settings, run_dir):
    exclude_provenance = settings.get("exclude_provenance")
    return mace_model_problems(settings, run_dir) + [f"file not found: {path}" for path in
                                                     missing_files([exclude_provenance] if exclude_provenance else [], run_dir)]


def vasp_problems(settings, run_dir):
    """INCAR tags that ASE does not know (VASP ignores misspelled tags), kpoints not {mesh, gamma}, missing POTCARs."""
    problems = []
    unknown_incar_tags = {tag.upper() for tag in settings.get(
        "incar", {})} - set(load_lookup("ase")["incar_tags"])
    if unknown_incar_tags:
        problems.append(f"INCAR tags {sorted(unknown_incar_tags)} are not known VASP tags "
                        f"(checked against ASE's lists in ase/calculators/vasp/create_input.py)")
    explicit_kpoints = settings.get("kpoints")
    if explicit_kpoints is not None and set(explicit_kpoints) != {"mesh", "gamma"}:
        problems.append(
            f"kpoints must have exactly the keys ['gamma', 'mesh'], got {sorted(explicit_kpoints)}")
    if settings.get("potcars"):
        potcar_dir = load_site_config()["vasp"]["potcar_dir"]
        potcar_files = [os.path.join(potcar_dir, potcar_subdir, "POTCAR")
                        for potcar_subdir in settings["potcars"].values()]
        problems += [
            f"file not found: {path}" for path in missing_files(potcar_files, run_dir)]
    return problems


def analyze_diffusion_problems(settings, run_dir):
    """Each species table and rigid_body_check must have exactly their expected keys."""
    problems = []
    for species, species_settings in settings.get("species", {}).items():
        if set(species_settings) != DIFFUSION_SPECIES_KEYS:
            problems.append(
                f"species.{species} must have exactly the keys {sorted(DIFFUSION_SPECIES_KEYS)}, got {sorted(species_settings)}")
    default_rigid_body_check = STAGES["analyze_diffusion"]["defaults"]["rigid_body_check"]
    rigid_body_check = settings.get(
        "rigid_body_check", default_rigid_body_check)
    if set(rigid_body_check) != set(default_rigid_body_check):
        problems.append(
            f"rigid_body_check must have exactly the keys {sorted(default_rigid_body_check)}, got {sorted(rigid_body_check)}")
    return problems


def plot_diffusion_problems(settings, run_dir):
    return [f"highlights entry {highlight} must have exactly the keys {sorted(HIGHLIGHT_KEYS)}"
            for highlight in settings.get("highlights", []) if set(highlight) != HIGHLIGHT_KEYS]


STAGE_RULES = {
    "generate_cluster": mace_model_problems,
    "run_md": run_md_problems,
    "select_frames": select_frames_problems,
    "relax_mlip": mace_model_problems,
    "relax_vasp": vasp_problems,
    "label_vasp": vasp_problems,
    "analyze_diffusion": analyze_diffusion_problems,
    "plot_diffusion": plot_diffusion_problems,
}


def check_stage(module_name, settings, input_files, run_dir=None):
    """Problems with one stage: missing input files, then the keys and rules of every item; empty if the stage can run.

    input_files are the stage's inputs that are files, not names of earlier stages. Files inside run_dir are not
    required to exist yet.
    """
    if module_name not in STAGES:
        return [f"unknown module {module_name!r}; known modules {sorted(STAGES)}"]
    stage_spec = STAGES[module_name]
    problems = [
        f"file not found: {path}" for path in missing_files(input_files, run_dir)]
    item_settings_list = [settings]
    if "items" in stage_spec:
        try:
            item_settings_list = [item_settings for _, item_settings, _ in stage_spec["items"](
                settings, input_files)] or [settings]
        except (KeyError, ValueError) as error:
            return problems + [f"cannot split into items: {type(error).__name__}: {error}"]
    for item_settings in item_settings_list:
        problems += check_item(module_name, item_settings, run_dir)
    return problems


def check_item(module_name, item_settings, run_dir):
    """Problems with the settings of one item: its keys and the stage's rules."""
    problems = key_problems(STAGES[module_name], item_settings)
    if module_name in STAGE_RULES:
        problems += STAGE_RULES[module_name](item_settings, run_dir)
    return problems


def raise_on_problems(module_name, settings, input_files):
    """Check a standalone run, which runs settings as one item without fan-out, and raise with all problems."""
    problems = [f"file not found: {path}" for path in missing_files(
        input_files, None)] + check_item(module_name, settings, None)
    if problems:
        raise ValueError(f"{module_name}: " + "; ".join(problems))
