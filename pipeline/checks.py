"""Settings checks of every stage, run before anything executes; stages.py is read as source, not imported.

Version-dependent name lists (MACE foundation models, VASP INCAR tags) come from lookups.json, written by
python -m pipeline.make_lookups for the installed mace-torch and ase versions.
"""
import ast
import json
import os
from importlib.metadata import version
from pathlib import Path

from pipeline.config import load_site_config
from pipeline.stage_specs import WORK_UNITS
from pipeline.vasp_jobs import VASP_DEFAULTS, VASP_REQUIRED

LOOKUPS_PATH = Path(__file__).resolve().parent / "lookups.json"
STAGES_PATH = Path(__file__).resolve().parent / "stages.py"
# keyword arguments of mace_mp the MACE stages pass on; mace_mp hands unknown keys silently to ASE's Calculator.set
MACE_CALCULATOR_KEYS = ("model", "device", "default_dtype",
                        "head", "dispersion", "damping", "dispersion_xc")
# keys of a stage's ** parameter, by its name: (required, optional)
BACKEND_KEYS = {"mace_settings": ((), MACE_CALCULATOR_KEYS),
                "vasp_settings": (VASP_REQUIRED, tuple(VASP_DEFAULTS))}
REQUIRED = object()
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


def stage_parameters(task):
    """({setting: default or REQUIRED} of the stage's entry in stages.py, name of its ** parameter or None),
    read from the source without importing it; None if stages.py has no such entry."""
    stage_functions = {node.name: node for node in ast.parse(STAGES_PATH.read_text()).body
                       if isinstance(node, ast.FunctionDef)}
    if task not in stage_functions:
        return None
    arguments = stage_functions[task].args
    parameter_defaults = {argument.arg: REQUIRED if default is None else ast.literal_eval(default)
                          for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)}
    return parameter_defaults, None if arguments.kwarg is None else arguments.kwarg.arg


def key_problems(task, settings):
    """Unknown keys, missing required keys and unknown slurm keys."""
    parameter_defaults, backend_parameter = stage_parameters(task)
    backend_required, backend_optional = BACKEND_KEYS.get(
        backend_parameter, ((), ()))
    allowed_keys = set(parameter_defaults) | set(
        backend_required) | set(backend_optional) | {"slurm"}
    problems = []
    unknown_keys = set(settings) - allowed_keys
    if unknown_keys:
        problems.append(f"unknown settings {sorted(unknown_keys)}")
    missing_keys = ({key for key, default in parameter_defaults.items() if default is REQUIRED}
                    | set(backend_required)) - set(settings)
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
    default_rigid_body_check = stage_parameters("analyze")[
        0]["rigid_body_check"]
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
    "generate_clusters": mace_model_problems,
    "md": run_md_problems,
    "select_frames": select_frames_problems,
    "relax_mlip": mace_model_problems,
    "relax_vasp": vasp_problems,
    "label_vasp": vasp_problems,
    "analyze": analyze_diffusion_problems,
    "plot": plot_diffusion_problems,
}


def check_stage(task, settings, input_files, run_dir=None):
    """Problems with one stage: missing input files, then the keys and rules of every work unit; empty if the stage can run.

    input_files are the stage's inputs that are files, not names of earlier stages. Files inside run_dir are not
    required to exist yet.
    """
    if stage_parameters(task) is None:
        return [f"unknown task {task!r}; it is not a function of {STAGES_PATH}"]
    problems = [
        f"file not found: {path}" for path in missing_files(input_files, run_dir)]
    unit_settings_list = [settings]
    if task in WORK_UNITS:
        try:
            unit_settings_list = [unit_settings for _, unit_settings, _ in WORK_UNITS[task](
                settings, input_files)] or [settings]
        except (KeyError, ValueError) as error:
            return problems + [f"cannot split into work units: {type(error).__name__}: {error}"]
    for unit_settings in unit_settings_list:
        problems += check_work_unit(task, unit_settings, run_dir)
    return problems


def check_work_unit(task, unit_settings, run_dir):
    """Problems with the settings of one work unit: its keys and the stage's rules."""
    problems = key_problems(task, unit_settings)
    if task in STAGE_RULES:
        problems += STAGE_RULES[task](unit_settings, run_dir)
    return problems
