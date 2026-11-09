"""Settings checks of every stage, run before anything executes.

The MACE foundation model names and the VASP INCAR tags come from the installed mace-torch and ase.
"""
import inspect
import os

from ase.calculators.vasp import create_input
from mace.calculators.foundations_models import mace_mp_names

from pipeline import stages
from pipeline.config import load_site_config
from pipeline.stage_specs import WORK_UNITS
from pipeline.vasp_jobs import VASP_DEFAULTS, VASP_REQUIRED

# every INCAR tag ASE writes; VASP ignores a misspelled tag without warning
KNOWN_INCAR_TAGS = {tag.upper() for tag_list in (create_input.float_keys, create_input.exp_keys, create_input.string_keys,
                                                 create_input.int_keys, create_input.list_int_keys, create_input.list_bool_keys,
                                                 create_input.list_float_keys, create_input.special_keys, create_input.dict_keys,
                                                 create_input.bool_keys)
                    for tag in tag_list}
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


def missing_files(paths, earlier_stage_names):
    """The paths that do not exist; the name of an earlier stage stands for that stage's output and is not checked."""
    return [path for path in paths if path not in earlier_stage_names and not os.path.exists(path)]


def stage_parameters(task):
    """({setting: default or REQUIRED} of the task's function in stages.py, name of its ** parameter or None);
    None if stages.py defines no such function."""
    stage_function = getattr(stages, task, None)
    if not inspect.isfunction(stage_function) or stage_function.__module__ != stages.__name__:
        return None
    parameters = inspect.signature(stage_function).parameters.values()
    parameter_defaults = {parameter.name: REQUIRED if parameter.default is parameter.empty else parameter.default
                          for parameter in parameters if parameter.kind is parameter.KEYWORD_ONLY}
    backend_parameter = next(
        (parameter.name for parameter in parameters if parameter.kind is parameter.VAR_KEYWORD), None)
    return parameter_defaults, backend_parameter


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


def mace_model_problems(settings, earlier_stage_names):
    """A model that is neither a MACE foundation model name nor a URL must be an existing file."""
    model = settings.get("model")
    if model is None or str(model).startswith("https:") or model in mace_mp_names:
        return []
    return [f"file not found: {path}" for path in missing_files([model], earlier_stage_names)]


def run_md_problems(settings, earlier_stage_names):
    plumed_templates = settings.get("plumed")
    if isinstance(plumed_templates, dict):
        plumed_templates = list(plumed_templates.values())
    elif plumed_templates is not None:
        plumed_templates = [plumed_templates]
    return mace_model_problems(settings, earlier_stage_names) + [f"file not found: {path}" for path in missing_files(plumed_templates or [], earlier_stage_names)]


def select_frames_problems(settings, earlier_stage_names):
    exclude_provenance = settings.get("exclude_provenance")
    return mace_model_problems(settings, earlier_stage_names) + [f"file not found: {path}" for path in
                                                     missing_files([exclude_provenance] if exclude_provenance else [], earlier_stage_names)]


def vasp_problems(settings, earlier_stage_names):
    """INCAR tags that ASE does not know (VASP ignores misspelled tags), kpoints not {mesh, gamma}, missing POTCARs."""
    problems = []
    unknown_incar_tags = {tag.upper() for tag in settings.get(
        "incar", {})} - KNOWN_INCAR_TAGS
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
            f"file not found: {path}" for path in missing_files(potcar_files, earlier_stage_names)]
    return problems


def analyze_diffusion_problems(settings, earlier_stage_names):
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


def plot_diffusion_problems(settings, earlier_stage_names):
    return [f"highlights entry {highlight} must have exactly the keys {sorted(HIGHLIGHT_KEYS)}"
            for highlight in settings.get("highlights", []) if set(highlight) != HIGHLIGHT_KEYS] + [
        f"file not found: {path}" for path in missing_files(list(settings.get("coordination_files", {}).values()),
                                                            earlier_stage_names)]


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


def check_stage(task, settings, input_files, earlier_stage_names=()):
    """Problems with one stage: missing input files, then the keys and rules of every work unit; empty if the stage can run.

    input_files are the stage's inputs that are files, not names of earlier stages.
    """
    if stage_parameters(task) is None:
        return [f"unknown task {task!r}; it is not a function of {stages.__file__}"]
    problems = [
        f"file not found: {path}" for path in missing_files(input_files, earlier_stage_names)]
    unit_settings_list = [settings]
    if task in WORK_UNITS:
        try:
            unit_settings_list = [unit_settings for _, unit_settings, _ in WORK_UNITS[task](
                settings, input_files)] or [settings]
        except (KeyError, ValueError) as error:
            return problems + [f"cannot split into work units: {type(error).__name__}: {error}"]
    for unit_settings in unit_settings_list:
        problems += check_work_unit(task, unit_settings, earlier_stage_names)
    return problems


def check_work_unit(task, unit_settings, earlier_stage_names):
    """Problems with the settings of one work unit: its keys and the stage's rules."""
    problems = key_problems(task, unit_settings)
    if task in STAGE_RULES:
        problems += STAGE_RULES[task](unit_settings, earlier_stage_names)
    return problems
