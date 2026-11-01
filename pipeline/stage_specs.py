"""Settings and fan-out of every stage, readable without importing the stage modules.

Per stage: "required" keys, "defaults", "optional" keys allowed without a default, and an optional "items" fan-out hook
items(settings, inputs) -> [(label, item settings, item inputs)]; each item runs in <stage dir>/<label>.
"""
import os

MACE_CALCULATOR_KEYS = ("model", "device", "default_dtype",
                        "head", "dispersion", "damping", "dispersion_xc")

VASP_REQUIRED = ("incar", "potcars", "ntasks", "time")
VASP_DEFAULTS = {
    "kpoints": None,
    "sbatch_args": [],
    "mem_per_cpu": "2G",
    "sync_interval_s": 60,
    "poll_interval_s": 60,
}


def one_item_per_input(settings, inputs):
    """One item per input file, labelled and named by the file name without extension; raises if two inputs give the same label."""
    item_labels = [os.path.splitext(os.path.basename(input_path))[
        0] for input_path in inputs]
    if len(set(item_labels)) != len(item_labels):
        raise ValueError(
            f"inputs {inputs} give non-unique labels {item_labels}")
    return [(item_label, {**settings, "name": item_label}, [input_path]) for item_label, input_path in zip(item_labels, inputs)]


def one_item_per_cluster(settings, inputs):
    """One item per entry of settings["clusters"]; the entry's keys override the stage settings. inputs is ignored."""
    settings_without_clusters = {key: value for key,
                                 value in settings.items() if key != "clusters"}
    return [(cluster_label, {**settings_without_clusters, **cluster_overrides, "name": cluster_label}, [])
            for cluster_label, cluster_overrides in settings["clusters"].items()]


def one_item_per_md_input(settings, inputs):
    """One item per input structure; a "plumed" table {label: template} gives each item its own template."""
    stage_items = one_item_per_input(settings, inputs)
    if isinstance(settings.get("plumed"), dict):
        for item_label, item_settings, _ in stage_items:
            item_settings["plumed"] = settings["plumed"][item_label]
    return stage_items


def split_name(path):
    """Split name of a file: the text after the last "_" of the file name without extension (selection_train.traj -> train)."""
    return os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]


def unique_split_names(split_files):
    """Split name of each file, in input order; raises if two files give the same split name."""
    split_names = [split_name(split_file) for split_file in split_files]
    if len(set(split_names)) != len(split_names):
        raise ValueError(
            f"frame files {split_files} give non-unique split names {split_names}")
    return split_names


STAGES = {
    "generate_cluster": {
        "required": ("composition", "radius", "temperature_K", "timestep_fs", "taut_fs", "equilibration_nsteps",
                     "write_interval_fs", "log_interval_fs"),
        "defaults": {
            "c_c_separation": 3.5,
            "cov_scale": 0.8,
            "max_attempts": 50000,
            "vacuum": 10.0,
            "seed": None,
            "fmax": 0.1,
            "opt_max_steps": 200,
            "remove_rigid_body_interval_fs": 10.0,
        },
        "optional": MACE_CALCULATOR_KEYS,
        "items": one_item_per_cluster,
    },
    "run_md": {
        "required": ("temperature_K", "timestep_fs", "taut_fs", "nsteps", "write_interval_fs", "log_interval_fs"),
        "defaults": {
            "start_structure_format": None,
            "seed": None,
            "plumed": None,
            "remove_rigid_body_interval_fs": 10.0,
        },
        "optional": MACE_CALCULATOR_KEYS,
        "items": one_item_per_md_input,
    },
    "select_frames": {
        "required": ("frames_per_traj", "device"),
        "defaults": {
            "thin_to": 3000,
            "atoms_per_frame_factor": 5,
            "val_fraction": 0.2,
            "seed": 42,
            "exclude_provenance": None,
        },
        "optional": MACE_CALCULATOR_KEYS,
    },
    "relax_mlip": {
        "required": (),
        "defaults": {
            "fmax": 0.1,
            "opt_max_steps": 500,
        },
        "optional": MACE_CALCULATOR_KEYS,
    },
    "relax_vasp": {"required": VASP_REQUIRED, "defaults": VASP_DEFAULTS, "optional": ()},
    "label_vasp": {"required": VASP_REQUIRED, "defaults": VASP_DEFAULTS, "optional": ()},
    "finetune_mace": {"required": ("device", "mace_args"), "defaults": {}, "optional": ()},
    "opes_minimum": {
        "required": ("temperature_K", "colvar_stride"),
        "defaults": {
            "colvar_file": "COLVAR",
            "state_file": "STATE",
            "fes_bins": None,
        },
        "optional": (),
        "items": one_item_per_input,
    },
    "analyze_coordination": {
        "required": ("write_interval_fs", "equilibration_ps", "cache_dir"),
        "defaults": {
            "stride": 10,
            "cutoff_factor": 1.1,
        },
        "optional": (),
        "items": one_item_per_input,
    },
    "analyze_diffusion": {
        "required": ("write_interval_fs", "equilibration_ps", "cache_dir", "species"),
        "defaults": {
            "analysis_stride": 1,
            "use_global_msd": False,
            "breakpoints_ps": [],
            "rigid_body_check": {"every_n_frames": 500, "rel_tolerance": 1e-6},
        },
        "optional": (),
        "items": one_item_per_input,
    },
    "plot_diffusion": {
        "required": (),
        "defaults": {
            "regimes": {},
            "trend_regimes": {},
            "highlights": [],
            "coordination_files": {},
        },
        "optional": (),
    },
    "plot_structure": {
        "required": (),
        "defaults": {
            "coordination_files": {},
        },
        "optional": (),
    },
}


def mace_calculator_kwargs(settings):
    """Keyword arguments for mace_mp: the keys of MACE_CALCULATOR_KEYS present in settings."""
    return {key: settings[key] for key in MACE_CALCULATOR_KEYS if key in settings}
