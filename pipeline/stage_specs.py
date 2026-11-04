"""Fan-out of the stages whose work units run separately, e.g. as one SLURM job each; readable without importing stages.py.

WORK_UNITS[stage](settings, inputs) -> [(label, work unit settings, work unit inputs)]; each work unit runs in <stage dir>/<label>.
"""
import os


def one_unit_per_input(settings, inputs):
    """One work unit per input file, labelled and named by the file name without extension; raises if two inputs give the same label."""
    unit_labels = [os.path.splitext(os.path.basename(input_path))[
        0] for input_path in inputs]
    if len(set(unit_labels)) != len(unit_labels):
        raise ValueError(
            f"inputs {inputs} give non-unique labels {unit_labels}")
    return [(unit_label, {**settings, "name": unit_label}, [input_path]) for unit_label, input_path in zip(unit_labels, inputs)]


def one_unit_per_cluster(settings, inputs):
    """One work unit per entry of settings["clusters"]; the entry's keys override the stage settings. inputs is ignored."""
    settings_without_clusters = {key: value for key,
                                 value in settings.items() if key != "clusters"}
    return [(cluster_label, {**settings_without_clusters, **cluster_overrides, "name": cluster_label}, [])
            for cluster_label, cluster_overrides in settings["clusters"].items()]


def one_unit_per_md_input(settings, inputs):
    """One work unit per input structure; a "plumed" table {label: template} gives each work unit its own template."""
    work_units = one_unit_per_input(settings, inputs)
    if isinstance(settings.get("plumed"), dict):
        for unit_label, unit_settings, _ in work_units:
            unit_settings["plumed"] = settings["plumed"][unit_label]
    return work_units


WORK_UNITS = {
    "generate_clusters": one_unit_per_cluster,
    "md": one_unit_per_md_input,
    "opes_minimum": one_unit_per_input,
    "analyze": one_unit_per_input,
}
