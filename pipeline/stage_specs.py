"""Fan-out of the stages whose items run separately, e.g. as one SLURM job each; readable without importing stages.py.

ITEMS[stage](settings, inputs) -> [(label, item settings, item inputs)]; each item runs in <stage dir>/<label>.
"""
import os


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


ITEMS = {
    "generate_cluster": one_item_per_cluster,
    "run_md": one_item_per_md_input,
    "opes_minimum": one_item_per_input,
    "analyze": one_item_per_input,
}
