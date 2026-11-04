import inspect
import json
import os
import sys
import tomllib
from pathlib import Path

from pipeline import slurm_jobs, stages
from pipeline.checks import check_stage
from pipeline.config import merge_stage_settings
from pipeline.io_utils import write_json, write_text
from pipeline.stage_specs import ITEMS


def run_stage_item(task, settings, item_dir, inputs):
    """Run one stage item in this process; writes its settings record and, when it succeeded, outputs.json."""
    stage_function = getattr(stages, task)
    stage_arguments = {key: value for key,
                       value in settings.items() if key != "slurm"}
    # the record holds every setting the stage runs with, defaults included
    bound_arguments = inspect.signature(stage_function).bind(
        item_dir, inputs, **stage_arguments)
    bound_arguments.apply_defaults()
    write_json(os.path.join(item_dir, f"{settings['name']}_settings.json"),
               {key: value for key, value in bound_arguments.arguments.items() if key != "output_dir"})
    outputs = stage_function(item_dir, inputs, **stage_arguments)
    write_json(os.path.join(item_dir, "outputs.json"), outputs)
    return outputs


def check_pipeline(pipeline_config):
    """Check every stage before anything runs and raise with the full list of problems."""
    run_dir = os.path.abspath(pipeline_config["run_dir"])
    problems = []
    earlier_stage_names = []
    for stage_index, stage_config in enumerate(pipeline_config["stage"]):
        stage_settings = merge_stage_settings(pipeline_config, stage_config)
        stage_label = f"[{stage_index:02d}_{stage_settings.get('name')}]"
        task = stage_settings.pop("task")
        explicit_inputs = stage_settings.pop("inputs", [])
        referenced_stage_names = [
            entry for entry in explicit_inputs if entry in earlier_stage_names]
        input_files = [
            entry for entry in explicit_inputs if entry not in earlier_stage_names]
        ambiguous_inputs = [
            entry for entry in referenced_stage_names if os.path.exists(entry)]
        if ambiguous_inputs:
            problems.append(
                f"{stage_label} inputs {ambiguous_inputs} are both earlier stage names and existing files")
        earlier_stage_names.append(stage_settings.get("name"))
        problems += [f"{stage_label} {problem}" for problem in check_stage(
            task, stage_settings, input_files, run_dir)]
    if problems:
        raise RuntimeError(
            "pipeline config check failed:\n" + "\n".join(problems))


def move_aside(stage_dir):
    """Rename stage_dir to the first free <stage_dir>_old_<n> and return the new path."""
    old_run_number = 1
    while os.path.exists(f"{stage_dir}_old_{old_run_number}"):
        old_run_number += 1
    os.rename(stage_dir, f"{stage_dir}_old_{old_run_number}")
    return f"{stage_dir}_old_{old_run_number}"


def run_stage(task, settings, stage_dir, inputs):
    """Run all items of one stage, as SLURM jobs if settings has "slurm", otherwise in this process.

    Fan-out items run in <stage_dir>/<label>; items with an outputs.json are skipped. Raises if any item failed, otherwise writes
    <stage_dir>/outputs.json with the outputs of all items and returns them.
    """
    if task in ITEMS:
        stage_items = []
        for item_label, item_settings, item_inputs in ITEMS[task](settings, inputs):
            item_dir = os.path.join(stage_dir, item_label)
            if not os.path.exists(item_dir):
                os.makedirs(item_dir)
                write_json(os.path.join(item_dir, "stage_input.json"), {
                           "task": task, "settings": item_settings, "inputs": item_inputs})
            stage_items.append(
                (item_label, item_settings, item_inputs, item_dir))
    else:
        stage_items = [(None, settings, inputs, stage_dir)]

    failure_reason_by_item_dir = {}
    if "slurm" in settings:
        for item_label, item_settings, _, item_dir in stage_items:
            if not os.path.exists(os.path.join(item_dir, "job.sh")):
                job_name = settings["name"] if item_label is None else f"{settings['name']}_{item_label}"
                write_text(os.path.join(item_dir, "job.sh"),
                           slurm_jobs.slurm_job_script(item_settings, item_dir, job_name))
        item_dirs = [item_dir for *_, item_dir in stage_items]
        poll_interval_s = settings["slurm"].get("poll_interval_s", 60)
        _, failed_job_state_by_dir = slurm_jobs.submit_and_wait_for_jobs(
            settings["name"], item_dirs, poll_interval_s)
        failure_reason_by_item_dir = {item_dir: f"SLURM job {job_state}, see {item_dir}/slurm_*.err"
                                      for item_dir, job_state in failed_job_state_by_dir.items()}
    else:
        for _, item_settings, item_inputs, item_dir in stage_items:
            if os.path.exists(os.path.join(item_dir, "outputs.json")):
                continue
            try:
                run_stage_item(task, item_settings, item_dir, item_inputs)
            except Exception as error:
                failure_reason_by_item_dir[item_dir] = f"{type(error).__name__}: {error}"

    for item_dir, failure_reason in failure_reason_by_item_dir.items():
        print(f"WARNING: {item_dir} failed: {failure_reason}",
              file=sys.stderr, flush=True)
    if failure_reason_by_item_dir:
        raise RuntimeError(
            f"{settings['name']}: {len(failure_reason_by_item_dir)} of {len(stage_items)} items failed")
    stage_outputs = [output_path for *_, item_dir in stage_items
                     for output_path in json.loads(Path(item_dir, "outputs.json").read_text())]
    write_json(os.path.join(stage_dir, "outputs.json"), stage_outputs)
    return stage_outputs


def run_pipeline(pipeline_config):
    """Run the [[stage]] list of a config in order, skipping, resuming or rerunning each stage; returns the last stage's outputs.

    A stage gets its "inputs", with names of earlier stages replaced by their outputs, else the previous stage's outputs.
    """
    check_pipeline(pipeline_config)
    previous_stage_outputs = []
    outputs_by_stage_name = {}
    # inputs are compared by path, not content: once a stage has run, every later stage must rerun
    earlier_stage_ran = False
    for stage_index, stage_config in enumerate(pipeline_config["stage"]):
        stage_settings = merge_stage_settings(pipeline_config, stage_config)
        task = stage_settings.pop("task")
        explicit_inputs = stage_settings.pop("inputs", None)
        # without "inputs" the stage takes the previous stage's outputs; in "inputs", the name of an earlier stage
        # stands for that stage's outputs and any other entry is a file path
        stage_inputs = previous_stage_outputs if explicit_inputs is None else [
            input_path for entry in explicit_inputs for input_path in outputs_by_stage_name.get(entry, [entry])]
        stage_dir = os.path.join(
            pipeline_config["run_dir"], f"{stage_index:02d}_{stage_settings['name']}")
        stage_label = f"[{stage_index:02d}_{stage_settings['name']}]"
        stage_outputs_file = os.path.join(stage_dir, "outputs.json")
        stage_input_file = os.path.join(stage_dir, "stage_input.json")
        # JSON round trip, so it compares equal to the stage_input.json read from disk
        current_stage_input = json.loads(json.dumps(
            {"task": task, "settings": stage_settings, "inputs": stage_inputs}))
        recorded_stage_input = json.loads(
            Path(stage_input_file).read_text()) if os.path.exists(stage_input_file) else None
        if not earlier_stage_ran and recorded_stage_input == current_stage_input:
            if os.path.exists(stage_outputs_file):
                stage_outputs = json.loads(
                    Path(stage_outputs_file).read_text())
                outputs_by_stage_name[stage_settings["name"]] = stage_outputs
                previous_stage_outputs = stage_outputs
                print(
                    f"{stage_label} {task} already done: {stage_outputs}", flush=True)
                continue
            print(f"{stage_label} {task} resuming in {stage_dir}", flush=True)
        else:
            if os.path.exists(stage_dir):
                print(
                    f"{stage_label} {task} settings or inputs changed, moved old run to {move_aside(stage_dir)}", flush=True)
            os.makedirs(stage_dir)
            write_json(stage_input_file, current_stage_input)
            print(f"{stage_label} {task} -> {stage_dir}", flush=True)
        earlier_stage_ran = True
        stage_outputs = run_stage(
            task, stage_settings, stage_dir, stage_inputs)
        outputs_by_stage_name[stage_settings["name"]] = stage_outputs
        previous_stage_outputs = stage_outputs
        print(f"{stage_label} done: {stage_outputs}", flush=True)
    return stage_outputs


if __name__ == "__main__":
    # entry point of the SLURM jobs written by slurm_jobs.slurm_job_script
    if sys.argv[1] == "--run-stage":
        stage_input = json.loads(
            Path(sys.argv[2], "stage_input.json").read_text())
        run_stage_item(stage_input["task"], stage_input["settings"], sys.argv[2], stage_input["inputs"])
        sys.exit(0)
    print(
        f"finished: {run_pipeline(tomllib.loads(Path(sys.argv[1]).read_text()))}", flush=True)
