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
from pipeline.stage_specs import WORK_UNITS


def run_work_unit(task, settings, unit_dir, inputs):
    """Run one work unit of a stage in this process; writes its settings record and, when it succeeded, outputs.json."""
    stage_function = getattr(stages, task)
    stage_arguments = {key: value for key,
                       value in settings.items() if key != "slurm"}
    # the record holds every setting the stage runs with, defaults included
    bound_arguments = inspect.signature(stage_function).bind(
        unit_dir, inputs, **stage_arguments)
    bound_arguments.apply_defaults()
    write_json(os.path.join(unit_dir, f"{settings['name']}_settings.json"),
               {key: value for key, value in bound_arguments.arguments.items() if key != "output_dir"})
    outputs = stage_function(unit_dir, inputs, **stage_arguments)
    write_json(os.path.join(unit_dir, "outputs.json"), outputs)
    return outputs


def split_stages(pipeline_config):
    """(task, settings, inputs) of every [[stage]], with its presets merged in; inputs is None if the stage has none."""
    pipeline_stages = []
    for stage_config in pipeline_config["stage"]:
        stage_settings = merge_stage_settings(pipeline_config, stage_config)
        task = stage_settings.pop("task")
        pipeline_stages.append(
            (task, stage_settings, stage_settings.pop("inputs", None)))
    return pipeline_stages


def check_pipeline(pipeline_stages, run_dir):
    """Check every stage before anything runs and raise with the full list of problems."""
    run_dir = os.path.abspath(run_dir)
    problems = []
    earlier_stage_names = []
    for stage_index, (task, stage_settings, explicit_inputs) in enumerate(pipeline_stages):
        stage_label = f"[{stage_index:02d}_{stage_settings.get('name')}]"
        referenced_stage_names = [
            entry for entry in explicit_inputs or [] if entry in earlier_stage_names]
        input_files = [
            entry for entry in explicit_inputs or [] if entry not in earlier_stage_names]
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
    """Run all work units of one stage, as SLURM jobs if settings has "slurm", otherwise in this process.

    Work units run in <stage_dir>/<label>; work units with an outputs.json are skipped. Raises if any work unit failed, otherwise writes
    <stage_dir>/outputs.json with the outputs of all work units and returns them.
    """
    # one (label, settings, inputs, directory) per work unit; a stage without fan-out is one unit in stage_dir, label None
    if task in WORK_UNITS:
        work_units = []
        for unit_label, unit_settings, unit_inputs in WORK_UNITS[task](settings, inputs):
            unit_dir = os.path.join(stage_dir, unit_label)
            if not os.path.exists(unit_dir):
                os.makedirs(unit_dir)
                write_json(os.path.join(unit_dir, "stage_input.json"), {
                           "task": task, "settings": unit_settings, "inputs": unit_inputs})
            work_units.append(
                (unit_label, unit_settings, unit_inputs, unit_dir))
    else:
        work_units = [(None, settings, inputs, stage_dir)]

    failure_reason_by_unit_dir = {}
    if "slurm" in settings:
        for unit_label, unit_settings, _, unit_dir in work_units:
            if not os.path.exists(os.path.join(unit_dir, "job.sh")):
                job_name = settings["name"] if unit_label is None else f"{settings['name']}_{unit_label}"
                write_text(os.path.join(unit_dir, "job.sh"),
                           slurm_jobs.slurm_job_script(unit_settings, unit_dir, job_name))
        unit_dirs = [unit_dir for *_, unit_dir in work_units]
        poll_interval_s = settings["slurm"].get("poll_interval_s", 60)
        _, failed_job_state_by_dir = slurm_jobs.submit_and_wait_for_jobs(
            settings["name"], unit_dirs, poll_interval_s)
        failure_reason_by_unit_dir = {unit_dir: f"SLURM job {job_state}, see {unit_dir}/slurm_*.err"
                                      for unit_dir, job_state in failed_job_state_by_dir.items()}
    else:
        for _, unit_settings, unit_inputs, unit_dir in work_units:
            if os.path.exists(os.path.join(unit_dir, "outputs.json")):
                continue
            try:
                run_work_unit(task, unit_settings, unit_dir, unit_inputs)
            except Exception as error:
                failure_reason_by_unit_dir[unit_dir] = f"{type(error).__name__}: {error}"

    for unit_dir, failure_reason in failure_reason_by_unit_dir.items():
        print(f"WARNING: {unit_dir} failed: {failure_reason}",
              file=sys.stderr, flush=True)
    if failure_reason_by_unit_dir:
        raise RuntimeError(
            f"{settings['name']}: {len(failure_reason_by_unit_dir)} of {len(work_units)} work units failed")
    stage_outputs = [output_path for *_, unit_dir in work_units
                     for output_path in json.loads(Path(unit_dir, "outputs.json").read_text())]
    write_json(os.path.join(stage_dir, "outputs.json"), stage_outputs)
    return stage_outputs


def run_pipeline(pipeline_config):
    """Run the [[stage]] list of a config in order, skipping, resuming or rerunning each stage; returns the last stage's outputs.

    A stage gets its "inputs", with names of earlier stages replaced by their outputs, else the previous stage's outputs.
    """
    pipeline_stages = split_stages(pipeline_config)
    check_pipeline(pipeline_stages, pipeline_config["run_dir"])
    previous_stage_outputs = []
    outputs_by_stage_name = {}
    # inputs are compared by path, not content: once a stage has run, every later stage must rerun
    earlier_stage_ran = False
    for stage_index, (task, stage_settings, explicit_inputs) in enumerate(pipeline_stages):
        # without "inputs" the stage takes the previous stage's outputs; in "inputs", the name of an earlier stage
        # stands for that stage's outputs and any other entry is a file path
        stage_inputs = previous_stage_outputs if explicit_inputs is None else [
            input_path for entry in explicit_inputs for input_path in outputs_by_stage_name.get(entry, [entry])]
        stage_dir = os.path.join(
            pipeline_config["run_dir"], f"{stage_index:02d}_{stage_settings['name']}")
        stage_label = f"[{stage_index:02d}_{stage_settings['name']}]"
        stage_outputs_file = os.path.join(stage_dir, "outputs.json")
        stage_input_file = os.path.join(stage_dir, "stage_input.json")
        current_stage_input = {"task": task,
                               "settings": stage_settings, "inputs": stage_inputs}
        recorded_stage_input = json.loads(
            Path(stage_input_file).read_text()) if os.path.exists(stage_input_file) else None
        if not earlier_stage_ran and recorded_stage_input == current_stage_input:
            if os.path.exists(stage_outputs_file):
                # already done: same settings and inputs as the recorded run, and that run finished
                stage_outputs = json.loads(
                    Path(stage_outputs_file).read_text())
                outputs_by_stage_name[stage_settings["name"]] = stage_outputs
                previous_stage_outputs = stage_outputs
                print(
                    f"{stage_label} {task} already done: {stage_outputs}", flush=True)
                continue
            # resuming: same settings and inputs, but the recorded run did not finish
            print(f"{stage_label} {task} resuming in {stage_dir}", flush=True)
        else:
            # new or changed: no record yet, different settings or inputs, or an earlier stage ran in this call
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
        run_work_unit(stage_input["task"], stage_input["settings"], sys.argv[2], stage_input["inputs"])
        sys.exit(0)
    print(
        f"finished: {run_pipeline(tomllib.loads(Path(sys.argv[1]).read_text()))}", flush=True)
