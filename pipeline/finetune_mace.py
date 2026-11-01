import os
import subprocess
import sys

from pipeline.checks import raise_on_problems
from pipeline.config import standalone_settings, write_merged_settings
from pipeline.io_utils import write_json
from pipeline.stage_specs import STAGES, unique_split_names

RESERVED_MACE_ARGS = {"train_file", "valid_file", "device"}


def mace_command_line_flags(mace_args):
    """run_train arguments from the mace_args table: true -> bare --key, false -> omitted, anything else -> --key str(value).

    For MACE options parsed with str2bool (e.g. multiheads_finetuning) the value must be the string "True"/"False";
    a TOML boolean is only right for store_true options (e.g. restart_latest).
    """
    command_line_flags = []
    for key, value in mace_args.items():
        if value is True:
            command_line_flags.append(f"--{key}")
        elif value is not False:
            command_line_flags += [f"--{key}", str(value)]
    return command_line_flags


def run(settings, output_dir, inputs):
    """Fine-tune MACE with python -m mace.cli.run_train in output_dir.

    inputs: exactly one *_train and one *_val extxyz file. mace_args must contain "name"; train_file, valid_file and device
    are set here. Returns [<output_dir>/<mace_args.name>.model].
    """
    merged_settings, _ = write_merged_settings(
        STAGES["finetune_mace"], settings, output_dir, inputs)

    reserved_keys_given = RESERVED_MACE_ARGS & set(
        merged_settings["mace_args"])
    if reserved_keys_given:
        raise ValueError(
            f"mace_args {sorted(reserved_keys_given)} are set by the stage; use inputs and device")

    split_names = unique_split_names(inputs)
    if sorted(split_names) != ["train", "val"]:
        raise ValueError(
            f"inputs {inputs} must be exactly one _train and one _val file")
    train_file, valid_file = (os.path.abspath(
        inputs[split_names.index(split)]) for split in ("train", "val"))
    print(
        f"{merged_settings['name']}: train {train_file}, valid {valid_file}", flush=True)

    run_train_command = [sys.executable, "-m", "mace.cli.run_train", "--train_file", train_file, "--valid_file", valid_file,
                         "--device", merged_settings["device"], *mace_command_line_flags(merged_settings["mace_args"])]
    print(
        f"{merged_settings['name']}: {' '.join(run_train_command)}", flush=True)
    subprocess.run(run_train_command, cwd=output_dir, check=True)

    model_path = os.path.join(
        output_dir, f"{merged_settings['mace_args']['name']}.model")
    if not os.path.exists(model_path):
        raise RuntimeError(f"run_train finished without {model_path}")
    outputs = [model_path]
    write_json(os.path.join(output_dir, "outputs.json"), outputs)
    return outputs


if __name__ == "__main__":
    settings, inputs = standalone_settings(
        sys.argv[1], "finetune_mace", sys.argv[2:])
    raise_on_problems("finetune_mace", settings, inputs)
    run(settings, ".", inputs)
