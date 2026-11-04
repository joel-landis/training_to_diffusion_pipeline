import os
import sys

import numpy as np
from ase.io import read, write
from mace.calculators import mace_mp

from pipeline.mlip_dynamics import relax_with_bfgs
from pipeline.checks import raise_on_problems
from pipeline.config import standalone_settings, write_merged_settings
from pipeline.io_utils import unique_split_names, write_csv, write_json
from pipeline.stage_specs import STAGES, mace_calculator_kwargs


def run(settings, output_dir, inputs):
    """Relax a copy of every frame whose largest MACE force exceeds fmax, and append the copies to their split.

    Returns one <name>_<split>.traj per input: the original frames, then the relaxed copies marked info["relaxed"] = "mlip".
    Also writes <name>_relax.csv and the BFGS files in <output_dir>/<split>/.
    """
    merged_settings, output_prefix = write_merged_settings(
        STAGES["relax_mlip"], settings, output_dir, inputs)
    mace_calculator = mace_mp(**mace_calculator_kwargs(merged_settings))

    relax_rows, outputs = [], []
    for split_path, split in zip(inputs, unique_split_names(inputs)):
        input_frames = read(split_path, index=":")
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        relaxed_copies = []
        for frame_index, frame in enumerate(input_frames):
            frame_copy = frame.copy()
            frame_copy.calc = mace_calculator
            initial_max_force = np.linalg.norm(
                frame_copy.get_forces(), axis=1).max()
            if initial_max_force <= merged_settings["fmax"]:
                relax_rows.append(
                    [split, frame_index, initial_max_force, "", "", ""])
                continue
            bfgs_converged, bfgs_steps, final_max_force = relax_with_bfgs(
                frame_copy, mace_calculator, merged_settings["fmax"], merged_settings["opt_max_steps"],
                os.path.join(output_dir, split, f"{frame_index:05d}"), f"{merged_settings['name']} {split} {frame_index}")
            # without a calculator, write() stores no MACE energy and forces for the relaxed copy
            frame_copy.calc = None
            frame_copy.info["relaxed"] = "mlip"
            relaxed_copies.append(frame_copy)
            relax_rows.append([split, frame_index, initial_max_force,
                              bfgs_converged, bfgs_steps, final_max_force])
        write(f"{output_prefix}_{split}.traj", input_frames + relaxed_copies)
        outputs.append(f"{output_prefix}_{split}.traj")
        print(f"{merged_settings['name']}: {split} {len(input_frames)} frames + {len(relaxed_copies)} relaxed copies "
              f"-> {output_prefix}_{split}.traj", flush=True)

    write_csv(f"{output_prefix}_relax.csv", [
              "split", "frame_index", "initial_fmax", "converged", "steps", "final_fmax"], relax_rows)
    write_json(os.path.join(output_dir, "outputs.json"), outputs)
    return outputs


if __name__ == "__main__":
    settings, inputs = standalone_settings(
        sys.argv[1], "relax_mlip", sys.argv[2:])
    raise_on_problems("relax_mlip", settings, inputs)
    run(settings, ".", inputs)
