import os
import sys

from ase.io import read, write

from pipeline import vasp_jobs
from pipeline.checks import raise_on_problems
from pipeline.config import standalone_settings, write_merged_settings
from pipeline.io_utils import unique_split_names, write_json
from pipeline.stage_specs import STAGES


def run(settings, output_dir, inputs):
    """VASP single point of every frame, one SLURM job per frame; requires SCF convergence.

    Returns one <name>_<split>.xyz per input: extxyz with info["dft_energy"] (energy σ->0) and arrays["dft_forces"],
    without constraints. Also writes <name>_<split>_constraints.traj (same frames with constraints) and <name>_status.csv.
    """
    merged_settings, output_prefix = write_merged_settings(
        STAGES["label_vasp"], settings, output_dir, inputs)
    split_names = unique_split_names(inputs)
    frames_to_label = [(split, frame_index, frame) for split_path, split in zip(inputs, split_names)
                       for frame_index, frame in enumerate(read(split_path, index=":"))]
    frame_dirs, job_id_by_dir, failed_state_by_dir = vasp_jobs.run_vasp_frames(merged_settings, output_dir, frames_to_label,
                                                                               vasp_jobs.SCF_CONVERGED_MARKER, False)

    labeled_frames_by_split = {split: [] for split in split_names}
    constrained_frames_by_split = {split: [] for split in split_names}
    status_rows = []
    for (split, frame_index, _), frame_dir in zip(frames_to_label, frame_dirs):
        if frame_dir in failed_state_by_dir:
            status_rows.append([split, frame_index, job_id_by_dir[frame_dir], failed_state_by_dir[frame_dir], "failed", "",
                                vasp_jobs.failure_reason(frame_dir, job_id_by_dir[frame_dir], failed_state_by_dir[frame_dir])])
            continue
        labeled_atoms = read(os.path.join(
            frame_dir, "OUTCAR"), format="vasp-out")
        # ASE's "energy" from an OUTCAR is the energy extrapolated to σ->0, not TOTEN
        labeled_atoms.info["dft_energy"] = labeled_atoms.get_potential_energy()
        # real forces on fixed atoms, not zeros
        labeled_atoms.arrays["dft_forces"] = labeled_atoms.get_forces(
            apply_constraint=False)
        # without a calculator, extxyz writes only the dft_* keys, not "energy" and "forces"
        labeled_atoms.calc = None
        constrained_frames_by_split[split].append(labeled_atoms.copy())
        # no constraints, so extxyz writes no move_mask column
        labeled_atoms.set_constraint()
        labeled_frames_by_split[split].append(labeled_atoms)
        status_rows.append([split, frame_index, job_id_by_dir[frame_dir],
                           "COMPLETED", "labeled", labeled_atoms.info["dft_energy"], ""])
    vasp_jobs.write_status(merged_settings, output_prefix, ["split", "frame_index", "job_id", "state", "status", "dft_energy", "reason"],
                           status_rows, failed_state_by_dir)

    outputs = []
    for split in split_names:
        with open(f"{output_prefix}_{split}.xyz", "w") as output_file:
            write(output_file, labeled_frames_by_split[split], format="extxyz")
            output_file.flush()
            os.fsync(output_file.fileno())
        outputs.append(f"{output_prefix}_{split}.xyz")
        write(f"{output_prefix}_{split}_constraints.traj",
              constrained_frames_by_split[split])
        print(
            f"{merged_settings['name']}: {split} {len(labeled_frames_by_split[split])} labeled frames -> {output_prefix}_{split}.xyz", flush=True)
    write_json(os.path.join(output_dir, "outputs.json"), outputs)
    return outputs


if __name__ == "__main__":
    settings, inputs = standalone_settings(
        sys.argv[1], "label_vasp", sys.argv[2:])
    raise_on_problems("label_vasp", settings, inputs)
    run(settings, ".", inputs)
