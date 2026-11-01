import os
import sys

from ase.io import read, write

from pipeline import vasp_jobs
from pipeline.checks import raise_on_problems
from pipeline.config import standalone_settings, write_merged_settings
from pipeline.io_utils import write_json
from pipeline.stage_specs import STAGES, unique_split_names


def run(settings, output_dir, inputs):
    """Relax frames with VASP, one SLURM job per frame; requires ionic convergence.

    Per split, frames marked info["relaxed"] (e.g. by relax_mlip) are replaced by their VASP-relaxed structure; if none
    are marked, relaxed copies of all frames are appended. Returns one <name>_<split>.traj per input.
    """
    merged_settings, output_prefix = write_merged_settings(
        STAGES["relax_vasp"], settings, output_dir, inputs)
    split_names = unique_split_names(inputs)

    frames_by_split, frames_to_relax = {}, []
    for split_path, split in zip(inputs, split_names):
        frames_by_split[split] = read(split_path, index=":")
        marked_indices = [frame_index for frame_index, frame in enumerate(
            frames_by_split[split]) if "relaxed" in frame.info]
        if not marked_indices:
            n_original_frames = len(frames_by_split[split])
            frames_by_split[split] += [frame.copy()
                                       for frame in frames_by_split[split]]
            marked_indices = list(
                range(n_original_frames, 2 * n_original_frames))
        frames_to_relax += [(split, frame_index, frames_by_split[split]
                             [frame_index]) for frame_index in marked_indices]
    frame_dirs, job_id_by_dir, failed_state_by_dir = vasp_jobs.run_vasp_frames(merged_settings, output_dir, frames_to_relax,
                                                                               vasp_jobs.RELAX_CONVERGED_MARKER, True)

    status_rows = []
    for (split, frame_index, _), frame_dir in zip(frames_to_relax, frame_dirs):
        if frame_dir in failed_state_by_dir:
            status_rows.append([split, frame_index, job_id_by_dir[frame_dir], failed_state_by_dir[frame_dir], "failed",
                                vasp_jobs.failure_reason(frame_dir, job_id_by_dir[frame_dir], failed_state_by_dir[frame_dir])])
            continue
        relaxed_atoms = read(os.path.join(
            frame_dir, "OUTCAR"), format="vasp-out", index=-1)
        # relaxation energies are not kept; labels come from label_vasp
        relaxed_atoms.calc = None
        relaxed_atoms.info["relaxed"] = "vasp"
        frames_by_split[split][frame_index] = relaxed_atoms
        status_rows.append(
            [split, frame_index, job_id_by_dir[frame_dir], "COMPLETED", "converged", ""])
    vasp_jobs.write_status(merged_settings, output_prefix, ["split", "frame_index", "job_id", "state", "status", "reason"],
                           status_rows, failed_state_by_dir)

    outputs = []
    for split in split_names:
        write(f"{output_prefix}_{split}.traj", frames_by_split[split])
        outputs.append(f"{output_prefix}_{split}.traj")
        print(
            f"{merged_settings['name']}: {split} {len(frames_by_split[split])} frames -> {output_prefix}_{split}.traj", flush=True)
    write_json(os.path.join(output_dir, "outputs.json"), outputs)
    return outputs


if __name__ == "__main__":
    settings, inputs = standalone_settings(
        sys.argv[1], "relax_vasp", sys.argv[2:])
    raise_on_problems("relax_vasp", settings, inputs)
    run(settings, ".", inputs)
