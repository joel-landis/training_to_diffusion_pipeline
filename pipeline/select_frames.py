import os
import sys

import numpy as np
import pandas as pd
import torch
from ase.io import write
from ase.io.trajectory import Trajectory
from mace.calculators import mace_mp
from torch_geometric.nn import fps

from pipeline.checks import raise_on_problems
from pipeline.config import standalone_settings, write_merged_settings
from pipeline.io_utils import write_json
from pipeline.stage_specs import STAGES, mace_calculator_kwargs


def thin_trajectory(trajectory_path, thin_to):
    """Every len // thin_to-th frame of a trajectory, at most thin_to frames.

    Returns (original frame indices, frames).
    """
    trajectory = Trajectory(trajectory_path)
    thinning_stride = max(1, len(trajectory) // thin_to)
    kept_frame_indices = list(
        range(0, len(trajectory), thinning_stride))[:thin_to]
    kept_frames = [trajectory[frame_index]
                   for frame_index in kept_frame_indices]
    trajectory.close()
    return kept_frame_indices, kept_frames


def evaluate_with_mace(frames, mace_calculator):
    """MACE energy per atom, largest per-atom force and per-atom descriptors of each frame.

    Stored energies and forces are not used; mace_calculator must support get_descriptors.
    Returns (energies in eV/atom, max forces in eV/Å, one (n_atoms, n_features) float32 descriptor array per frame).
    """
    energies_per_atom, max_forces, atom_descriptors_per_frame = [], [], []
    for frame in frames:
        # copy() drops the calculator with the stored MD results, so the stage's model is evaluated
        frame_with_calculator = frame.copy()
        frame_with_calculator.calc = mace_calculator
        energies_per_atom.append(
            frame_with_calculator.get_potential_energy() / len(frame_with_calculator))
        max_forces.append(np.linalg.norm(
            frame_with_calculator.get_forces(), axis=1).max())
        atom_descriptors_per_frame.append(
            mace_calculator.get_descriptors(frame).astype(np.float32))
    return energies_per_atom, max_forces, atom_descriptors_per_frame


def pick_frames_by_fps(atom_descriptors_per_frame, n_frames, atoms_per_frame_factor, device, excluded_frame_indices):
    """Pick up to n_frames frames by farthest point sampling over the descriptors of all atoms of all frames.

    FPS samples atoms_per_frame_factor * n_frames atoms, since several can fall in one frame; excluded_frame_indices are never picked.
    Returns {frame index: index of the atom whose pick selected the frame}, in FPS order.
    """
    stacked_atom_descriptors = np.vstack(atom_descriptors_per_frame)
    frame_index_of_row = np.repeat(np.arange(len(atom_descriptors_per_frame)), [
                                   len(descriptors) for descriptors in atom_descriptors_per_frame])
    atom_index_of_row = np.concatenate(
        [np.arange(len(descriptors)) for descriptors in atom_descriptors_per_frame])
    n_atoms_to_sample = min(atoms_per_frame_factor *
                            n_frames, len(stacked_atom_descriptors))
    descriptor_tensor = torch.tensor(stacked_atom_descriptors, device=device)
    picked_atom_by_frame = {}
    # deterministic start at the first atom
    for fps_row in fps(descriptor_tensor, ratio=n_atoms_to_sample / len(stacked_atom_descriptors), random_start=False).cpu().numpy():
        frame_index = int(frame_index_of_row[fps_row])
        if frame_index in excluded_frame_indices or frame_index in picked_atom_by_frame:
            continue
        picked_atom_by_frame[frame_index] = int(atom_index_of_row[fps_row])
        if len(picked_atom_by_frame) == n_frames:
            break
    return picked_atom_by_frame


def run(settings, output_dir, inputs):
    """Select frames_per_traj frames per input trajectory by FPS on MACE descriptors and split them into train and val.

    Frames listed in exclude_provenance, the provenance CSV of an earlier selection, are not picked again.
    Returns [<name>_train.traj, <name>_val.traj]; also writes <name>_provenance.csv.
    """
    merged_settings, output_prefix = write_merged_settings(
        STAGES["select_frames"], settings, output_dir, inputs)
    mace_calculator = mace_mp(**mace_calculator_kwargs(merged_settings))

    excluded_trajectory_frames = set()
    if merged_settings["exclude_provenance"] is not None:
        previous_provenance = pd.read_csv(
            merged_settings["exclude_provenance"])
        excluded_trajectory_frames = set(
            zip(previous_provenance["traj_file"], previous_provenance["orig_index"]))

    selected_frames, provenance_records = [], []
    for trajectory_path in inputs:
        thinned_indices, thinned_frames = thin_trajectory(
            trajectory_path, merged_settings["thin_to"])
        energies_per_atom, max_forces, atom_descriptors_per_frame = evaluate_with_mace(
            thinned_frames, mace_calculator)
        excluded_frame_indices = {frame_index for frame_index, original_index in enumerate(thinned_indices)
                                  if (trajectory_path, original_index) in excluded_trajectory_frames}
        picked_atom_by_frame = pick_frames_by_fps(atom_descriptors_per_frame, merged_settings["frames_per_traj"],
                                                  merged_settings["atoms_per_frame_factor"], merged_settings["device"], excluded_frame_indices)
        if len(picked_atom_by_frame) < merged_settings["frames_per_traj"]:
            print(
                f"{trajectory_path}: only {len(picked_atom_by_frame)} of {merged_settings['frames_per_traj']} frames selected", flush=True)
        for frame_index, atom_index in picked_atom_by_frame.items():
            selected_frames.append(thinned_frames[frame_index])
            provenance_records.append({"traj_file": trajectory_path, "orig_index": thinned_indices[frame_index], "atom_index": atom_index,
                                       "energy_per_atom": energies_per_atom[frame_index], "max_force": max_forces[frame_index]})

    shuffled_order = np.random.default_rng(
        merged_settings["seed"]).permutation(len(selected_frames))
    n_val = int(len(shuffled_order) * merged_settings["val_fraction"])
    val_positions, train_positions = shuffled_order[:n_val], shuffled_order[n_val:]
    for position in val_positions:
        provenance_records[position]["split"] = "val"
    for position in train_positions:
        provenance_records[position]["split"] = "train"

    write(f"{output_prefix}_train.traj", [
          selected_frames[position] for position in train_positions])
    write(f"{output_prefix}_val.traj", [
          selected_frames[position] for position in val_positions])
    with open(f"{output_prefix}_provenance.csv", "w") as output_file:
        pd.DataFrame(provenance_records).to_csv(output_file, index=False)
        output_file.flush()
        os.fsync(output_file.fileno())
    print(
        f"selected {len(selected_frames)} frames: train {len(train_positions)}, val {len(val_positions)}", flush=True)
    outputs = [f"{output_prefix}_train.traj", f"{output_prefix}_val.traj"]
    write_json(os.path.join(output_dir, "outputs.json"), outputs)
    return outputs


if __name__ == "__main__":
    settings, inputs = standalone_settings(
        sys.argv[1], "select_frames", sys.argv[2:])
    raise_on_problems("select_frames", settings, inputs)
    run(settings, ".", inputs)
