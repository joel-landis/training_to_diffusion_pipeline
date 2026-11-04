import itertools
import os

import numpy as np
from ase.data import covalent_radii
from ase.io.trajectory import Trajectory
from tqdm import tqdm

from pipeline.analysis_utils import cache_key, frames_from_ps, load_or_compute_cached, trajectory_cache_fields


def compute_coordination(trajectory_path, first_frame_index, stride, cutoff_factor, write_interval_fs):
    """Coordination numbers of every element pair, pair distances and distances to the centre of mass, every stride-th frame from
    first_frame_index on.

    Two atoms are neighbours within cutoff_factor * (sum of their covalent radii). cn_<neighbour>_<centre> is the mean
    number of neighbour atoms per centre atom in each frame, counts_<neighbour>_around_<centre> the count per centre atom,
    hist_* and states_* its distribution; dist_to_com_<element> in Å. distances_<A>_<B> (Å, per frame) holds every A-B pair
    of elements without C: for A = B the pairs i < j of np.triu_indices(n_A, 1), otherwise all A x B pairs, row-major.
    frame_interval_fs is the time between two stored frames.
    """
    trajectory = Trajectory(trajectory_path)
    if first_frame_index >= len(trajectory):
        raise ValueError(
            f"equilibration cut at frame {first_frame_index} but {trajectory_path} has {len(trajectory)} frames")
    frame_indices = range(first_frame_index, len(trajectory), stride)
    first_frame = trajectory[first_frame_index]
    symbols = np.array(first_frame.get_chemical_symbols())
    elements = sorted(set(symbols.tolist()))
    element_masks = {element: symbols == element for element in elements}
    masses = first_frame.get_masses()
    atom_cutoffs = cutoff_factor * covalent_radii[first_frame.numbers]
    pair_cutoffs = atom_cutoffs[:, None] + atom_cutoffs[None, :]
    metal_pairs = [(first_element, second_element) for first_element, second_element in itertools.combinations_with_replacement(elements, 2)
                   if "C" not in (first_element, second_element)]

    neighbor_counts = {(neighbor_element, center_element): np.zeros((len(frame_indices), element_masks[center_element].sum()), dtype=int)
                       for neighbor_element in elements for center_element in elements}
    distances_to_com = {element: np.zeros(
        (len(frame_indices), element_masks[element].sum())) for element in elements}
    pair_distances = {(first_element, second_element): []
                      for first_element, second_element in metal_pairs}
    for row_index, frame_index in enumerate(tqdm(frame_indices, desc="coordination")):
        frame = trajectory[frame_index]
        if frame.get_chemical_symbols() != symbols.tolist():
            raise ValueError(
                f"{trajectory_path}: atoms or their order changed at frame {frame_index}")
        all_distances = frame.get_all_distances(mic=True)
        connectivity = all_distances < pair_cutoffs
        np.fill_diagonal(connectivity, False)
        for neighbor_element, center_element in neighbor_counts:
            neighbor_counts[neighbor_element, center_element][row_index] = (
                connectivity[np.ix_(element_masks[center_element], element_masks[neighbor_element])].sum(axis=1))
        positions = frame.get_positions()
        distances = np.linalg.norm(
            positions - masses @ positions / masses.sum(), axis=1)
        for element in elements:
            distances_to_com[element][row_index] = distances[element_masks[element]]
        for first_element, second_element in metal_pairs:
            block = all_distances[np.ix_(
                element_masks[first_element], element_masks[second_element])]
            if first_element == second_element:
                block = block[np.triu_indices(len(block), 1)]
            pair_distances[first_element, second_element].append(
                block.reshape(-1).astype(np.float32))
    trajectory.close()

    result = {
        "traj_file": np.array(trajectory_path),
        "system_name": np.array(os.path.splitext(os.path.basename(trajectory_path))[0]),
        "first_frame_index": np.array(first_frame_index),
        "stride": np.array(stride),
        "base_cutoff_factor": np.array(cutoff_factor),
        "symbols": symbols,
        "unique_elements": np.array(elements),
        "n_frames": np.array(len(frame_indices)),
        "n_atoms": np.array(len(symbols)),
        "frame_interval_fs": np.array(stride * write_interval_fs),
    }
    for (first_element, second_element), frame_distances in pair_distances.items():
        result[f"distances_{first_element}_{second_element}"] = np.array(
            frame_distances)
    for element in elements:
        result[f"mask_{element}"] = element_masks[element]
        result[f"indices_{element}"] = np.flatnonzero(element_masks[element])
        result[f"n_{element}"] = np.array(element_masks[element].sum())
        result[f"dist_to_com_{element}"] = distances_to_com[element]
        result[f"dist_to_com_{element}_raw"] = distances_to_com[element].reshape(
            -1)
    for (neighbor_element, center_element), counts in neighbor_counts.items():
        pair_tag = f"{neighbor_element}_{center_element}"
        result[f"cn_{pair_tag}"] = counts.mean(axis=1)
        result[f"counts_{neighbor_element}_around_{center_element}"] = counts
        result[f"counts_{neighbor_element}_around_{center_element}_raw"] = counts.reshape(
            -1)
        histogram_counts = np.bincount(counts.reshape(-1))
        histogram_probabilities = histogram_counts / histogram_counts.sum()
        occupied = histogram_counts > 0
        result[f"hist_counts_{pair_tag}"] = histogram_counts
        result[f"hist_n_{pair_tag}"] = np.arange(histogram_counts.size)
        result[f"hist_prob_{pair_tag}"] = histogram_probabilities
        result[f"states_{pair_tag}"] = np.flatnonzero(occupied)
        result[f"state_fractions_{pair_tag}"] = histogram_probabilities[occupied]
    return result


def coordination_analysis(trajectory_path, *, write_interval_fs, equilibration_ps, cache_dir, stride, cutoff_factor):
    """Coordination analysis of one trajectory, computed once per trajectory and settings and kept in cache_dir.

    Returns <cache_dir>/coordination_<key>.npz.
    """
    trajectory_path = os.path.abspath(trajectory_path)
    os.makedirs(cache_dir, exist_ok=True)
    first_frame_index = frames_from_ps(equilibration_ps, write_interval_fs)
    coordination_cache_key = cache_key(**trajectory_cache_fields(trajectory_path), start_idx=first_frame_index, stride=stride,
                                       cutoff_factor=cutoff_factor, write_interval_fs=write_interval_fs)
    cache_path = os.path.join(
        cache_dir, f"coordination_{coordination_cache_key}.npz")
    load_or_compute_cached(cache_path, lambda: compute_coordination(trajectory_path, first_frame_index, stride, cutoff_factor,
                                                                    write_interval_fs))
    print(f"coordination of {trajectory_path} -> {cache_path}", flush=True)
    return cache_path
